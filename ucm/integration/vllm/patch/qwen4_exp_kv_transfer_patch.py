"""Connect CUDA Qwen3.8-Flash-Next QSA and GDN layers to UCM layerwise KV hooks.

QSA full attention is the row-save layer (indices 3, 7, 11, ... 47). Its
``forward`` is inlined by ``torch.compile``, so a wrapper on ``forward`` never
runs again after the warmup trace. The KV write itself is ``_run_qsa``, which
vLLM already marks ``@eager_break_during_capture``. The save hook has to sit
*inside* that decorator: breakable CUDA-graph replay calls only the inner
function, and a wrapper outside the decorator is recorded into the graph and
skipped on replay.

GDN reaches its cache update from ``_forward_core``, which is already a custom
op and still runs as Python.
"""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = "_ucm_kv_hooks_patched"
_LOGGED = set()
_OPS_READY = False


def _log_once(key, message: str, *args) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(message, *args)


def _layer_index(layer_name: str) -> str:
    parts = layer_name.split(".")
    if "layers" in parts:
        idx = parts.index("layers") + 1
        if idx < len(parts):
            return parts[idx]
    return "?"


def _attn_metadata(layer_name: str):
    """Return ``(metadata, reason)``. ``reason`` is ``ok`` when this layer is present."""
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else None
    if metadata is None:
        return None, "attn_metadata_none"
    if not isinstance(metadata, dict):
        return None, f"attn_metadata_type={type(metadata).__name__}"
    if layer_name not in metadata:
        sample = list(metadata)[:4]
        return None, f"layer_missing_in_attn_metadata sample={sample}"
    return metadata[layer_name], "ok"


def _row_save_state(connector, layer_name: str) -> str:
    inner = getattr(connector, "connector", connector)
    save_layers = getattr(inner, "row_save_layer", None)
    if not isinstance(save_layers, dict):
        return "row_save_layer_missing"
    if layer_name in save_layers.values():
        return "in_row_save_layer"
    sample = list(save_layers.values())[:4]
    return f"not_row_save_layer sample={sample}"


def _qsa_module(layer_name: str):
    from vllm.forward_context import get_forward_context

    layers = getattr(get_forward_context(), "no_compile_layers", None)
    if isinstance(layers, dict):
        return layers.get(layer_name)
    return None


def _qsa_connector(layer_name: str):
    """Return ``(connector, attn_metadata, reason)`` or ``(None, None, reason)``."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return None, None, "no_kv_transfer_group"
    if not is_v1_kv_transfer_group():
        return None, None, "not_v1_kv_transfer_group"
    connector = get_kv_transfer_group()
    if not connector.has_connector_metadata():
        return None, None, "no_connector_metadata"
    attn_metadata, meta_reason = _attn_metadata(layer_name)
    if attn_metadata is None:
        return None, None, meta_reason
    return connector, attn_metadata, "ok"


def _qsa_wait(layer_name: str) -> None:
    connector, _, reason = _qsa_connector(layer_name)
    if connector is None:
        _log_once(
            ("qsa-skip", "begin", layer_name, reason),
            "HLA qwen4 qsa op skip: phase=begin layer_idx=%s layer=%s reason=%s",
            _layer_index(layer_name),
            layer_name,
            reason,
        )
        return
    _log_once(
        ("qsa-begin", layer_name),
        "HLA qwen4 qsa op: phase=begin layer_idx=%s layer=%s",
        _layer_index(layer_name),
        layer_name,
    )
    connector.wait_for_layer_load(layer_name)


def _qsa_save(layer_name: str) -> None:
    connector, attn_metadata, reason = _qsa_connector(layer_name)
    if connector is None:
        _log_once(
            ("qsa-skip", "end", layer_name, reason),
            "HLA qwen4 qsa op skip: phase=end layer_idx=%s layer=%s reason=%s",
            _layer_index(layer_name),
            layer_name,
            reason,
        )
        return
    module = _qsa_module(layer_name)
    row_state = _row_save_state(connector, layer_name)
    _log_once(
        ("qsa-save", layer_name, row_state),
        "HLA qwen4 qsa op save_kv_layer: layer_idx=%s layer=%s row_save=%s",
        _layer_index(layer_name),
        layer_name,
        row_state,
    )
    connector.save_kv_layer(
        layer_name,
        getattr(module, "kv_cache", None) if module is not None else None,
        attn_metadata,
    )


def _register_qsa_ops() -> bool:
    """Custom ops stay in the compiled graph when ``_run_qsa`` is inlined."""
    global _OPS_READY
    if _OPS_READY:
        return True
    try:
        import torch
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        logger.warning("Skip Qwen4Exp QSA KV ops: vLLM custom-op helper is missing")
        return False

    def begin(layer_name: str, output: torch.Tensor) -> torch.Tensor:
        _qsa_wait(layer_name)
        return output

    def end(layer_name: str, output: torch.Tensor) -> torch.Tensor:
        _qsa_save(layer_name)
        return output

    def fake_begin(layer_name: str, output: torch.Tensor) -> torch.Tensor:
        return output

    def fake_end(layer_name: str, output: torch.Tensor) -> torch.Tensor:
        return output

    tags = ()
    if hasattr(torch, "Tag") and hasattr(torch.Tag, "cudagraph_unsafe"):
        tags = (torch.Tag.cudagraph_unsafe,)

    def _define(op_name: str, op_func, fake_impl) -> None:
        kwargs = {
            "op_name": op_name,
            "op_func": op_func,
            "mutates_args": ["output"],
            "fake_impl": fake_impl,
        }
        try:
            direct_register_custom_op(**kwargs, tags=tags)
        except TypeError:
            direct_register_custom_op(**kwargs)

    try:
        _define("ucm_qwen4_kv_layer_begin", begin, fake_begin)
        _define("ucm_qwen4_kv_layer_end", end, fake_end)
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "duplicate" in message or "exists" in message:
            _OPS_READY = True
            return True
        logger.warning("UCM Qwen4Exp QSA KV op registration failed: %s", exc)
        return False
    _OPS_READY = True
    return True


def _qsa_output(args, kwargs):
    import torch

    output = kwargs.get("output")
    if isinstance(output, torch.Tensor):
        return output
    # _run_qsa(projected_qk, positions, query, key, value, output, ...)
    if len(args) >= 6 and isinstance(args[5], torch.Tensor):
        return args[5]
    return None


def _eager_break_cell(fn):
    """Closure cell of ``@eager_break_during_capture`` that replay invokes.

    ``functools.wraps`` replaces ``__qualname__`` with ``_run_qsa``'s, so the
    wrapper has to be recognized by its code object instead.
    """
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or closure is None or "fn" not in code.co_freevars:
        return None
    filename = (code.co_filename or "").replace("\\", "/")
    if (
        "breakable_cudagraph" not in filename
        and "BreakableCUDAGraphCapture" not in code.co_names
    ):
        return None
    idx = code.co_freevars.index("fn")
    if idx >= len(closure):
        return None
    return closure[idx]


def _install_qsa_run_hook(layer_cls) -> bool:
    current = getattr(layer_cls, "_run_qsa", None)
    if not callable(current):
        raise RuntimeError(
            "UCM Qwen4Exp KV hooks require Qwen4ExpQSAAttention._run_qsa; "
            "check compatibility with the installed vLLM version."
        )
    if getattr(current, _PATCHED, False):
        return False

    cell = _eager_break_cell(current)
    original = cell.cell_contents if cell is not None else current
    if getattr(original, _PATCHED, False):
        return False
    use_op = _register_qsa_ops()

    def hooked(self, *args, **kwargs):
        # Unconditional op calls. A Python ``if`` here is deleted when the
        # warmup trace has no connector metadata.
        layer_name = self.layer_name
        output = _qsa_output(args, kwargs)
        if use_op and output is not None:
            import torch

            torch.ops.vllm.ucm_qwen4_kv_layer_begin(layer_name, output)
            result = original(self, *args, **kwargs)
            torch.ops.vllm.ucm_qwen4_kv_layer_end(layer_name, output)
            return result
        _qsa_wait(layer_name)
        result = original(self, *args, **kwargs)
        _qsa_save(layer_name)
        return result

    setattr(hooked, _PATCHED, True)
    if cell is not None:
        cell.cell_contents = hooked
        where = "inside eager_break_during_capture"
    else:
        setattr(layer_cls, "_run_qsa", hooked)
        where = "Qwen4ExpQSAAttention._run_qsa"
    setattr(current, _PATCHED, True)
    code = getattr(current, "__code__", None)
    logger.info(
        "UCM Qwen4Exp QSA KV hooks applied on %s (custom_op=%s file=%s qual=%s)",
        where,
        use_op,
        getattr(code, "co_filename", ""),
        getattr(current, "__qualname__", ""),
    )
    return True


def _install_layer_hook(layer_cls, method_name: str, layer_name_of) -> bool:
    original = getattr(layer_cls, method_name, None)
    if not callable(original):
        raise RuntimeError(
            "UCM Qwen4Exp KV hooks require "
            f"{layer_cls.__name__}.{method_name}; "
            "check compatibility with the installed vLLM version."
        )
    if getattr(original, _PATCHED, False):
        return False

    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        layer_name = layer_name_of(self)
        _log_once(
            ("enter", method_name, layer_name),
            "HLA qwen4 hook enter: method=%s layer=%s",
            method_name,
            layer_name,
        )
        if not has_kv_transfer_group():
            _log_once(
                ("skip", method_name, layer_name, "no_kv_transfer_group"),
                "HLA qwen4 hook skip: method=%s layer=%s reason=no_kv_transfer_group",
                method_name,
                layer_name,
            )
            return original(self, *args, **kwargs)
        if not is_v1_kv_transfer_group():
            _log_once(
                ("skip", method_name, layer_name, "not_v1"),
                "HLA qwen4 hook skip: method=%s layer=%s reason=not_v1_kv_transfer_group",
                method_name,
                layer_name,
            )
            return original(self, *args, **kwargs)

        connector = get_kv_transfer_group()
        if not connector.has_connector_metadata():
            _log_once(
                ("skip", method_name, layer_name, "no_connector_metadata"),
                "HLA qwen4 hook skip: method=%s layer=%s reason=no_connector_metadata",
                method_name,
                layer_name,
            )
            return original(self, *args, **kwargs)

        attn_metadata, meta_reason = _attn_metadata(layer_name)
        if attn_metadata is None:
            _log_once(
                ("skip", method_name, layer_name, meta_reason),
                "HLA qwen4 hook skip: method=%s layer=%s reason=%s",
                method_name,
                layer_name,
                meta_reason,
            )
            return original(self, *args, **kwargs)

        row_state = _row_save_state(connector, layer_name)
        connector.wait_for_layer_load(layer_name)
        result = original(self, *args, **kwargs)
        _log_once(
            ("save", method_name, layer_name, row_state),
            "HLA qwen4 hook save_kv_layer: method=%s layer=%s row_save=%s",
            method_name,
            layer_name,
            row_state,
        )
        connector.save_kv_layer(
            layer_name, getattr(self, "kv_cache", None), attn_metadata
        )
        return result

    setattr(wrapped, _PATCHED, True)
    setattr(layer_cls, method_name, wrapped)
    return True


def _patch_qsa_module(mod) -> None:
    logger.info(
        "UCM Qwen4Exp QSA module imported: %s", getattr(mod, "__file__", mod)
    )
    layer_cls = getattr(mod, "Qwen4ExpQSAAttention", None)
    if layer_cls is None:
        raise RuntimeError(
            "UCM Qwen4Exp KV hooks require Qwen4ExpQSAAttention; "
            "check compatibility with the installed vLLM version."
        )
    _install_qsa_run_hook(layer_cls)


@when_imported("vllm.models.qwen4_exp.nvidia.qsa")
def patch_qwen4_exp_qsa_kv_hooks(mod):
    _patch_qsa_module(mod)


logger.info("UCM Qwen4Exp KV patch loaded")


@when_imported("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
def patch_qwen_gdn_kv_hooks(mod):
    layer_cls = getattr(mod, "QwenGatedDeltaNetAttention", None)
    if layer_cls is None:
        raise RuntimeError(
            "UCM Qwen4Exp KV hooks require QwenGatedDeltaNetAttention; "
            "check compatibility with the installed vLLM version."
        )
    hooked = [
        name
        for name in ("_forward_core", "_forward_core_fused_norm_packed")
        if _install_layer_hook(layer_cls, name, lambda self: self.prefix)
    ]
    if hooked:
        logger.info("UCM Qwen GDN KV hooks applied on %s", ", ".join(hooked))
