"""Connect CUDA Qwen3.8-Flash-Next QSA and GDN layers to UCM layerwise KV hooks.

Qwen4Exp full attention (``Qwen4ExpQSAAttention``) and linear attention
(``QwenGatedDeltaNetAttention``) do not go through the standard attention op
that calls ``maybe_save_kv_layer_to_connector``. Without an explicit hook,
``UCMHybridLinearAttentionLayerWiseConnector.save_kv_layer`` never runs, so
the scheduler's dump hash is never written and the next lookup is a prefix miss.

A hybrid row is dumped when its last registered layer finishes, after the
QSA pages, GDN state, compressed history, and PLE short-conv in that row have
already been written. These wrappers call ``wait_for_layer_load`` and
``save_kv_layer`` only for that layer name. Other layers in the row, and
models whose row-save layer is still a standard attention module, are left on
the existing path.

QSA full attention is that row-save layer. Its ``forward`` is traced by
``torch.compile``, and a Python ``if`` around the hook is eliminated when the
trace sees no connector metadata. The QSA hook therefore calls a custom op
unconditionally; the op is marked unsafe for CUDA graph capture so piecewise
compilation runs it on every forward, after the QSA KV write.
"""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_HOOK_ATTR = "_ucm_qwen4_kv_hook"
_OPS_READY = False


def _row_save_connector(layer_name: str):
    """Return ``(connector, attn_metadata)`` when this layer closes a hybrid row."""
    if not layer_name:
        return None
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
            is_v1_kv_transfer_group,
        )
        from vllm.forward_context import get_forward_context
    except ImportError:
        return None
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return None

    connector = get_kv_transfer_group()
    if (
        hasattr(connector, "has_connector_metadata")
        and not connector.has_connector_metadata()
    ):
        return None

    inner = getattr(connector, "connector", connector)
    save_layers = getattr(inner, "row_save_layer", None)
    if not isinstance(save_layers, dict) or layer_name not in save_layers.values():
        return None

    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else None
    if not isinstance(metadata, dict) or layer_name not in metadata:
        return None
    return connector, metadata[layer_name]


def _wait_row(layer_name: str) -> None:
    hook = _row_save_connector(layer_name)
    if hook is None:
        return
    connector, _attn_metadata = hook
    connector.wait_for_layer_load(layer_name)


def _save_row(layer_name: str, kv_layer) -> None:
    hook = _row_save_connector(layer_name)
    if hook is None:
        return
    connector, attn_metadata = hook
    connector.save_kv_layer(layer_name, kv_layer, attn_metadata)


def _register_qwen4_kv_ops() -> bool:
    """Register QSA hooks as custom ops so torch.compile cannot drop them."""
    global _OPS_READY
    if _OPS_READY:
        return True
    try:
        import torch
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        logger.warning("Skip Qwen4Exp QSA KV ops: vLLM custom-op helper is missing")
        return False

    def begin(layer_name: str, hidden_states: torch.Tensor) -> None:
        _wait_row(layer_name)

    def end(layer_name: str, output: torch.Tensor) -> None:
        _save_row(layer_name, output)

    def fake_begin(layer_name: str, hidden_states: torch.Tensor) -> None:
        return None

    def fake_end(layer_name: str, output: torch.Tensor) -> None:
        return None

    tags = ()
    if hasattr(torch, "Tag") and hasattr(torch.Tag, "cudagraph_unsafe"):
        tags = (torch.Tag.cudagraph_unsafe,)

    def _define(op_name: str, op_func, fake_impl, mutated: str) -> None:
        kwargs = {
            "op_name": op_name,
            "op_func": op_func,
            "mutates_args": [mutated],
            "fake_impl": fake_impl,
        }
        try:
            direct_register_custom_op(**kwargs, tags=tags)
        except TypeError:
            direct_register_custom_op(**kwargs)

    try:
        _define("ucm_qwen4_kv_layer_begin", begin, fake_begin, "hidden_states")
        _define("ucm_qwen4_kv_layer_end", end, fake_end, "output")
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "duplicate" in message or "exists" in message:
            _OPS_READY = True
            return True
        logger.warning(
            "UCM Qwen4Exp QSA KV op registration failed: %s. "
            "QSA forward will stay eager so the hook still runs.",
            exc,
        )
        return False
    _OPS_READY = True
    return True


def _qsa_hidden_states(args, kwargs):
    import torch

    hidden = kwargs.get("hidden_states")
    if isinstance(hidden, torch.Tensor):
        return hidden
    for arg in reversed(args):
        if isinstance(arg, torch.Tensor) and arg.ndim >= 2:
            return arg
    return None


def _install_qsa_forward_hook(layer_cls) -> bool:
    original = getattr(layer_cls, "forward", None)
    if not callable(original):
        logger.warning(
            "Skip Qwen4Exp QSA KV hook: Qwen4ExpQSAAttention.forward is missing"
        )
        return False
    if getattr(original, _HOOK_ATTR, False):
        return False

    use_op = _register_qwen4_kv_ops()

    if use_op:

        @wraps(original)
        def wrapped(self, *args, **kwargs):
            # Both ops are unconditional so Dynamo cannot drop the hook when
            # the trace sees no connector metadata.
            import torch

            layer_name = self.layer_name
            hidden_states = _qsa_hidden_states(args, kwargs)
            torch.ops.vllm.ucm_qwen4_kv_layer_begin(layer_name, hidden_states)
            result = original(self, *args, **kwargs)
            torch.ops.vllm.ucm_qwen4_kv_layer_end(layer_name, result)
            return result

    else:

        @wraps(original)
        def wrapped(self, *args, **kwargs):
            layer_name = self.layer_name
            _wait_row(layer_name)
            result = original(self, *args, **kwargs)
            _save_row(layer_name, getattr(self, "kv_cache", None))
            return result

        import torch

        wrapped = torch.compiler.disable(wrapped)

    setattr(wrapped, _HOOK_ATTR, True)
    layer_cls.forward = wrapped
    return True


def _install_row_save_hook(layer_cls, method_name: str, layer_name_of) -> bool:
    original = getattr(layer_cls, method_name, None)
    if not callable(original):
        logger.warning(
            "Skip Qwen4Exp KV hook: %s.%s is missing",
            getattr(layer_cls, "__name__", layer_cls),
            method_name,
        )
        return False
    if getattr(original, _HOOK_ATTR, False):
        return False

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        layer_name = layer_name_of(self)
        hook = _row_save_connector(layer_name)
        if hook is not None:
            connector, attn_metadata = hook
            connector.wait_for_layer_load(layer_name)
        result = original(self, *args, **kwargs)
        if hook is not None:
            connector.save_kv_layer(
                layer_name, getattr(self, "kv_cache", None), attn_metadata
            )
        return result

    setattr(wrapped, _HOOK_ATTR, True)
    setattr(layer_cls, method_name, wrapped)
    return True


@when_imported("vllm.models.qwen4_exp.nvidia.qsa")
def patch_qwen4_exp_qsa_kv_hooks(mod):
    layer_cls = getattr(mod, "Qwen4ExpQSAAttention", None)
    if layer_cls is None:
        logger.warning("Skip Qwen4Exp QSA KV hook: Qwen4ExpQSAAttention not found")
        return
    if _install_qsa_forward_hook(layer_cls):
        logger.info(
            "UCM Qwen4Exp QSA layerwise KV hook applied: "
            "custom ops ucm_qwen4_kv_layer_begin/end on "
            "Qwen4ExpQSAAttention.forward"
        )


@when_imported("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
def patch_qwen_gdn_kv_hooks(mod):
    layer_cls = getattr(mod, "QwenGatedDeltaNetAttention", None)
    if layer_cls is None:
        logger.warning(
            "Skip Qwen GDN KV hook: QwenGatedDeltaNetAttention not found"
        )
        return
    hooked = [
        name
        for name in ("_forward_core", "_forward_core_fused_norm_packed")
        if _install_row_save_hook(layer_cls, name, lambda self: self.prefix)
    ]
    if hooked:
        logger.info(
            "UCM Qwen GDN layerwise KV hook applied on %s",
            ", ".join(hooked),
        )
