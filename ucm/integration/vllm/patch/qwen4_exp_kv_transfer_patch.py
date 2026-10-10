"""Connect CUDA Qwen3.8-Flash-Next QSA and GDN layers to UCM layerwise KV hooks.

Same contract as the MiniMax M3 CUDA patch: before the layer writes its cache,
``wait_for_layer_load``; after the write, ``save_kv_layer``. QSA full attention
does not go through ``maybe_save_kv_layer_to_connector``. GDN reaches its cache
update from ``_forward_core``, which the standard attention op never calls.

``UCMHybridLinearAttentionLayerWiseConnector.save_kv_layer`` still ignores every
layer except the row's last registered name. QSA ``forward`` is wrapped with
``torch.compiler.disable`` so ``torch.compile`` cannot delete that call when a
warmup trace sees no connector metadata.
"""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = "_ucm_kv_hooks_patched"
_LOGGED = set()


def _log_once(key, message: str, *args) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(message, *args)


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


def _install_layer_hook(layer_cls, method_name: str, layer_name_of, *, eager: bool):
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
            "HLA qwen4 hook enter: method=%s layer=%s eager=%s",
            method_name,
            layer_name,
            eager,
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
        # Load before this layer writes its cache. save_kv_layer itself ignores
        # every name except the hybrid row's last layer.
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

    if eager:
        import torch

        wrapped = torch.compiler.disable(wrapped)
    setattr(wrapped, _PATCHED, True)
    setattr(layer_cls, method_name, wrapped)
    return True


@when_imported("vllm.models.qwen4_exp.nvidia.qsa")
def patch_qwen4_exp_qsa_kv_hooks(mod):
    layer_cls = getattr(mod, "Qwen4ExpQSAAttention", None)
    if layer_cls is None:
        raise RuntimeError(
            "UCM Qwen4Exp KV hooks require Qwen4ExpQSAAttention; "
            "check compatibility with the installed vLLM version."
        )
    if _install_layer_hook(
        layer_cls, "forward", lambda self: self.layer_name, eager=True
    ):
        logger.info(
            "UCM Qwen4Exp QSA KV hooks applied: "
            "wait_for_layer_load / save_kv_layer on Qwen4ExpQSAAttention.forward"
        )


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
        if _install_layer_hook(layer_cls, name, lambda self: self.prefix, eager=False)
    ]
    if hooked:
        logger.info("UCM Qwen GDN KV hooks applied on %s", ", ".join(hooked))
