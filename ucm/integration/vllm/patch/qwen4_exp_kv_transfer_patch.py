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
"""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_HOOK_ATTR = "_ucm_qwen4_kv_hook"


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
    if _install_row_save_hook(layer_cls, "forward", lambda self: self.layer_name):
        logger.info(
            "UCM Qwen4Exp QSA layerwise KV hook applied: "
            "wait_for_layer_load / save_kv_layer on Qwen4ExpQSAAttention.forward"
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
