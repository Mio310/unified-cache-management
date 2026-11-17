"""GSA (Geometry Sparse Attention) runtime patches for vLLM 0.23.0.

These monkey-patches replicate the ``vllm-adapt-1020-c8.patch`` (targeting
``/vllm-workspace/vllm``) and the Python portions of
``vllm-ascend-adapt-1020-c8.patch`` (targeting ``/vllm-workspace/vllm-ascend``)
so that no explicit ``git apply`` is needed for the Python source.

The C++ additions in ``csrc/torch_binding.cpp`` / ``csrc/torch_binding_meta.cpp``
(registering the ``gsa_*`` torch ops) cannot be applied at runtime and are kept
as the sibling build-time patch
``ucm/sandbox/sparse/gsav2/patch/vllm-ascend-csrc-build-1020-c8.patch``;
it must be ``git apply``-ed before ``pip install -v -e .`` so the ops are
compiled in.

This package (``ucm.sandbox.sparse.gsav2.patch``, targeting vLLM /
vLLM-Ascend 0.23.0 — previously ``ucm.integration.vllm.patch.gsa.v0_23_0``
merged with the former ``gsa/v0.23.0`` build-patch directory) holds the runtime
Python monkey-patches, toggled by ``ENABLE_GSA_PATCH=1`` (in addition to
``ENABLE_UCM_PATCH=1``, via ``apply_patch.apply_all_patches``) and no-ops at
runtime unless GSA is actually initialized (``has_ucm_sparse()`` /
``IS_OPEN_GSA``).

Each patch module mirrors the ``sfa_kv_transfer_patch.py`` approach: it defines
the full replacement function (extracted from the patched source tree) and
rebinds it to the target class/method via ``FunctionType(code, vars(mod), ...)``
+ ``patch_or_inject``, so no string manipulation of method source is used.
"""
