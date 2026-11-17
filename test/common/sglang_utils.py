"""Shared source loading and fake modules for SGLang tests."""

import __future__

import ast
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import types
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List
from uuid import uuid4

import numpy as np
from common.path_utils import get_test_root


def _load_source(relative_path, definitions=None, **dependencies):
    """Load selected Python definitions with caller-provided dependencies."""
    path = get_test_root().parent / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if definitions is not None:
        tree.body = [
            node for node in tree.body if getattr(node, "name", None) in definitions
        ]
    module = types.ModuleType("_test_" + path.stem)
    module.__dict__.update(dependencies)
    sys.modules[module.__name__] = module
    exec(
        compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
        module.__dict__,
    )
    return module


def load_source(relative_path, definitions=None, **dependencies):
    defaults = dict(
        dataclass=dataclass,
        replace=replace,
        np=np,
        copy=copy,
        hashlib=hashlib,
        json=json,
        logging=logging,
        init_logger=logging.getLogger,
        time=time,
        re=re,
        threading=threading,
        Path=Path,
        UnifiedCacheLinker=object,
        get_device_module=lambda: None,
        Any=Any,
        List=List,
        Task=object,
        UcmKVStoreBaseV1=object,
        torch=types.SimpleNamespace(Tensor=object),
        uuid4=uuid4,
    )
    defaults.update(dependencies)
    return _load_source(relative_path, definitions=definitions, **defaults)


trace_module = load_source(
    "ucm/integration/sglang/load_trace.py", os=os, ContextVar=ContextVar
)
adapter_module = load_source(
    "ucm/integration/sglang/ucm_direct_store_adapter.py", load_trace=trace_module
)
linker_module = load_source(
    "ucm/integration/sglang/ucm_direct_linker.py",
    load_trace=trace_module,
    **{
        name: getattr(adapter_module, name)
        for name in ("UcmDirectStoreAdapter", "UcmLayerTransfer", "UcmPoolMetadata")
    },
)
pipeline_module = load_source(
    "ucm/store/pipeline/connector.py",
    definitions={
        "UcmPipelineStoreTransTask",
        "UcmPipelineStore",
    },
)
