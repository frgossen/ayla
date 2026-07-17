"""Wrap a :class:`PtxLauncher` as a torch custom op so it's callable from
eager torch and visible to ``torch.compile`` / ``torch.export``.
"""
from __future__ import annotations

from enum import Enum

import torch

from .gpu_dialect import BF16Type, F16Type, F32Type, F64Type
from .ptx_runtime import PtxLauncher

_TORCH_DTYPES = [
    (F16Type(), torch.float16),
    (BF16Type(), torch.bfloat16),
    (F32Type(), torch.float32),
    (F64Type(), torch.float64),
]


def _to_torch_dtype(dtype) -> torch.dtype:
    for ayla_dtype, torch_dtype in _TORCH_DTYPES:
        if dtype == ayla_dtype:
            return torch_dtype
    raise ValueError(f"unsupported dtype: {dtype}")

DEFAULT_NAMESPACE = "ayla"

_LIBRARIES: dict[str, torch.library.Library] = {}


class OperandKind(Enum):
    R = "r"
    RW = "rw"
    RESULT = "result"


def _get_library(namespace: str) -> torch.library.Library:
    lib = _LIBRARIES.get(namespace)
    if lib is None:
        lib = torch.library.Library(namespace, "FRAGMENT")
        _LIBRARIES[namespace] = lib
    return lib


def _make_schema(op_name: str, signature_mapping: list[OperandKind]) -> str:
    params = []
    input_idx = 0
    for i, kind in enumerate(signature_mapping):
        if kind == OperandKind.R:
            params.append(f"Tensor arg{input_idx}")
            input_idx += 1
        elif kind == OperandKind.RW:
            params.append(f"Tensor(a{i}!) arg{input_idx}")
            input_idx += 1

    num_outputs = signature_mapping.count(OperandKind.RESULT)
    ret = ', '.join(['Tensor'] * num_outputs)
    if num_outputs != 1:
        ret = f"({ret})"
    params_str = ', '.join(params)
    return f"{op_name}({params_str}) -> {ret}"


def make_torch_custom_call(
    launcher: PtxLauncher,
    signature_mapping: list[OperandKind],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> torch._ops.OpOverload:
    op_name = launcher.name
    schema = _make_schema(op_name, signature_mapping)

    lib = _get_library(namespace)
    lib.define(schema)

    def cuda_impl(*input_tensors: torch.Tensor):
        kernel_args = []
        output_tensors = []
        in_idx = 0
        device = input_tensors[0].device if input_tensors else torch.device("cuda")
        for kind, arg_type in zip(signature_mapping, launcher.signature):
            if kind == OperandKind.R or kind == OperandKind.RW:
                kernel_args.append(input_tensors[in_idx])
                in_idx += 1
            elif kind == OperandKind.RESULT:
                t = torch.empty(arg_type.sizes, dtype=_to_torch_dtype(arg_type.dtype), device=device)
                output_tensors.append(t)
                kernel_args.append(t)

        launcher(*kernel_args)

        if len(output_tensors) == 1:
            return output_tensors[0]
        if output_tensors:
            return tuple(output_tensors)

    lib.impl(op_name, cuda_impl, "CUDA")

    return getattr(getattr(torch.ops, namespace), op_name).default
