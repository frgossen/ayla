"""Ask the collection which kernels can run a given GEMM.

Every kernel here is built for one specific shape and dtype -- the shape is a
literal in its module source, so ``auto_matmul_4096`` multiplies 4096x4096 and
nothing else. A caller holding a shape therefore cannot just pick an op by
name; it has to ask what fits. This is where that question is answered::

    from ayla import gemm_collection

    for kernel in gemm_collection.query(m=4096, k=4096, n=4096):
        c = kernel.op(a, b)

The metadata is not declared by hand. :func:`register` reads it back off
the built kernel's own signature, so a kernel cannot disagree with what it is
registered as, and adding one to the collection means decorating its builder --
nothing here needs updating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..gpu_dialect import GmemTensorType
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, _to_torch_dtype, make_torch_custom_call


@dataclass(frozen=True)
class GemmKernel:
    """A registered kernel and the single GEMM it was compiled for.

    ``m``/``k``/``n``/``dtype`` are read off the built kernel rather than
    declared, so they cannot disagree with what the kernel actually does.
    """

    name: str
    op: Callable  # torch.ops.ayla.<name>, called as op(a, b) -> c
    m: int
    k: int
    n: int
    dtype: torch.dtype

    @property
    def label(self) -> str:
        return f"ayla.{self.name}"

    def matches(
        self,
        m: int | None = None,
        k: int | None = None,
        n: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> bool:
        """Whether this kernel runs that GEMM. ``None`` matches anything."""
        return (
            (m is None or m == self.m)
            and (k is None or k == self.k)
            and (n is None or n == self.n)
            and (dtype is None or dtype == self.dtype)
        )


_KERNELS: list[GemmKernel] = []

_MATMUL_OPERANDS = [OperandKind.R, OperandKind.R, OperandKind.RESULT]


def register(builder):
    """Build and register the kernel a builder produces, at import time.

    Decorating runs the builder immediately -- which compiles the kernel, adds
    it to the collection, and defines its ``torch.ops.ayla.*`` op. The builder
    itself is returned unchanged, so it stays callable::

        @register
        def auto_matmul_4096_module():
            ...

    The builder takes no arguments: one builder is one kernel. A module that can
    be built several ways spells each one out as its own builder over a shared
    private helper, so every registered kernel has a name you can grep for.

    Shape and dtype are not declared either -- they are read off the built
    kernel below, so there is nothing to keep in sync.
    """
    kernel = make_ptx_kernel(builder())
    launcher = make_ptx_launcher(kernel)
    op = make_torch_custom_call(launcher, _MATMUL_OPERANDS)

    tensors = [typ for typ in kernel.signature if isinstance(typ, GmemTensorType)]
    if len(tensors) != 3:
        raise ValueError(
            f"{kernel.name}: expected a 3-tensor (a, b, c) GEMM signature, "
            f"got {len(tensors)} global-memory tensors"
        )
    lhs, rhs, out = tensors
    (m, k), (rhs_k, n) = lhs.sizes, rhs.sizes
    if rhs_k != k or list(out.sizes) != [m, n]:
        raise ValueError(
            f"{kernel.name}: operand shapes {lhs.sizes} @ {rhs.sizes} -> "
            f"{out.sizes} are not a consistent GEMM"
        )

    _KERNELS.append(
        GemmKernel(
            name=kernel.name,
            op=op,
            m=int(m),
            k=int(k),
            n=int(n),
            dtype=_to_torch_dtype(lhs.dtype),
        )
    )
    return builder


def query(
    m: int | None = None,
    k: int | None = None,
    n: int | None = None,
    dtype: torch.dtype | None = None,
) -> list[GemmKernel]:
    """The kernels that can run this GEMM. ``None`` leaves a field unconstrained."""
    return [kernel for kernel in _KERNELS if kernel.matches(m, k, n, dtype)]
