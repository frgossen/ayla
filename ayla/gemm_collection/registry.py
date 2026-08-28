"""Ask the collection which kernels can run a given GEMM.

Every kernel here is built for one specific shape and dtype -- the shape is a
literal in its module source, so ``auto_matmul_4096`` multiplies 4096x4096 and
nothing else. A caller holding a shape therefore cannot just pick an op by
name; it has to ask what fits. This is where that question is answered::

    from ayla import gemm_collection

    for kernel in gemm_collection.query(m=4096, k=4096, n=4096):
        c = kernel.op(a, bt)  # a is (m, k), bt is (n, k), c is (m, n)

Mind the operand layout: these kernels compute ``a @ bt.T``, not ``a @ b``. The
dialect's matmul takes its right operand pre-transposed as ``[N, K]`` (hence the
``%bt`` in the hand-written kernels), and nothing validates the shapes you pass,
so handing one a row-major ``(k, n)`` matrix reads the wrong elements and
silently returns wrong numbers rather than raising.

The metadata is not declared by hand. :func:`register` reads it back off the
built kernel's own signature, so a kernel cannot disagree with the shape it is
registered as, and adding one to the collection means decorating its builder --
nothing here needs updating.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import torch

from ..gpu_dialect import GmemTensorType
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..targets import PtxTarget
from ..torch_custom_call import OperandKind, make_torch_custom_call, to_torch_dtype


@dataclass(frozen=True)
class GemmKernel:
    """A registered kernel and the single GEMM it was compiled for.

    The dims are the mathematical ones -- ``(m, k) @ (k, n) -> (m, n)`` -- read
    off the built kernel rather than declared, so they cannot disagree with what
    the kernel actually does. The *operands* do not have those shapes: ``op``
    takes the right one transposed, as ``(n, k)``. See the module docstring.
    """

    name: str
    op: Callable  # torch.ops.ayla.<name>, called as op(a, bt) -> c
    m: int
    k: int
    n: int
    dtype: torch.dtype
    target: PtxTarget

    @property
    def label(self) -> str:
        return f"ayla.{self.name}"

    def matches(
        self,
        m: int | None = None,
        k: int | None = None,
        n: int | None = None,
        dtype: torch.dtype | None = None,
        compute_capability: tuple[int, int] | None = None,
    ) -> bool:
        """Whether this kernel runs that GEMM. ``None`` matches anything."""
        return (
            (m is None or m == self.m)
            and (k is None or k == self.k)
            and (n is None or n == self.n)
            and (dtype is None or dtype == self.dtype)
            and (
                compute_capability is None
                or self.target.supports_compute_capability(compute_capability)
            )
        )


_KERNELS: list[GemmKernel] = []

_MATMUL_OPERANDS = (OperandKind.R, OperandKind.R, OperandKind.RESULT)


def _gemm_dims(kernel) -> tuple[int, int, int, torch.dtype]:
    """The (m, k, n, dtype) a built kernel implements, from its signature.

    Global-memory tensors are the kernel's own arguments in order, so the three
    of them are lhs, rhs and dst; TMA descriptors are appended after and skipped
    here. The dialect's matmul holds the right operand transposed, so rhs is
    ``[N, K]``, not ``[K, N]``.
    """
    tensors = [typ for typ in kernel.signature if isinstance(typ, GmemTensorType)]
    if len(tensors) != 3:
        raise ValueError(
            f"{kernel.name}: expected a 3-tensor (a, bt, c) GEMM signature, "
            f"got {len(tensors)} global-memory tensors"
        )
    lhs, rhs, out = tensors

    if any(len(t.sizes) != 2 for t in tensors):
        raise ValueError(
            f"{kernel.name}: expected 2-D operands, got "
            f"{[list(t.sizes) for t in tensors]}"
        )
    (m, k), (n, rhs_k) = lhs.sizes, rhs.sizes
    if rhs_k != k or list(out.sizes) != [m, n]:
        raise ValueError(
            f"{kernel.name}: operand shapes {lhs.sizes} @ {rhs.sizes}^T -> "
            f"{out.sizes} are not a consistent GEMM"
        )

    dtypes = {to_torch_dtype(t.dtype) for t in tensors}
    if len(dtypes) != 1:
        raise ValueError(
            f"{kernel.name}: operands disagree on dtype, got {sorted(map(str, dtypes))}"
        )

    return int(m), int(k), int(n), dtypes.pop()


def register(builder: Callable[[], object]) -> Callable[[], object]:
    """Build and register the kernel a builder produces, at import time.

    Decorating runs the builder immediately -- which compiles the kernel, adds
    it to the collection, and defines its ``torch.ops.ayla.*`` op. The builder
    itself is returned unchanged, so it stays callable::

        @register
        def auto_matmul_4096_module():
            ...

    The builder takes no arguments: one builder is one kernel. A module that can
    be built several ways spells each one out as its own builder over a shared
    private helper, so every registered kernel has a name you can grep for --
    and a builder that still takes arguments is rejected rather than quietly
    registering only its defaults.

    Shape and dtype are not declared either -- they are read off the built
    kernel, so there is nothing to keep in sync.
    """
    if inspect.signature(builder).parameters:
        raise TypeError(
            f"{builder.__name__}: @register builders take no arguments; give each "
            f"variant its own builder over a shared private helper"
        )

    kernel = make_ptx_kernel(builder())
    # Derive before defining the op: a bad signature should not leave a
    # torch.ops.ayla.* entry behind that makes the retry fail with "registered
    # multiple times" instead of the real error.
    m, k, n, dtype = _gemm_dims(kernel)

    op = make_torch_custom_call(make_ptx_launcher(kernel), list(_MATMUL_OPERANDS))
    assert kernel.target is not None
    _KERNELS.append(GemmKernel(
        name=kernel.name, op=op, m=m, k=k, n=n, dtype=dtype,
        target=kernel.target,
    ))
    return builder


def query(
    m: int | None = None,
    k: int | None = None,
    n: int | None = None,
    dtype: torch.dtype | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> list[GemmKernel]:
    """The kernels that can run this GEMM. ``None`` leaves a field unconstrained."""
    return [
        kernel for kernel in _KERNELS
        if kernel.matches(m, k, n, dtype, compute_capability)
    ]
