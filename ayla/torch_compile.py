"""Route matmuls that ayla has a kernel for through ayla, under torch.compile.

Installs lowerings for ``aten.mm`` and ``aten.addmm`` that claim a matmul only
when the collection actually holds a kernel for that exact static shape and
dtype. Everything else returns ``None``, which falls through to Inductor's own
lowering -- so coverage is declared by what is in the registry, not by this
file.

``addmm`` earns its place: a ``Linear`` with a bias reaches Inductor as
``addmm``, not ``mm``, so claiming only ``mm`` leaves the forward pass of a
typical model untouched however well the shape is covered.

    from ayla import torch_compile
    torch_compile.install()

Every kernel matching the shape becomes an autotuning candidate, alongside
Inductor's own choices, so the fastest one wins rather than the first one
found.

**Claiming the ops.** ``user_lowerings`` is consulted before the built-in table
but holds one lowering per op, so ``install`` displaces anyone else registered
for these. ``uninstall`` puts them back, and stands down on any op someone has
claimed since.

**Operand layout.** ``aten.mm(a, b)`` takes ``b`` as ``[K, N]``; ayla's ops
take it as ``[N, K]``, the layout the weight is usually stored in. A model
written as ``x @ w.t()`` reaches us as ``mm(x, permute(w))``, so ``b.t()`` is
the original contiguous weight and the permute costs nothing. When ``b`` is
genuinely ``[K, N]`` in memory that identity does not hold, and the matmul is
left alone rather than paying for a real transpose out of sight.

**No epilogue fusion.** ``ExternKernelChoice`` produces an ``ExternKernel``,
which Inductor never fuses a consumer into. That is the whole reason it is
only a few lines. Fusing an epilogue means producing a ``TemplateBuffer``
instead and implementing two more seams -- ``can_fuse_vertical`` to answer what
ayla's epilogues can absorb, and ``codegen_template`` to lower Inductor's
epilogue IR into ayla's. The shape-coverage boundary below carries over
unchanged when that happens.
"""
from __future__ import annotations

import torch
from torch._inductor.ir import is_contiguous_strides_for_shape
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import lowerings, user_lowerings
from torch._inductor.select_algorithm import (
    ExternKernelChoice,
    autotune_select_algorithm,
)
from torch._inductor.utils import has_free_symbols

from .gemm_collection import query


def _choice_for(kernel):
    """An autotuning candidate that calls ``kernel``, cached by op name."""
    name = f"ayla_{kernel.name}"
    existing = ExternKernelChoice._registry.get(name)
    if existing is not None:
        return existing

    def call(a, b):
        # `b` is the [K, N] view; `b.t()` hands back the [N, K] weight.
        return kernel.op(a, b.t())

    return ExternKernelChoice(call, name=name, has_out_variant=False)


def _lower_mm(mat1, mat2, *, layout=None):
    """Claim this matmul if the collection covers it, else fall through."""
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    # ayla's ops want the weight as [N, K], so `mat2.t()` has to be that tensor.
    if not is_contiguous_strides_for_shape(
        tuple(reversed(mat2.get_stride())), tuple(reversed(mat2.get_size()))
    ):
        return None
    # No kernel is built for a shape that is not known yet.
    if has_free_symbols((m, k, n)):
        return None

    kernels = query(m=int(m), k=int(k), n=int(n), dtype=mat1.get_dtype())
    if not kernels:
        return None

    choices = [
        _choice_for(kernel).bind(input_nodes=(mat1, mat2), layout=layout)
        for kernel in kernels
    ]
    # Returns (node, choice); with one candidate it short-circuits and never
    # benchmarks.
    node, _ = autotune_select_algorithm("ayla_mm", choices, [mat1, mat2], layout)
    return node


def _lower_addmm(inp, mat1, mat2, *, beta=1, alpha=1, layout=None):
    """Claim ``inp + mat1 @ mat2`` if the collection covers the matmul.

    A scaled variant falls through: the collection has no kernel that applies
    ``alpha`` or ``beta``, and folding them into the epilogue below would cost
    a multiply over the whole output to save nothing.
    """
    if beta != 1 or alpha != 1:
        return None

    node = _lower_mm(mat1, mat2, layout=layout)
    if node is None:
        return None

    # The bias lands as a separate pointwise, which Inductor is then free to
    # fuse into whatever consumes the matmul -- an epilogue it already has to
    # emit for the surrounding graph. It cannot fuse into ayla's kernel, which
    # is an ExternKernel, so leaving it to the scheduler is the cheaper half.
    return lowerings[torch.ops.aten.add](inp, node)


_LOWERINGS = {
    torch.ops.aten.mm.default: _lower_mm,
    torch.ops.aten.addmm.default: _lower_addmm,
}
_DISPLACED_LOWERINGS: dict = {}


def install() -> None:
    """Claim ``aten.mm`` and ``aten.addmm`` for the shapes ayla covers."""
    for target, lowering in _LOWERINGS.items():
        if user_lowerings.get(target) is lowering:
            continue
        _DISPLACED_LOWERINGS[target] = user_lowerings.get(target)
        user_lowerings[target] = lowering


def uninstall() -> None:
    """Hand the ops back, unless someone else has claimed them since."""
    for target, lowering in _LOWERINGS.items():
        if user_lowerings.get(target) is not lowering:
            continue
        del user_lowerings[target]
        displaced = _DISPLACED_LOWERINGS.pop(target, None)
        if displaced is not None:
            user_lowerings[target] = displaced
