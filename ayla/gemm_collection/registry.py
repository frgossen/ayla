"""Register the collection's kernel modules as Torch ops with JIT dispatch.

Every kernel here is built for one specific shape and dtype -- the shape is a
literal in its module source, so ``auto_matmul_4096`` multiplies 4096x4096 and
nothing else. Adding one means a module here with a decorated builder, plus
that module in ``__init__``; decorating defines its op at import time, and the
kernel is built on the op's first use::

    @register_gemm
    def auto_matmul_4096_module():
        ...

The name stays bound to the builder, so registrations stack; the op itself is
reached the ordinary way, once anything has imported the package::

    import ayla.gemm_collection  # noqa: F401 -- defines every op

    c = torch.ops.ayla.auto_matmul_4096(a, bt)

Mind the operand layout: these kernels compute ``a @ bt.T``, not ``a @ b``. The
dialect's matmul takes its right operand pre-transposed as ``[N, K]`` (hence the
``%bt`` in the hand-written kernels), and nothing validates the shapes you pass,
so handing one a row-major ``(k, n)`` matrix reads the wrong elements and
silently returns wrong numbers rather than raising.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable

from ..gpu_dialect import ModuleOp
from ..jit_compiler import JITCompiler
from ..torch_custom_call import OperandKind, make_torch_custom_call

Builder = Callable[[], ModuleOp]


@functools.cache
def _launcher(builder: Builder):
    return JITCompiler(builder())


class _LazyLauncher:
    """Builds the kernel on first use, so importing the collection is cheap."""

    def __init__(self, builder: Builder) -> None:
        self.name = builder.__name__.removesuffix("_module")
        self._builder = builder

    @property
    def signature(self):
        return _launcher(self._builder).signature

    def __call__(self, *tensors) -> None:
        _launcher(self._builder)(*tensors)


def register(
    *torch_signature: OperandKind, op_name: str | None = None
) -> Callable[[Builder], Builder]:
    """Define the op for the kernel a builder produces, at import time.

    The arguments say what the kernel does with each of its own -- read,
    read-written, or produced -- which is what `PtxKernel.signature` does not
    carry::

        @register(OperandKind.R, OperandKind.RESULT)
        def cwise_sigmoid_4096_module():
            ...

    The op is ``torch.ops.ayla.<kernel name>``. Registering a kernel twice
    means stacking the decorators and naming all but one of them, since the
    namespace holds one op per name::

        @register(
            OperandKind.R, OperandKind.R, OperandKind.RW,
            op_name="auto_matmul_4096_out",
        )
        @register_gemm
        def auto_matmul_4096_module():
            ...

    The builder is handed back so they stack, and takes no arguments: one
    builder is one kernel. A module that can be built several ways spells each
    one out as its own builder over a shared private helper, so every
    registered kernel has a name you can grep for -- and a builder that still
    takes arguments is rejected rather than quietly registering only its
    defaults.
    """

    def decorate(builder: Builder) -> Builder:
        if inspect.signature(builder).parameters:
            raise TypeError(
                f"{builder.__name__}: builders take no arguments; give each "
                f"variant its own builder over a shared private helper"
            )
        make_torch_custom_call(
            _LazyLauncher(builder), list(torch_signature), name=op_name
        )
        return builder

    return decorate


def register_gemm(builder: Builder) -> Builder:
    """Register a GEMM: ``a`` and ``bt`` read, ``c`` produced."""
    return register(OperandKind.R, OperandKind.R, OperandKind.RESULT)(builder)
