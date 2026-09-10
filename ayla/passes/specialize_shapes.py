"""Specialize dynamic kernel arguments to concrete tensor shapes."""
from __future__ import annotations

from ..base.ir import Type, clone_type, iter_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (
    GmemTensorType,
    KernelOp,
    MemTensorType,
    ModuleOp,
    TmaDescType,
)


def _specialize_argument(
    typ: GmemTensorType, sizes: tuple[int, ...], arg_idx: int,
) -> GmemTensorType:
    if len(sizes) != typ.rank:
        raise ValueError(
            f"kernel argument {arg_idx}: expected rank {typ.rank}, got {sizes}"
        )
    if any(size <= 0 for size in sizes):
        raise ValueError(
            f"kernel argument {arg_idx}: dimensions must be positive: {sizes}"
        )
    for dim, (declared, concrete) in enumerate(zip(typ.sizes, sizes)):
        if declared is not None and declared != concrete:
            raise ValueError(
                f"kernel argument {arg_idx} dimension {dim}: "
                f"expected {declared}, got {concrete}"
            )

    specialized = clone_type(typ, {
        "sizes": list(sizes),
        "strides": GmemTensorType.canonical_strides(list(sizes)),
    })
    assert isinstance(specialized, GmemTensorType)
    return specialized


def _dynamic_type(typ: Type) -> bool:
    if isinstance(typ, MemTensorType):
        return typ.is_dynamic
    if isinstance(typ, TmaDescType):
        return _dynamic_type(typ.gmem_type)
    return False


class SpecializeShapes(Pass):
    """Bind concrete shapes to a high-level kernel's gmem arguments.

    This intentionally specializes only entry-block arguments. Dynamic shapes
    introduced elsewhere are rejected instead of silently passing partially
    specialized IR onward.

    TODO: Propagate dynamic shapes introduced elsewhere through the IR.
    """

    def __init__(self, argument_shapes: list[tuple[int, ...]]) -> None:
        super().__init__(argument_shapes=argument_shapes)

    def run(self, module: ModuleOp) -> ModuleOp:
        top_ops = module.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError("SpecializeShapes requires exactly one kernel")
        kernel = top_ops[0]
        arguments = kernel.body.arguments
        if len(arguments) != len(self.argument_shapes):
            raise ValueError(
                f"SpecializeShapes expected {len(arguments)} argument shapes, "
                f"got {len(self.argument_shapes)}"
            )

        for arg_idx, (argument, sizes) in enumerate(
            zip(arguments, self.argument_shapes, strict=True)
        ):
            if not isinstance(argument.type, GmemTensorType):
                raise ValueError(
                    f"kernel argument {arg_idx}: expected gmem type, "
                    f"got {argument.type}"
                )
            argument.type = _specialize_argument(argument.type, sizes, arg_idx)

        for op in iter_ops(module):
            for block in op.blocks:
                for argument in block.arguments:
                    if _dynamic_type(argument.type):
                        raise NotImplementedError(
                            "SpecializeShapes does not yet propagate dynamic "
                            f"shape {argument.type} through the IR"
                        )
            for value in (*op.operands, *op.results):
                if _dynamic_type(value.type):
                    raise NotImplementedError(
                        "SpecializeShapes does not yet propagate dynamic "
                        f"shape {value.type} through the IR"
                    )

        return module


def specialize_shapes(
    module: ModuleOp,
    argument_shapes: list[tuple[int, ...]],
) -> ModuleOp:
    """Run :class:`SpecializeShapes` on ``module``."""
    return SpecializeShapes(argument_shapes).run(module)
