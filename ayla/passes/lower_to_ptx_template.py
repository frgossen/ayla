"""Lower high-level GPU-dialect ops to ptx_template ops.

Implements `LowerToPtxTemplate`, a `LoweringPass` subclass that
dispatches to per-op rewriter methods. Today only `tile` is lowered;
other ops (alloc, load, store, matmul) will be added incrementally.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
import itertools
import math
import struct
import textwrap
from typing import cast

from ..base.ir import Op, Type, Value, find_ancestor_of_type, find_ops_of_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BF16Type,
    CollectiveAxis,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    PtrAddOp,
    PtrType,
    PTXTemplateOp,
    ReinterpretOp,
    ReshapeOp,
    RmemTensorType,
    ConstantOp,
    SmemTensorType,
    ThreadIdOp,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    ToPtrOp,
    WgrmemTensorType,
    physical_layout,
    lock_has_tma_multicast,
    root_buffer,
)


def _lower_mul_registers(
    dst_regs: list[str],
    lhs_regs: list[str],
    rhs_regs: list[str],
    dtype: Type,
) -> str:
    if isinstance(dtype, F32Type):
        opcode = "mul.f32"
    elif isinstance(dtype, BF16Type):
        opcode = "mul.rn.bf16x2"
    else:
        raise NotImplementedError(
            f"lower_elementwise: mul does not support {dtype}"
        )
    return "".join(
        f"{opcode} {dst}, {lhs}, {rhs};\n"
        for dst, lhs, rhs in zip(
            dst_regs, lhs_regs, rhs_regs, strict=True,
        )
    )


def _lower_elementwise_sigmoid_ptx(
    dst_regs: list[str], src_regs: list[str],
) -> str:
    """Lower an F32 sigmoid into the provided registers."""
    return "".join(
        f"mul.f32 {dst}, {src}, 0fBFB8AA3B;\n"
        f"ex2.approx.f32 {dst}, {dst};\n"
        f"add.f32 {dst}, {dst}, 0f3F800000;\n"
        f"rcp.approx.f32 {dst}, {dst};\n"
        for dst, src in zip(dst_regs, src_regs, strict=True)
    )


def _lower_elementwise_sigmoid_bf16_ptx(
    dst_regs: list[str], src_regs: list[str],
) -> str:
    """Lower sigmoid with FTZ, which is lossless for a BF16 result."""
    # Group independent registers by instruction to expose SFU latency.
    phases = (
        "mul.f32 {dst}, {src}, 0fBFB8AA3B;",
        "ex2.approx.ftz.f32 {dst}, {dst};",
        "fma.rn.f32 {dst}, {dst}, 0f3E800000, 0f3E800000;",
        "rcp.approx.ftz.f32 {dst}, {dst};",
        "mul.f32 {dst}, {dst}, 0f3E800000;",
    )
    return "".join(
        phase.format(dst=dst, src=src) + "\n"
        for phase in phases
        for dst, src in zip(dst_regs, src_regs, strict=True)
    )


def _elementwise_operand_regs(value: Value, operand_name: str) -> list[str]:
    """Return the PTX-template registers occupied by one WGRMEM operand."""
    dtype = cast(MemTensorType, value.type).dtype
    if isinstance(dtype, F32Type):
        return [
            f"${{f32:operand.{operand_name}.r{reg}}}"
            for reg in _view_f32_register_indices(value)
        ]
    if isinstance(dtype, (BF16Type, F16Type)):
        return [
            f"${{b32:operand.{operand_name}.r{reg}}}"
            for reg in _view_pair_indices(value)
        ]
    raise NotImplementedError(
        f"lower_elementwise: unsupported register dtype {dtype}"
    )


def _lower_f32_to_packed16_ptx(
    dst_regs: list[str],
    src_regs: list[str],
    src_dtype: Type,
    dst_dtype: Type,
) -> str:
    """Lower F32 to packed BF16/F16; both use PTX's x2 form."""
    assert isinstance(src_dtype, F32Type)
    assert isinstance(dst_dtype, (BF16Type, F16Type))
    assert len(src_regs) == 2 * len(dst_regs)
    return "".join(
        f"cvt.rn.{dst_dtype.name}x2.f32 {dst}, {src_hi}, {src_lo};\n"
        for dst, src_lo, src_hi in zip(
            dst_regs, src_regs[::2], src_regs[1::2], strict=True,
        )
    )


def _lower_packed16_to_f32_ptx(
    dst_regs: list[str],
    src_regs: list[str],
    src_dtype: Type,
    dst_dtype: Type,
) -> str:
    """Unpack paired BF16/F16 registers and convert each element to F32."""
    assert isinstance(src_dtype, (BF16Type, F16Type))
    assert isinstance(dst_dtype, F32Type)
    assert len(dst_regs) == 2 * len(src_regs)
    template = ""
    for index, (src, dst_lo, dst_hi) in enumerate(zip(
        src_regs, dst_regs[::2], dst_regs[1::2], strict=True,
    )):
        src_lo = f"${{b16:convert_lo{index}}}"
        src_hi = f"${{b16:convert_hi{index}}}"
        template += (
            f"mov.b32 {{{src_lo}, {src_hi}}}, {src};\n"
            f"cvt.f32.{src_dtype.name} {dst_lo}, {src_lo};\n"
            f"cvt.f32.{src_dtype.name} {dst_hi}, {src_hi};\n"
        )
    return template


def _lower_dtype_conversion_ptx(
    dst_regs: list[str],
    src_regs: list[str],
    src_dtype: Type,
    dst_dtype: Type,
) -> str:
    """Lower a conversion independently of the element-wise function."""
    if isinstance(src_dtype, (BF16Type, F16Type)) and isinstance(dst_dtype, F32Type):
        return _lower_packed16_to_f32_ptx(dst_regs, src_regs, src_dtype, dst_dtype)
    if (
        isinstance(src_dtype, F32Type)
        and isinstance(dst_dtype, (BF16Type, F16Type))
    ):
        return _lower_f32_to_packed16_ptx(
            dst_regs, src_regs, src_dtype, dst_dtype,
        )
    raise NotImplementedError(
        f"lower_elementwise: unsupported conversion "
        f"{src_dtype} -> {dst_dtype}"
    )


def _lower_convert_elementwise_ptx(op: ElementwiseOp) -> str:
    return _lower_dtype_conversion_ptx(
        _elementwise_operand_regs(op.dst, "dst"),
        _elementwise_operand_regs(op.src(0), "src"),
        cast(MemTensorType, op.src(0).type).dtype,
        cast(MemTensorType, op.dst.type).dtype,
    )


def _elementwise_source_names(op: ElementwiseOp) -> list[str]:
    return ["src"] if op.num_srcs == 1 else ["lhs", "rhs"]


def _sigmoid_ptx(dst_dtype: Type) -> Callable[[list[str], list[str]], str]:
    return (
        _lower_elementwise_sigmoid_bf16_ptx
        if isinstance(dst_dtype, BF16Type)
        else _lower_elementwise_sigmoid_ptx
    )


def _lower_elementwise_in_batches(
    op: ElementwiseOp,
    lower_registers: Callable[
        [ElementwiseOp, list[str], list[list[str]]], str
    ],
) -> str:
    """Lower with either bounded scratch registers or one wide ILP batch."""
    src_regs = [
        _elementwise_operand_regs(op.src(i), name)
        for i, name in enumerate(_elementwise_source_names(op))
    ]
    register_count = len(src_regs[0])
    assert all(len(regs) == register_count for regs in src_regs)

    src_dtype = cast(MemTensorType, op.src(0).type).dtype
    dst_dtype = cast(MemTensorType, op.dst.type).dtype
    dst_regs = _elementwise_operand_regs(op.dst, "dst")
    conversion_width = 1
    if src_dtype != dst_dtype:
        assert register_count == 2 * len(dst_regs)
        conversion_width = 2
    batch_size = register_count if op.ilp else conversion_width

    template = ""
    for start in range(0, register_count, batch_size):
        end = start + batch_size
        batch_src_regs = [regs[start:end] for regs in src_regs]
        if src_dtype == dst_dtype:
            result_regs = dst_regs[start:end]
        else:
            result_regs = [
                f"${{f32:{op.fn}_result{i}}}"
                for i in range(batch_size)
            ]
        template += lower_registers(op, result_regs, batch_src_regs)
        if src_dtype != dst_dtype:
            # TODO: Have PromoteBuffers emit an explicit convert after the F32
            # epilogue; only promotion knows the register accumulator exists.
            template += _lower_dtype_conversion_ptx(
                dst_regs[start // conversion_width:end // conversion_width],
                result_regs,
                src_dtype,
                dst_dtype,
            )
    return template


def _lower_sigmoid_registers(
    op: ElementwiseOp,
    dst_regs: list[str],
    src_regs: list[list[str]],
) -> str:
    src_dtype = cast(MemTensorType, op.src(0).type).dtype
    dst_dtype = cast(MemTensorType, op.dst.type).dtype
    if not isinstance(src_dtype, F32Type):
        raise NotImplementedError(
            f"lower_elementwise: sigmoid does not support {src_dtype} input"
        )
    [src] = src_regs
    return _sigmoid_ptx(dst_dtype)(dst_regs, src)


def _lower_mul_register_batches(
    op: ElementwiseOp,
    dst_regs: list[str],
    src_regs: list[list[str]],
) -> str:
    dtype = cast(MemTensorType, op.dst.type).dtype
    if any(
        cast(MemTensorType, value.type).dtype != dtype
        for value in op.operands
    ):
        raise NotImplementedError("lower_elementwise: mul requires matching operand dtypes")
    lhs_regs, rhs_regs = src_regs
    return _lower_mul_registers(dst_regs, lhs_regs, rhs_regs, dtype)


def _lower_silu_registers(
    op: ElementwiseOp,
    dst_regs: list[str],
    src_reg_batches: list[list[str]],
) -> str:
    src = op.src(0)
    src_dtype = cast(MemTensorType, src.type).dtype
    dst_dtype = cast(MemTensorType, op.dst.type).dtype
    if not isinstance(src_dtype, F32Type):
        raise NotImplementedError(
            f"lower_elementwise: silu does not support {src_dtype} input"
        )
    [src_regs] = src_reg_batches
    tmp_regs = (
        [f"${{f32:silu_tmp{i}}}" for i in range(len(src_regs))]
        if op.dst is src else dst_regs
    )
    template = _sigmoid_ptx(dst_dtype)(tmp_regs, src_regs)
    template += _lower_mul_registers(dst_regs, src_regs, tmp_regs, F32Type())
    return template


_ELEMENTWISE_LOWERERS = {
    "convert": _lower_convert_elementwise_ptx,
    "mul": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_mul_register_batches,
    ),
    "sigmoid": partial(
        _lower_elementwise_in_batches, lower_registers=_lower_sigmoid_registers,
    ),
    "silu": partial(
        _lower_elementwise_in_batches, lower_registers=_lower_silu_registers,
    ),
}


class LowerToPtxTemplate(LoweringPass):
    """Pass that rewrites high-level ops to ptx_template ops."""

    def __init__(self) -> None:
        super().__init__()
        self.cache_policy_values: dict[str, Value] = {}

    @lowers(ConstantOp)
    def lower_constant(
        self, op: ConstantOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        if not isinstance(result_type, RmemTensorType):
            raise NotImplementedError(
                f"lower_constant: unsupported result type {result_type}"
            )
        value = op.value
        values = (
            value
            if isinstance(value, list)
            else itertools.repeat(value, cast(int, result_type.num_elements))
        )
        lowered = PTXTemplateOp.build(
            operands=[],
            result_types=[result_type],
            template=_mov_literals_ptx("value", result_type.dtype, values),
            result_names=["value"],
            description=str(op),
        )
        return [lowered], [lowered.result()]

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        # TODO: Materialize cache policies in an earlier pass, potentially
        # alongside buffer promotion, so this pass only renders their PTX.
        cache_hints = sorted({
            hint
            for tma in find_ops_of_type(op, TmaMoveOp)
            if (hint := tma.l2_cache_hint) is not None
        })
        self.cache_policy_values = {}
        if cache_hints:
            names = [f"policy{i}" for i in range(len(cache_hints))]
            policy_op = PTXTemplateOp.build(
                operands=[],
                result_types=[PtrType.build("gmem") for _ in cache_hints],
                template="\n".join(
                    f"createpolicy.fractional.L2::{hint}.b64 "
                    f"${{b64:result.{name}.ptr}}, 1.0;"
                    for hint, name in zip(cache_hints, names)
                ) + "\n",
                result_names=names,
                description="create TMA L2 cache policies",
            )
            op.body.ops.insert(0, policy_op)
            self.cache_policy_values = dict(zip(cache_hints, policy_op.results))
        self.lower_block(op.body)
        return [op], []

    def _tma_instruction(
        self,
        op: TmaMoveOp,
        opcode: str,
        arguments: str,
        operands: list[Value],
        operand_names: list[str],
        policy_name: str,
    ) -> str:
        hint = op.l2_cache_hint
        if hint is None:
            return f"{opcode} {arguments};"
        operands.append(self.cache_policy_values[hint])
        operand_names.append(policy_name)
        policy = f"${{b64:operand.{policy_name}.ptr}}"
        return f"{opcode}.L2::cache_hint {arguments}, {policy};"

    @lowers(TileOp)
    def lower_tile(
        self, op: TileOp,
    ) -> tuple[list[Op], list[object]] | None:
        """Lower an address tile to a pointer, or erase a WGRMEM view.

        A WGRMEM tile is a zero-cost view of its parent's distributed register
        fragment. Consumers inspect the original TileOp chain to select the
        corresponding root registers; mapping the result to the root value
        prevents the view from allocating another register tensor.
        """
        arg = cast(Value, self.value_map[op.arg])
        new_ops: list[Op] = []

        if isinstance(op.arg.type, WgrmemTensorType):
            _wgmma_view_pairs(op.result())  # Validate the static view.
            assert isinstance(arg, Value)
            return [], [arg]

        # First tile: convert tensor to ptr.
        if isinstance(arg, Value) and isinstance(arg.type, AddrMemTensorType):
            to_ptr = ToPtrOp.build(arg)
            new_ops.append(to_ptr)
            ptr = to_ptr.result()
        elif isinstance(arg, Value) and isinstance(arg.type, PtrType):
            ptr = arg
        else:
            raise NotImplementedError(
                f"lower_tile: unexpected arg type {type(arg)}"
            )

        view_root = op.arg
        while isinstance(view_root.definition, TileOp):
            view_root = view_root.definition.arg
        view_root_type = view_root.type
        assert isinstance(view_root_type, AddrMemTensorType)
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type, view_root_type.dtype,
        ).byte_size
        result_type = op.result().type
        assert isinstance(result_type, AddrMemTensorType)
        result_sizes = cast(tuple[int, ...], result_type.sizes)
        layout = (
            physical_layout(view_root_type)
            if isinstance(view_root_type, SmemTensorType) else None
        )

        # Accumulate byte offset for each dimension.
        zero = IndexConstantOp.build(0)
        new_ops.append(zero)
        byte_offset = zero.result()

        for dim, idx in enumerate(op.indices):
            if layout is not None and layout[1] == "block_major" and dim == 1:
                block_width, _ = layout
                if result_sizes[1] % block_width:
                    parent_type = op.arg.type
                    assert isinstance(parent_type, AddrMemTensorType)
                    parent_width = parent_type.sizes[1]
                    assert isinstance(parent_width, int)
                    assert parent_width % block_width == 0
                    assert isinstance(idx, int), (
                        "block-major sub-block tiles require a static index"
                    )
                    block, inner = divmod(
                        idx * result_sizes[1], block_width,
                    )
                    byte_delta = (
                        block
                        * cast(int, view_root_type.sizes[0])
                        * block_width
                        + inner
                    ) * dtype_bytes
                    byte_term = IndexConstantOp.build(byte_delta)
                    new_ops.append(byte_term)
                    add_op = IndexAddOp.build(byte_offset, byte_term.result())
                    new_ops.append(add_op)
                    byte_offset = add_op.result()
                    continue

            if isinstance(idx, int):
                const_op = IndexConstantOp.build(idx)
                new_ops.append(const_op)
                idx = const_op.result()
            else:
                idx = cast(Value, self.value_map[idx])

            tile_size_op = IndexConstantOp.build(result_sizes[dim])
            new_ops.append(tile_size_op)
            elem_off = IndexMulOp.build(idx, tile_size_op.result())
            new_ops.append(elem_off)

            stride = view_root_type.strides[dim]
            if layout is not None and layout[1] == "block_major":
                block_width, _ = layout
                assert view_root_type.rank == 2
                if dim == 0:
                    stride = block_width
                elif dim == 1:
                    stride = view_root_type.sizes[0]

            assert stride is not None
            stride_bytes = stride * dtype_bytes
            stride_op = IndexConstantOp.build(stride_bytes)
            new_ops.append(stride_op)
            byte_term = IndexMulOp.build(elem_off.result(), stride_op.result())
            new_ops.append(byte_term)
            add_op = IndexAddOp.build(byte_offset, byte_term.result())
            new_ops.append(add_op)
            byte_offset = add_op.result()

        ptr_add = PtrAddOp.build(ptr, byte_offset)
        new_ops.append(ptr_add)
        return new_ops, [ptr_add.result()]

    @lowers(AllocOp)
    def lower_alloc(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        if isinstance(result_type, SmemTensorType):
            return self._lower_alloc_smem(op)
        if isinstance(result_type, RmemTensorType):
            return self._lower_alloc_rmem(op)
        if isinstance(result_type, WgrmemTensorType):
            return self._lower_alloc_wgrmem(op)
        if isinstance(result_type, LockType):
            return self._lower_alloc_lock(op)
        raise NotImplementedError(
            f"lower_alloc: unsupported result type {type(result_type).__name__}"
        )

    @lowers(ReinterpretOp)
    def lower_reinterpret(
        self, op: ReinterpretOp,
    ) -> tuple[list[Op], list[object]]:
        arg = cast(Value, self.value_map[op.arg])
        if op.byte_offset == 0:
            return [], [arg]

        new_ops: list[Op] = []
        if isinstance(arg.type, AddrMemTensorType):
            to_ptr = ToPtrOp.build(arg)
            new_ops.append(to_ptr)
            ptr = to_ptr.result()
        else:
            assert isinstance(arg.type, PtrType)
            ptr = arg
        byte_offset = IndexConstantOp.build(op.byte_offset)
        new_ops.append(byte_offset)
        ptr_add = PtrAddOp.build(ptr, byte_offset.result())
        new_ops.append(ptr_add)
        return new_ops, [ptr_add.result()]

    @lowers(ReshapeOp)
    def lower_reshape(
        self, op: ReshapeOp,
    ) -> tuple[list[Op], list[object]]:
        return [], [cast(Value, self.value_map[op.arg])]

    def _lower_alloc_lock(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        assert isinstance(lock_type, LockType)
        if lock_type.kind == "cluster_mbar":
            return self._lower_alloc_cluster_mbar(op)
        if lock_type.kind != "mbar":
            return [], [None]
        # InsertSyncs represents publication explicitly with BarrierOps.
        template = textwrap.dedent("""\
            .shared .align 8 .b64 ${smem:bar}[1];
            mov.u32 ${b32:result.dst.val}, ${smem:bar};
            mov.u32 ${b32:result.dst.parity}, 0;
            mov.u32 ${b32:tid_x}, %tid.x;
            mov.u32 ${b32:tid_y}, %tid.y;
            or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_y};
            mov.u32 ${b32:tid_z}, %tid.z;
            or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_z};
            setp.ne.s32 ${pred:not_t0}, ${b32:tid_x}, 0;
            @${pred:not_t0} bra ${label:after_init};
            mbarrier.init.shared::cta.b64 [${b32:result.dst.val}], 1;
            fence.mbarrier_init.release.cluster;
            ${label:after_init}:
        """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[lock_type], template=template,
            operand_names=[], result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_cluster_mbar(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
        # InsertSyncs represents publication explicitly with BarrierOps.
        template = textwrap.dedent(f"""\
            .shared .align 8 .b64 ${{smem:bar}}[1];
            mov.u32 ${{b32:result.dst.val}}, ${{smem:bar}};
            mov.u32 ${{b32:tid_x}}, %tid.x;
            mov.u32 ${{b32:tid_y}}, %tid.y;
            or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_y}};
            mov.u32 ${{b32:tid_z}}, %tid.z;
            or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_z}};
            setp.ne.s32 ${{pred:not_t0}}, ${{b32:tid_x}}, 0;
            @${{pred:not_t0}} bra ${{label:after_init}};
            mov.u32 ${{b32:cluster_rank}}, %cluster_ctarank;
            setp.ne.u32 ${{pred:not_rank0}}, ${{b32:cluster_rank}}, 0;
            @${{pred:not_rank0}} bra ${{label:after_init}};
            mbarrier.init.shared::cta.b64 [${{b32:result.dst.val}}], {cluster_size};
            fence.mbarrier_init.release.cluster;
            ${{label:after_init}}:
        """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[lock_type], template=template,
            operand_names=[], result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_rmem(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        assert isinstance(result_type, RmemTensorType)
        template = "" if op.init is None else _mov_literals_ptx(
            "dst",
            result_type.dtype,
            itertools.repeat(op.init, cast(int, result_type.num_elements)),
        )

        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[result_type],
            template=template,
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_wgrmem(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        assert isinstance(result_type, WgrmemTensorType)
        n = result_type.num_elements_per_thread
        suffix = result_type.dtype.name
        if cast(
            BF16Type | F16Type | F32Type | F64Type, result_type.dtype,
        ).byte_size == 2:  # bf16/f16 packed in pairs into b32
            n //= 2
            suffix = "b32"

        template = ""
        if op.init is not None:
            literal = _ptx_literal(result_type.dtype, op.init)
            for i in range(n):
                template += f"mov.{suffix} ${{{suffix}:result.dst.r{i}}}, {literal};\n"

        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[result_type],
            template=template,
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_smem(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        smem_type = op.result().type
        assert isinstance(smem_type, SmemTensorType)
        if op.init is not None:
            raise NotImplementedError("lower_alloc: smem init not supported")
        num_elements = smem_type.num_elements
        assert num_elements is not None
        n_bytes = num_elements * cast(
            BF16Type | F16Type | F32Type | F64Type, smem_type.dtype,
        ).byte_size
        # TMA swizzling and the corresponding software address swizzle operate
        # within 1 KiB regions and require the shared-memory base to match.
        alignment = (
            1024
            if smem_type.swizzle is not None
            and physical_layout(smem_type) is not None
            else 128
        )
        alignment = op.attrs.get("allocation_alignment", alignment)
        template = textwrap.dedent(f"""\
            .shared .align {alignment} .b8 ${{smem:sym}}[{n_bytes}];
            mov.u32 ${{b32:result.dst.ptr}}, ${{smem:sym}};
        """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[smem_type], template=template,
            operand_names=[], result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    @lowers(ElementwiseOp)
    def lower_elementwise(
        self, op: ElementwiseOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower an element-wise op on wgrmem tensors, one function per element.

        Source and destination dtypes are independent. Function lowering runs
        in the source dtype and a supported conversion writes the result to the
        destination dtype.
        """
        sources = [op.src(i) for i in range(op.num_srcs)]
        dst_value = cast(Value, self.value_map[op.dst])
        source_values = [cast(Value, self.value_map[src]) for src in sources]
        dst_type = op.dst.type

        if not isinstance(dst_type, WgrmemTensorType):
            raise NotImplementedError(
                f"lower_elementwise: destination must be wgrmem; got {dst_type}"
            )
        for src in sources:
            src_type = src.type
            if not isinstance(src_type, WgrmemTensorType):
                raise NotImplementedError(
                    f"lower_elementwise: source must be wgrmem; got {src_type}"
                )
            assert tuple(src_type.sizes) == tuple(dst_type.sizes)
            assert src_type.warps == dst_type.warps
        lower = _ELEMENTWISE_LOWERERS.get(op.fn)
        if lower is None:
            raise NotImplementedError(
                f"lower_elementwise: unsupported fn {op.fn!r}"
            )
        template = lower(op)

        source_names = _elementwise_source_names(op)
        assert len(source_names) == op.num_srcs
        new_op = PTXTemplateOp.build(
            operands=[dst_value, *source_values],
            result_types=[],
            template=template,
            operand_names=["dst", *source_names],
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    @lowers(MatmulOp)
    def lower_matmul(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]]:
        lhs_type = op.lhs.type
        rhs_type = op.rhs.type
        dst_type = op.dst.type
        if (isinstance(lhs_type, RmemTensorType)
                and isinstance(rhs_type, RmemTensorType)
                and isinstance(dst_type, RmemTensorType)):
            return self._lower_matmul_fma(op)
        if (isinstance(lhs_type, WgrmemTensorType)
                and isinstance(rhs_type, WgrmemTensorType)
                and isinstance(dst_type, WgrmemTensorType)):
            return self._lower_matmul_mma_sync(op)
        raise NotImplementedError(
            f"lower_matmul: unsupported operand types; got "
            f"{lhs_type} @ {rhs_type} -> {dst_type}"
        )

    def _lower_matmul_fma(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]]:
        lhs = cast(Value, self.value_map[op.lhs])
        rhs = cast(Value, self.value_map[op.rhs])
        dst = cast(Value, self.value_map[op.dst])
        lhs_type = lhs.type
        rhs_type = rhs.type
        dst_type = dst.type
        assert isinstance(lhs_type, RmemTensorType)
        assert isinstance(rhs_type, RmemTensorType)
        assert isinstance(dst_type, RmemTensorType)
        if not isinstance(lhs_type.dtype, F32Type):
            raise NotImplementedError(
                f"lower_async_matmul fma: only f32 supported; got {lhs_type.dtype}"
            )
        if not (lhs_type.dtype == rhs_type.dtype == dst_type.dtype):
            raise NotImplementedError(
                f"lower_async_matmul fma: dtype mismatch; got "
                f"{lhs_type.dtype}, {rhs_type.dtype}, {dst_type.dtype}"
            )
        m_dim, k_dim = cast(tuple[int, int], lhs_type.sizes)
        n_dim, k_dim2 = cast(tuple[int, int], rhs_type.sizes)
        if k_dim != k_dim2:
            raise NotImplementedError(
                f"lower_async_matmul fma: k-dim mismatch; got "
                f"{lhs.type} @ {rhs.type}"
            )
        if tuple(dst_type.sizes) != (m_dim, n_dim):
            raise NotImplementedError(
                f"lower_async_matmul fma: dst shape must be {m_dim}x{n_dim}; got {dst.type}"
            )
        template = ""
        for m in range(m_dim):
            for n in range(n_dim):
                for k in range(k_dim):
                    a = f"${{f32:operand.lhs.r{m * k_dim + k}}}"
                    b = f"${{f32:operand.rhs.r{n * k_dim + k}}}"
                    c = f"${{f32:operand.dst.r{m * n_dim + n}}}"
                    template += f"fma.rn.f32 {c}, {a}, {b}, {c};\n"
        new_op = PTXTemplateOp.build(
            operands=[lhs, rhs, dst],
            result_types=[],
            template=template,
            operand_names=["lhs", "rhs", "dst"],
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_matmul_mma_sync(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]]: # TODO: rename "mma" to "mma_layout_0, 1, 2" and impl it the right way
        lhs = cast(Value, self.value_map[op.lhs])
        rhs = cast(Value, self.value_map[op.rhs])
        dst = cast(Value, self.value_map[op.dst])
        lhs_type = lhs.type
        rhs_type = rhs.type
        dst_type = dst.type
        assert isinstance(lhs_type, WgrmemTensorType)
        assert isinstance(rhs_type, WgrmemTensorType)
        assert isinstance(dst_type, WgrmemTensorType)
        assert dst_type.layout == "mma", (
            f"_lower_async_matmul_mma_sync: dst layout must be 'mma'; got {dst_type.layout!r}"
        )
        if not (isinstance(lhs_type.dtype, (BF16Type, F16Type))
                and isinstance(rhs_type.dtype, (BF16Type, F16Type))
                and isinstance(dst_type.dtype, F32Type)):
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: expected bf16/f16 inputs, f32 output; "
                f"got {lhs_type.dtype}, {rhs_type.dtype} -> {dst_type.dtype}"
            )
        m_dim, k_dim = cast(tuple[int, int], lhs_type.sizes)
        n_dim, k_dim2 = cast(tuple[int, int], rhs_type.sizes)
        if k_dim != k_dim2:
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: k-dim mismatch; got "
                f"{lhs.type} @ {rhs.type}"
            )
        if tuple(dst_type.sizes) != (m_dim, n_dim):
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: dst shape must be {m_dim}x{n_dim}; "
                f"got {dst.type}"
            )
        if m_dim != 16 or k_dim % 16 != 0 or n_dim % 8 != 0:
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: need m=16, k%%16=0, n%%8=0; "
                f"got {m_dim}x{k_dim} @ {k_dim}x{n_dim}"
            )
        in_dtype = lhs_type.dtype.name
        template = ""
        for k_tile in range(k_dim // 16):
            a_base = k_tile * 4
            for n_tile in range(n_dim // 8):
                b_base = (k_tile * (n_dim // 8) + n_tile) * 2
                c_base = n_tile * 4
                a_list = ", ".join(
                    f"${{b32:operand.lhs.r{a_base + i}}}" for i in range(4)
                )
                b_list = ", ".join(
                    f"${{b32:operand.rhs.r{b_base + i}}}" for i in range(2)
                )
                d_list = ", ".join(
                    f"${{f32:operand.dst.r{c_base + i}}}" for i in range(4)
                )
                template += (
                    f"mma.sync.aligned.m16n8k16.row.col.f32.{in_dtype}.{in_dtype}.f32\n"
                    f"  {{{d_list}}},\n"
                    f"  {{{a_list}}},\n"
                    f"  {{{b_list}}},\n"
                    f"  {{{d_list}}};\n"
                )
        new_op = PTXTemplateOp.build(
            operands=[lhs, rhs, dst],
            result_types=[],
            template=template,
            operand_names=["lhs", "rhs", "dst"],
            result_names=[],
            description=str(op),
        )
        return [new_op], []



    @lowers(AsyncOp)
    def lower_async(
        self, op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        body_ops = op.body.ops
        tmas = [body_op for body_op in body_ops if isinstance(body_op, TmaMoveOp)]
        if body_ops and len(tmas) == len(body_ops):
            lock_type = op.lock.type
            assert isinstance(lock_type, LockType)
            kind = lock_type.kind
            if kind == "mbar":  # one or more gmem→smem loads sharing the mbar
                lock_value = cast(Value, self.value_map[op.lock])
                return self._lower_async_tma_loads(op, tmas, lock_value)
            if kind == "bulk":  # smem→gmem store (single)
                assert len(body_ops) == 1, (
                    f"lower_async: bulk TMA store group must be a single move; "
                    f"got {len(body_ops)}"
                )
                return self._lower_async_tma_store(tmas[0])
            raise NotImplementedError(
                f"lower_async: tma_move with lock kind {kind!r}"
            )
        mms = [body_op for body_op in body_ops if isinstance(body_op, MatmulOp)]
        if body_ops and len(mms) == len(body_ops):
            return self._lower_async_matmul_wgmma(op, mms)
        raise NotImplementedError(
            f"lower_async: unsupported body ops "
            f"{[type(o).__name__ for o in body_ops]}"
        )

    def _lower_async_matmul_wgmma(
        self, op: AsyncOp, mms: list[MatmulOp],
    ) -> tuple[list[Op], list[object]]:
        """Lower a group of smem × smem → wgrmem matmuls to wgmma.mma_async.

        The whole group is one wgmma commit group: a single ``wgmma.fence``, all
        the ``mma_async`` (accumulating into the shared dst), and a single
        ``wgmma.commit_group``. The hardware serializes same-accumulator mmas, so
        the await's lone ``wgmma.wait_group`` covers them all (this is the batched
        form nvcc emits, cf. ``fastcu_5_1``). The group's await op (lock kind
        ``wgmma``) emits the wait plus a ``bar.sync`` that closes the TMA-reload
        vs warpgroup-read WAR (see ``_lower_await_wgmma``).
        """
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        assert lock_type.kind == "wgmma", (
            f"_lower_async_matmul_wgmma: expected lock kind 'wgmma'; got {lock_type.kind!r}"
        )

        _swizzle_to_bits = {None: 0, "128B": 1, "64B": 2, "32B": 3}

        def _build_static_desc(lbo, sbo, swizzle):
            return (lbo >> 4) << 16 | (sbo >> 4) << 32 | swizzle << 62

        # Descriptors are deliberately rematerialized near each WGMMA group:
        # keeping them live across the loop raises register pressure.
        def _build_desc(name, ptr_reg, desc_const, offset=None):
            t = ""
            if offset is not None:
                t += f"add.u32 ${{b32:{name}_addr}}, ${{b32:{ptr_reg}}}, {offset};\n"
                t += f"shr.b32 ${{b32:{name}_addr}}, ${{b32:{name}_addr}}, 4;\n"
            else:
                t += f"shr.b32 ${{b32:{name}_addr}}, ${{b32:{ptr_reg}}}, 4;\n"
            t += f"cvt.u64.u32 ${{b64:{name}}}, ${{b32:{name}_addr}};\n"
            t += f"or.b64 ${{b64:{name}}}, ${{b64:{name}}}, {desc_const};\n"
            return t

        def _tile_delta(value: Value, base: Value, group: int) -> int | None:
            if group == 0 or not (
                isinstance(value.definition, TileOp)
                and isinstance(base.definition, TileOp)
            ):
                return None
            tile = value.definition
            base_tile = base.definition
            # Descriptor reuse is limited to siblings in one WGMMA commit
            # group that share the same direct parent/base.
            if (
                tile.arg is not base_tile.arg
                or tile.result().type != base_tile.result().type
            ):
                return None

            typ = tile.result().type
            assert isinstance(typ, AddrMemTensorType)
            assert len(tile.indices) == len(base_tile.indices) == typ.rank, (
                "_tile_delta: TileOp indices must match the result rank; "
                f"got {len(tile.indices)}, {len(base_tile.indices)}, and {typ.rank}"
            )

            def constant(index: Value | int) -> int | None:
                if isinstance(index, int):
                    return index
                if isinstance(index.definition, IndexConstantOp):
                    return index.definition.value
                return None

            indices = [constant(index) for index in tile.indices]
            base_indices = [constant(index) for index in base_tile.indices]
            if any(index is None for index in indices + base_indices):
                return None
            # A physical layout is not affine in the logical coordinates, so
            # rematerialize its descriptors from the already-lowered pointers.
            if typ.attrs.get("physical_layout") is not None:
                return None
            delta = 0
            for index, base_index, size, stride in zip(
                indices, base_indices, typ.sizes, typ.strides,
            ):
                assert index is not None and base_index is not None
                assert size is not None and stride is not None
                delta += (index - base_index) * size * stride
            byte_delta = delta * cast(
                BF16Type | F16Type | F32Type | F64Type, typ.dtype,
            ).byte_size
            return byte_delta if byte_delta >= 0 and byte_delta % 16 == 0 else None

        # Each distinct operand value gets a single unique placeholder name —
        # matmuls accumulating into the same dst share that operand/register set.
        operands: list[Value] = []
        operand_names: list[str] = []
        name_of: dict[Value, str] = {}

        def _operand(value: Value, hint: str) -> str:
            if value in name_of:
                return name_of[value]
            name = f"{hint}{len(operand_names)}"
            name_of[value] = name
            operands.append(value)
            operand_names.append(name)
            return name

        def _desc(name, base, delta, value, hint, const, offset=None):
            if delta is not None:
                return f"add.u64 ${{b64:{name}}}, ${{b64:{base}}}, {delta // 16};\n"
            operand = _operand(cast(Value, self.value_map[value]), hint)
            return _build_desc(
                name, f"operand.{operand}.ptr", const, offset,
            )

        desc_template = ""
        mma_template = ""
        for j, mm in enumerate(mms):
            lhs_type = mm.lhs.type
            rhs_type = mm.rhs.type
            dst_type = mm.dst.type
            assert (isinstance(lhs_type, SmemTensorType)
                    and isinstance(rhs_type, SmemTensorType)
                    and isinstance(dst_type, WgrmemTensorType)
                    and dst_type.warps == 4), (
                f"_lower_async_matmul_wgmma: unsupported matmul operand types; "
                f"got {lhs_type} @ {rhs_type} -> {dst_type}"
            )
            assert dst_type.layout == "wgmma_layout_0", (
                f"_lower_async_matmul_wgmma: dst layout must be 'wgmma'; got {dst_type.layout!r}"
            )
            assert lhs_type.rank == 2 and rhs_type.rank == 2 and dst_type.rank == 2
            assert lhs_type.strides[-1] == 1, (
                f"_lower_async_matmul_wgmma: lhs innermost stride must be 1; "
                f"got strides={list(lhs_type.strides)}"
            )
            assert rhs_type.strides[-1] == 1, (
                f"_lower_async_matmul_wgmma: rhs innermost stride must be 1; "
                f"got strides={list(rhs_type.strides)}"
            )
            m_dim, k_dim = cast(tuple[int, int], lhs_type.sizes)
            n_dim, k_dim2 = cast(tuple[int, int], rhs_type.sizes)
            assert k_dim == k_dim2
            assert k_dim == 16, (
                f"_lower_async_matmul_wgmma: k must be 16; got {k_dim}"
            )
            assert m_dim % 64 == 0, (
                f"_lower_async_matmul_wgmma: m must be multiple of 64; got {m_dim}"
            )
            assert n_dim % 8 == 0 and 8 <= n_dim <= 256, (
                f"_lower_async_matmul_wgmma: n must be a multiple of 8 in [8, 256]; got {n_dim}"
            )

            m_iters = m_dim // 64
            num_regs_per_wgmma = 64 * n_dim // (dst_type.warps * 32)
            lhs_layout = physical_layout(lhs_type)
            rhs_layout = physical_layout(rhs_type)
            lhs_row_stride = (
                lhs_layout[0]
                if lhs_layout is not None and lhs_layout[1] == "block_major"
                else lhs_type.strides[0]
            )
            rhs_row_stride = (
                rhs_layout[0]
                if rhs_layout is not None and rhs_layout[1] == "block_major"
                else rhs_type.strides[0]
            )
            lhs_row_stride_bytes = (
                cast(int, lhs_row_stride)
                * cast(
                    BF16Type | F16Type | F32Type | F64Type, lhs_type.dtype,
                ).byte_size
            )
            rhs_row_stride_bytes = (
                cast(int, rhs_row_stride)
                * cast(
                    BF16Type | F16Type | F32Type | F64Type, rhs_type.dtype,
                ).byte_size
            )

            assert lhs_type.swizzle in _swizzle_to_bits, (
                f"_lower_async_matmul_wgmma: lhs has unknown swizzle {lhs_type.swizzle!r}"
            )
            assert rhs_type.swizzle in _swizzle_to_bits, (
                f"_lower_async_matmul_wgmma: rhs has unknown swizzle {rhs_type.swizzle!r}"
            )
            lead_dim = 16
            desc_const_lhs = _build_static_desc(
                lead_dim, lhs_row_stride_bytes * 8, _swizzle_to_bits[lhs_type.swizzle])
            desc_const_rhs = _build_static_desc(
                lead_dim, rhs_row_stride_bytes * 8, _swizzle_to_bits[rhs_type.swizzle])

            lhs_delta = _tile_delta(mm.lhs, mms[0].lhs, j)
            rhs_delta = _tile_delta(mm.rhs, mms[0].rhs, j)
            dst_name = _operand(cast(Value, self.value_map[mm.dst]), "dst")

            desc_rhs = f"desc_rhs{j}"
            desc_template += _desc(
                desc_rhs, "desc_rhs0", rhs_delta,
                mm.rhs, "rhs", desc_const_rhs,
            )
            for m_it in range(m_iters):
                lhs_offset = m_it * 64 * lhs_row_stride_bytes
                desc_lhs = f"desc_lhs{j}_{m_it}"
                desc_template += _desc(
                    desc_lhs, f"desc_lhs0_{m_it}", lhs_delta,
                    mm.lhs, "lhs", desc_const_lhs,
                    lhs_offset if m_it > 0 else None,
                )

            input_dtype = lhs_type.dtype.name
            if rhs_type.dtype.name != input_dtype:
                raise NotImplementedError(
                    "wgmma inputs must have the same dtype; got "
                    f"{lhs_type.dtype} and {rhs_type.dtype}"
                )
            reg_offset = 0
            for m_it in range(m_iters):
                regs = ", ".join(
                    f"${{f32:operand.{dst_name}.r{reg_offset + i}}}"
                    for i in range(num_regs_per_wgmma)
                )
                mma_template += textwrap.dedent(f"""\
                    wgmma.mma_async.sync.aligned.m64n{n_dim}k16.f32.{input_dtype}.{input_dtype}
                      {{{regs}}},
                      ${{b64:desc_lhs{j}_{m_it}}},
                      ${{b64:{desc_rhs}}},
                      1, 1, 1, 0, 0;
                """)
                reg_offset += num_regs_per_wgmma

        # One commit group for the whole batch: single fence, all mma_async, one
        # commit. Hardware serializes same-accumulator mmas, so the await's lone
        # wait_group covers them all. The await also emits an asymmetric named
        # barrier that closes the cross-warp WAR (thread-0 TMA reload vs
        # warpgroup-wide wgmma reads).
        template = (
            "wgmma.fence.sync.aligned;\n"
            + desc_template
            + mma_template
            + "wgmma.commit_group.sync.aligned;\n"
        )

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    @lowers(MoveOp)
    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower synchronous MoveOp to PTX."""
        dst_type = op.dst.type
        src_type = op.src.type

        if {type(dst_type), type(src_type)} == {RmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_rmem(op)
        if {type(dst_type), type(src_type)} == {RmemTensorType, SmemTensorType}:
            return self._lower_move_smem_rmem(op)
        if {type(dst_type), type(src_type)} == {SmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_smem(op)
        if isinstance(dst_type, WgrmemTensorType) and isinstance(src_type, AddrMemTensorType) and dst_type.layout == "wgmma_layout_0":
            return self._lower_move_wgrmem_wgmma_addrmem(op)
        if isinstance(dst_type, WgrmemTensorType) and isinstance(src_type, SmemTensorType) and dst_type.layout == "mma":
            return self._lower_move_smem_to_wgrmem_mma(op)
        if isinstance(dst_type, SmemTensorType) and isinstance(src_type, WgrmemTensorType) and src_type.layout == "mma":
            return self._lower_move_wgrmem_mma_to_smem(op)
        if (
            isinstance(dst_type, (SmemTensorType, GmemTensorType))
            and isinstance(src_type, WgrmemTensorType)
            and src_type.layout == "wgmma_layout_0"
        ):
            return self._lower_move_wgrmem_wgmma(op)
        raise NotImplementedError(
            f"lower_move: unsupported; got "
            f"{type(dst_type).__name__} <- {type(src_type).__name__}"
        )

    @lowers(AwaitOp)
    def lower_await(
        self, op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        kind = lock_type.kind
        if kind == "mbar":
            lock = cast(Value, self.value_map[op.lock])
            return self._lower_await_mbar(op, lock)
        if kind == "wgmma":
            return self._lower_await_wgmma(op)
        if kind == "bulk":
            return self._lower_await_bulk(op)
        raise NotImplementedError(
            f"lower_await: unsupported lock kind {kind!r}"
        )

    @lowers(BarrierOp)
    def lower_barrier(
        self, op: BarrierOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower a cross-thread ordering fence."""
        scope = op.attrs.get("scope")
        operands: list[Value] = []
        operand_names: list[str] = []
        if scope == "thread":
            return [], []  # per-thread ordering is implicit in program order
        if scope == "block":
            template = "bar.sync 0;\n"
        elif scope == "cluster_to_leader":
            assert op.sync is not None
            sync = self.value_map[op.sync]
            assert isinstance(sync, Value)
            template = textwrap.dedent("""\
                bar.sync 0;
                mov.u32 ${b32:cl_tid}, %tid.x;
                mov.u32 ${b32:cl_tidy}, %tid.y;
                or.b32 ${b32:cl_tid}, ${b32:cl_tid}, ${b32:cl_tidy};
                mov.u32 ${b32:cl_tidz}, %tid.z;
                or.b32 ${b32:cl_tid}, ${b32:cl_tid}, ${b32:cl_tidz};
                setp.ne.u32 ${pred:cl_not_t0}, ${b32:cl_tid}, 0;
                @${pred:cl_not_t0} bra ${label:cl_done};
                mov.u32 ${b32:cluster_rank}, %cluster_ctarank;
                setp.eq.u32 ${pred:rank0}, ${b32:cluster_rank}, 0;
                @${pred:rank0} bra ${label:cl_rank0};
                mov.u32 ${b32:cluster_rank}, 0;
                mapa.shared::cluster.u32 ${b32:remote_empty}, ${b32:operand.sync.val}, ${b32:cluster_rank};
                mbarrier.arrive.shared::cluster.b64 _, [${b32:remote_empty}];
                bra ${label:cl_done};
                ${label:cl_rank0}:
                mbarrier.arrive.shared::cta.b64 ${b64:cluster_state}, [${b32:operand.sync.val}];
                ${label:cl_wait}:
                mbarrier.try_wait.acquire.cluster.shared::cta.b64 ${pred:ready}, [${b32:operand.sync.val}], ${b64:cluster_state};
                @!${pred:ready} bra ${label:cl_wait};
                ${label:cl_done}:
                """)
            operands = [sync]
            operand_names = ["sync"]
        elif scope in ("leader_to_cluster", "cluster"):
            template = textwrap.dedent("""\
                barrier.cluster.arrive.release;
                barrier.cluster.wait.acquire;
                """)
        elif scope == "block_leader":
            # Warp 0 (t0) waits, others arrive, on the named barrier id chosen by
            # insert_syncs (unique among concurrently-live block-leader barriers).
            bar_id = op.attrs.get("id", 1)
            num_threads = op.attrs["num_threads"]
            template = textwrap.dedent(f"""\
                mov.u32 ${{b32:bl_tid}}, %tid.x;
                setp.lt.u32 ${{pred:bl_w0}}, ${{b32:bl_tid}}, 32;
                @${{pred:bl_w0}} bar.sync {bar_id}, {num_threads};
                @!${{pred:bl_w0}} bar.arrive {bar_id}, {num_threads};
                """)
        elif scope == "grid":
            assert op.sync is not None
            ancestor = find_ancestor_of_type(op, KernelOp)
            assert ancestor is not None
            grid_size = math.prod(
                dim if dim is not None else 1 for dim in ancestor.grid_dim
            )
            sync = cast(Value, self.value_map[op.sync])
            template = textwrap.dedent(f"""\
                bar.sync 0;
                mov.u32 ${{b32:grid_tid}}, %tid.x;
                setp.ne.u32 ${{pred:grid_not_leader}}, ${{b32:grid_tid}}, 0;
                @${{pred:grid_not_leader}} bra ${{label:grid_after_leader}};
                ld.acquire.gpu.global.u32 ${{b32:grid_sense}}, [${{b64:operand.sync.ptr}}+4];
                atom.acq_rel.gpu.global.inc.u32 ${{b32:grid_ticket}}, [${{b64:operand.sync.ptr}}], {grid_size - 1};
                setp.eq.u32 ${{pred:grid_last}}, ${{b32:grid_ticket}}, {grid_size - 1};
                @!${{pred:grid_last}} bra ${{label:grid_wait}};
                xor.b32 ${{b32:grid_seen}}, ${{b32:grid_sense}}, 1;
                st.release.gpu.global.u32 [${{b64:operand.sync.ptr}}+4], ${{b32:grid_seen}};
                bra ${{label:grid_done}};
                ${{label:grid_wait}}:
                ld.acquire.gpu.global.u32 ${{b32:grid_seen}}, [${{b64:operand.sync.ptr}}+4];
                setp.eq.u32 ${{pred:grid_spinning}}, ${{b32:grid_seen}}, ${{b32:grid_sense}};
                @${{pred:grid_spinning}} bra ${{label:grid_wait}};
                ${{label:grid_done}}:
                ${{label:grid_after_leader}}:
                bar.sync 0;
                """)
            operands = [sync]
            operand_names = ["sync"]
        else:
            raise NotImplementedError(
                f"lower_barrier: unsupported barrier scope {scope!r}"
            )
        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_await_mbar(
        self, op: AwaitOp, lock: Value,
    ) -> tuple[list[Op], list[object]]:
        # Materializing the optional hint in a virtual register lets ptxas
        # select the compact fused wait; an immediate or test_wait expands
        # into substantially more phase-check, sleep, or yield instructions.
        multicast_async = lock_has_tma_multicast(op.lock)
        acquire_scope = "cluster" if multicast_async else "cta"
        template = textwrap.dedent(fr"""
            mov.u32 ${{b32:ticks}}, 0x989680;
            ${{label:wait}}:
            mbarrier.try_wait.parity.acquire.{acquire_scope}.shared::cta.b64 ${{pred:mbar_done}}, [${{b32:operand.lock.val}}], ${{b32:operand.lock.parity}}, ${{b32:ticks}};
            @!${{pred:mbar_done}} bra ${{label:wait}};
            xor.b32 ${{b32:operand.lock.parity}}, ${{b32:operand.lock.parity}}, 1;
        """)
        new_op = PTXTemplateOp.build(
            operands=[lock],
            result_types=[],
            template=template,
            operand_names=["lock"],
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_await_wgmma(
        self, op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        # wait_group drains the commit group so the warpgroup's wgmma reads of
        # the smem operands have retired. The buffer-reuse WAR (thread-0's TMA
        # reload vs those reads) is handled by insert_syncs, which emits a barrier
        # before the reload — so no barrier is needed here.
        template = textwrap.dedent(r"""
            wgmma.wait_group.sync.aligned 0;
        """)
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[],
            template=template,
            operand_names=[],
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_await_bulk(
        self, op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        template = textwrap.dedent(r"""
            cp.async.bulk.wait_group.read 0;
        """)
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[],
            template=template,
            operand_names=[],
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_gmem_rmem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower per-thread move between gmem and a faster memory (rmem).

        Handles both directions (load and store) via `ld.global` / `st.global`.
        Each thread independently computes its address from tile indices
        and accesses its own elements.
        """
        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])

        is_load = isinstance(op.dst.type, RmemTensorType)
        gmem_ir_type = op.src.type if is_load else op.dst.type
        rmem_type = dst_value.type if is_load else src_value.type
        assert isinstance(gmem_ir_type, GmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        gmem_label = "src" if is_load else "dst"
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type, gmem_ir_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = f"${{b64:operand.{gmem_label}.ptr}}"

        for flat_idx, idx in enumerate(
            itertools.product(
                *[range(s) for s in cast(tuple[int, ...], rmem_type.sizes)]
            )
        ):
            strides = cast(tuple[int, ...], gmem_ir_type.strides)
            offset = sum(
                c * s for c, s in zip(idx, strides)
            ) * dtype_bytes
            if is_load:
                nc = ".nc" if op.attrs.get("nc") else ""
                template += (
                    f"ld.global{nc}.{suffix} "
                    f"${{{suffix}:operand.dst.r{flat_idx}}}, "
                    f"[{base_reg}+{offset}];\n"
                )
            else:
                template += (
                    f"st.global.{suffix} "
                    f"[{base_reg}+{offset}], "
                    f"${{{suffix}:operand.src.r{flat_idx}}};\n"
                )

        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_gmem_smem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower cooperative gmem ↔ smem move (block-collective).

        All threads participate: each handles a strided subset of b32
        elements. The flat smem index is decomposed into (row, col) to
        compute the gmem address using the gmem strides.

        The smem WAR/RAW hazards (prior readers/writers of the tile) are handled
        by insert_syncs, which emits a barrier before this move when needed.
        """
        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])

        is_load = isinstance(op.dst.type, SmemTensorType)
        smem_type = op.dst.type if is_load else op.src.type
        gmem_type = op.src.type if is_load else op.dst.type
        assert isinstance(smem_type, SmemTensorType)
        assert isinstance(gmem_type, GmemTensorType)
        gmem_label = "src" if is_load else "dst"
        smem_label = "dst" if is_load else "src"

        assert gmem_type.dtype == smem_type.dtype, (
            f"_lower_move_gmem_smem: dtype mismatch; "
            f"gmem {gmem_type.dtype} vs smem {smem_type.dtype}"
        )
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type, gmem_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(gmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        if is_load:
            ld_inst = f"ld.global.{suffix}"
            st_inst = f"st.shared.{suffix}"
            ld_addr = "${b64:co_gaddr}"
            st_addr = "${b32:co_saddr}"
        else:
            ld_inst = f"ld.shared.{suffix}"
            st_inst = f"st.global.{suffix}"
            ld_addr = "${b32:co_saddr}"
            st_addr = "${b64:co_gaddr}"

        co_dim = smem_type.rank - 1
        co_size = cast(int, smem_type.sizes[co_dim])
        gmem_co_stride_bytes = cast(int, gmem_type.strides[-1]) * dtype_bytes
        smem_co_stride_bytes = cast(int, smem_type.strides[-1]) * dtype_bytes

        template += textwrap.dedent(f"""\
            mov.u32 ${{b32:co_idx}}, %tid.x;
            mov.u32 ${{b32:co_ntid}}, %ntid.x;
            ${{label:co_idx_loop}}:
            setp.ge.u32 ${{pred:co_idx_done}}, ${{b32:co_idx}}, {co_size};
            @${{pred:co_idx_done}} bra ${{label:co_end}};
            mad.wide.u32 ${{b64:co_gpar}}, ${{b32:co_idx}}, {gmem_co_stride_bytes}, ${{b64:operand.{gmem_label}.ptr}};
            mul.lo.u32 ${{b32:co_spar}}, ${{b32:co_idx}}, {smem_co_stride_bytes};
            add.u32 ${{b32:co_spar}}, ${{b32:co_spar}}, ${{b32:operand.{smem_label}.ptr}};
        """)

        # Unroll sequential dims at lowering time. For large tiles this
        # may produce too many instructions; consider emitting a PTX loop.
        for seq_dim in range(smem_type.rank - 1):
            seq_size = cast(int, smem_type.sizes[seq_dim])
            gmem_seq_stride_bytes = (
                cast(int, gmem_type.strides[seq_dim]) * dtype_bytes
            )
            smem_seq_stride_bytes = (
                cast(int, smem_type.strides[seq_dim]) * dtype_bytes
            )
            for seq_idx in range(seq_size):
                g_off = seq_idx * gmem_seq_stride_bytes
                s_off = seq_idx * smem_seq_stride_bytes
                template += textwrap.dedent(f"""\
                    add.u64 ${{b64:co_gaddr}}, ${{b64:co_gpar}}, {g_off};
                    add.u32 ${{b32:co_saddr}}, ${{b32:co_spar}}, {s_off};
                    {ld_inst} ${{{suffix}:co_tmp}}, [{ld_addr}];
                    {st_inst} [{st_addr}], ${{{suffix}:co_tmp}};
                """)

        template += textwrap.dedent("""\
            add.u32 ${b32:co_idx}, ${b32:co_idx}, ${b32:co_ntid};
            bra ${label:co_idx_loop};
            ${label:co_end}:
        """)

        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_wgrmem_mma_to_smem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower wgrmem → smem store (per-thread scalar st.shared).

        Each thread stores its fragment registers to smem at addresses
        determined by the mma output layout (gid, lid from laneid).
        Same-dtype only — use convert first for type changes.
        """

        # TODO: this needs unique wgrmem layouts
        raise NotImplementedError("this needs unique wgrmem layouts")

        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])
        smem_type = dst_value.type
        wgrmem_type = src_value.type
        assert isinstance(smem_type, SmemTensorType)
        assert isinstance(wgrmem_type, WgrmemTensorType)
        assert smem_type.dtype == wgrmem_type.dtype, (
            f"_lower_move_wgrmem_to_smem: dtype mismatch; "
            f"smem {smem_type.dtype} vs wgrmem {wgrmem_type.dtype}. "
            f"Use convert first."
        )

        assert wgrmem_type.rank == 2, (
            f"_lower_move_wgrmem_to_smem: need rank 2; got {wgrmem_type.rank}"
        )
        m_size, n_size = wgrmem_type.sizes
        m_stride, n_stride = smem_type.strides
        assert n_stride == 1, (
            f"_lower_move_wgrmem_to_smem: smem col stride must be 1; got {n_stride}"
        )
        dtype_bytes = smem_type.dtype.byte_size
        suffix = _ptx_suffix(smem_type.dtype)
        m_stride_bytes = m_stride * dtype_bytes
        if m_size != 16 or n_size % 8 != 0:
            raise NotImplementedError(
                f"_lower_move_wgrmem_to_smem: need m=16, n%%8=0; got {m_size}x{n_size}"
            )

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = "${b32:operand.dst.ptr}"

        # mma.sync m16n8k16 output layout: each thread owns 4 elements
        # of a 16×8 tile, determined by gid = lane/4, lid = lane%4:
        #   - (gid,   lid*2)
        #   - (gid,   lid*2+1)
        #   - (gid+8, lid*2)
        #   - (gid+8, lid*2+1)
        # st_roff and st_coff are per-thread byte offsets from the smem
        # base pointer:
        #   - st_roff = gid * m_stride_bytes (row offset)
        #   - st_coff = lid * 2 * dtype_bytes (column pair offset)
        # The first address to store (for gid, lid*2) to is then:
        #   - st_addr = base + st_roff + st_coff
        # The unrolled stores add fixed byte offsets for the other 3 elements.
        st_coff_shift = 1 + int(math.log2(dtype_bytes))  # lid * 2_cols * dtype_bytes
        template += textwrap.dedent(f"""\
            mov.u32 ${{b32:st_lane}}, %laneid;
            shr.b32 ${{b32:st_gid}}, ${{b32:st_lane}}, 2;
            and.b32 ${{b32:st_lid}}, ${{b32:st_lane}}, 3;
            mul.lo.u32 ${{b32:st_roff}}, ${{b32:st_gid}}, {m_stride_bytes};
            shl.b32 ${{b32:st_coff}}, ${{b32:st_lid}}, {st_coff_shift};
            add.u32 ${{b32:st_addr}}, {base_reg}, ${{b32:st_roff}};
            add.u32 ${{b32:st_addr}}, ${{b32:st_addr}}, ${{b32:st_coff}};
        """)

        # 16-bit types (bf16/f16) are packed in pairs into b32 registers —
        # unpack each b32 into two b16 values before storing. 32-bit types
        # (f32) have one element per register and store directly.
        row8_off = 8 * m_stride_bytes
        reg_idx = 0
        if dtype_bytes == 2:
            for n_tile in range(n_size // 8):
                col_off = n_tile * 8 * dtype_bytes
                for m_half in range(2):
                    r_off = m_half * row8_off
                    src_reg = f"${{b32:operand.src.r{reg_idx}}}"
                    lo = f"${{{suffix}:st_lo}}"
                    hi = f"${{{suffix}:st_hi}}"
                    template += textwrap.dedent(f"""\
                        mov.b32 {{{lo}, {hi}}}, {src_reg};
                        st.shared.{suffix} [${{b32:st_addr}} + {col_off + r_off}], {lo};
                        st.shared.{suffix} [${{b32:st_addr}} + {col_off + r_off + dtype_bytes}], {hi};
                    """)
                    reg_idx += 1
        else:
            for n_tile in range(n_size // 8):
                col_off = n_tile * 8 * dtype_bytes
                for m_half in range(2):
                    r_off = m_half * row8_off
                    for elem in range(2):
                        e_off = elem * dtype_bytes
                        template += (
                            f"st.shared.{suffix} "
                            f"[${{b32:st_addr}} + {col_off + r_off + e_off}], "
                            f"${{{suffix}:operand.src.r{reg_idx}}};\n"
                        )
                        reg_idx += 1

        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []



    def _lower_move_wgrmem_wgmma(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        register_starts = _stmatrix_register_starts(op)
        if register_starts is None:
            return self._lower_move_wgrmem_wgmma_addrmem(op)
        return self._lower_move_wgrmem_wgmma_to_stmatrix(op, register_starts)

    def _lower_move_wgrmem_wgmma_addrmem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Store a wgmma fragment to, or load it from, address memory.

        The register layout and static element offsets are identical for both
        directions and memories; only the instruction, address width and PTX
        memory space differ. The memory is the warp group's own tile, so its
        shape matches the fragment -- the block-wide SMEM staging tile is tiled
        by warp-group id upstream (see promote_buffers._build_tma_writeback), so
        the row offset is already folded into the memory pointer.
        """
        is_load = isinstance(op.dst.type, WgrmemTensorType)
        fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
        memory_type = memory.type
        wgrmem_type = fragment.type
        assert isinstance(memory_type, (GmemTensorType, SmemTensorType))
        assert isinstance(wgrmem_type, WgrmemTensorType)
        assert wgrmem_type.layout == "wgmma_layout_0"
        assert wgrmem_type.rank == 2 and memory_type.rank == 2
        assert memory_type.dtype == wgrmem_type.dtype, (
            f"wgmma move: dtype mismatch; "
            f"memory {memory_type.dtype} vs wgrmem {wgrmem_type.dtype}"
        )

        m_dim, n_dim = cast(tuple[int, int], wgrmem_type.sizes)
        root = root_buffer(fragment)
        assert root is not None
        root_type = root.type
        assert isinstance(root_type, WgrmemTensorType)
        root_n = cast(int, root_type.sizes[1])
        if isinstance(fragment.definition, TileOp):
            assert list(memory_type.sizes) == [m_dim, n_dim], (
                f"wgmma move: memory shape {list(memory_type.sizes)} does "
                f"not match fragment {[m_dim, n_dim]}"
            )
            pairs = _wgmma_view_pairs(fragment)
        else:
            # Keep accepting the legacy producer until PromoteBuffers switches
            # from an attribute-based slice to an explicit WGRMEM view.
            col_start = op.attrs.get("wgmma_col_start", 0)
            assert isinstance(col_start, int)
            memory_m, memory_n = cast(tuple[int, int], memory_type.sizes)
            assert memory_m == m_dim and 0 <= col_start < n_dim
            assert col_start + memory_n <= n_dim
            pairs = [
                (pair, row, col - col_start)
                for pair, row, col in _wgmma_view_pairs(fragment)
                if col_start <= col < col_start + memory_n
            ]
        assert m_dim % 64 == 0, (
            f"wgmma store: m must be a multiple of 64; got {m_dim}"
        )
        assert root_n % 8 == 0 and 8 <= root_n <= 256, (
            f"wgmma store: n must be a multiple of 8 in [8, 256]; got {root_n}"
        )
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type, memory_type.dtype,
        ).byte_size
        row_stride = cast(int, memory_type.strides[0])
        assert row_stride % 2 == 0 and memory_type.strides[1] == 1, (
            f"wgmma move: element pairs must be contiguous and aligned in "
            f"memory; got strides {list(memory_type.strides)}"
        )
        row_stride_bytes = row_stride * dtype_bytes

        new_ops, byte_offset = _build_wgmma_layout_0_dynamic_offset(
            row_stride_bytes, dtype_bytes,
        )
        if isinstance(memory_type, GmemTensorType):
            template = textwrap.dedent(r"""
                cvt.u64.u32 ${b64:off64}, ${b32:operand.offset.val};
                add.u64 ${b64:base}, ${b64:operand.memory.ptr}, ${b64:off64};
            """)
            address = "${b64:base}"
            space = "global"
        else:
            template = (
                "add.u32 ${b32:base}, ${b32:operand.memory.ptr}, "
                "${b32:operand.offset.val};\n"
            )
            address = "${b32:base}"
            space = "shared"
        # A pair is one packed BF16/F16 register or two F32 registers.
        f32 = isinstance(wgrmem_type.dtype, F32Type)
        instruction = f"{'ld' if is_load else 'st'}.{space}{'.v2' if f32 else ''}.b32"

        for pair, di, dj in pairs:
            offset = di * row_stride_bytes + dj * dtype_bytes
            registers = (
                f"{{${{f32:operand.fragment.r{2 * pair}}}, "
                f"${{f32:operand.fragment.r{2 * pair + 1}}}}}"
                if f32 else f"${{b32:operand.fragment.r{pair}}}"
            )
            location = f"[{address}+{offset}]"
            template += (
                f"{instruction} {registers}, {location};\n" if is_load
                else f"{instruction} {location}, {registers};\n"
            )

        new_op = PTXTemplateOp.build(
            operands=[
                cast(Value, self.value_map[memory]),
                cast(Value, self.value_map[fragment]),
                byte_offset,
            ],
            result_types=[], template=template,
            operand_names=["memory", "fragment", "offset"], result_names=[],
            description=str(op),
        )
        new_ops.append(new_op)
        return new_ops, []

    def _lower_move_wgrmem_wgmma_to_stmatrix(
        self, op: MoveOp, register_starts: list[int],
    ) -> tuple[list[Op], list[object]]:
        dst = cast(Value, self.value_map[op.dst])
        src = cast(Value, self.value_map[op.src])
        dst_type = op.dst.type
        src_type = op.src.type
        assert isinstance(dst_type, SmemTensorType)
        assert isinstance(src_type, WgrmemTensorType)
        col_start = (
            _static_view_origin(op.src)[1]
            if isinstance(op.src.definition, TileOp)
            else op.attrs.get("wgmma_col_start", 0)
        )
        assert isinstance(col_start, int)
        src_root = root_buffer(op.src)
        dst_root = root_buffer(op.dst)
        assert src_root is not None and dst_root is not None
        src_root_type = src_root.type
        assert isinstance(src_root_type, WgrmemTensorType)
        n_dim = cast(int, src_root_type.sizes[1])
        root_type = dst_root.type
        assert isinstance(root_type, SmemTensorType)
        root_sizes = cast(tuple[int, ...], root_type.sizes)
        # TODO: Model writer ownership as a warpgroup-uniform IR condition so
        # loop peeling/unrolling can specialize each slice and lower the whole
        # gather to one branch instead of predicating every stmatrix.
        writer_wg = op.attrs.get("writer_wg")
        predicate = ""
        writer_setup = ""
        if writer_wg is not None:
            predicate = "@${pred:st_writer} "
            writer_setup = textwrap.dedent(f"""\
                mov.u32 ${{b32:st_full_tid}}, %tid.x;
                shr.u32 ${{b32:st_wg}}, ${{b32:st_full_tid}}, 7;
                setp.eq.u32 ${{pred:st_writer}}, ${{b32:st_wg}}, {writer_wg};
            """)
        template = writer_setup + textwrap.dedent("""\
            mov.u32 ${b32:st_lane}, %laneid;
            mov.u32 ${b32:st_tid}, %tid.x;
            and.b32 ${b32:st_tid}, ${b32:st_tid}, 127;
            shr.u32 ${b32:st_warp}, ${b32:st_tid}, 5;
            and.b32 ${b32:st_row}, ${b32:st_lane}, 7;
            mad.lo.u32 ${b32:st_row}, ${b32:st_warp}, 16, ${b32:st_row};
            shr.u32 ${b32:st_col}, ${b32:st_lane}, 3;
            shl.b32 ${b32:st_col}, ${b32:st_col}, 3;
            mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64, ${b32:st_col};
            shl.b32 ${b32:st_off}, ${b32:st_off}, 1;
        """)
        groups: list[tuple[list[int], int]] = []
        for i in range(0, len(register_starts), 2):
            first0 = register_starts[i]
            first1 = register_starts[i + 1]
            di, dj = _wgmma_pair_static_offset(first0, n_dim)
            next_di, next_dj = _wgmma_pair_static_offset(
                first1, n_dim,
            )
            assert next_di == di and next_dj == dj + 16
            offset0 = _stmatrix_store_static_offset(
                first0, n_dim, col_start, root_sizes[0],
            )
            offset1 = _stmatrix_store_static_offset(
                first0 + 1, n_dim, col_start, root_sizes[0],
            )
            groups.extend([
                ([first0, first0 + 2, first1, first1 + 2], offset0),
                (
                    [first0 + 1, first0 + 3, first1 + 1, first1 + 3],
                    offset1,
                ),
            ])
        for registers, physical_offset in groups:
            regs = ", ".join(
                f"${{b32:operand.src.r{i}}}" for i in registers
            )
            template += textwrap.dedent(f"""\
                add.u32 ${{b32:st_addr_off}}, ${{b32:st_off}}, {physical_offset};
                and.b32 ${{b32:st_swizzle_mask}}, ${{b32:st_addr_off}}, 896;
                shr.u32 ${{b32:st_swizzle_mask}}, ${{b32:st_swizzle_mask}}, 3;
                xor.b32 ${{b32:st_addr_off}}, ${{b32:st_addr_off}},
                    ${{b32:st_swizzle_mask}};
                add.u32 ${{b32:st_addr}}, ${{b32:operand.dst.ptr}},
                    ${{b32:st_addr_off}};
            """)
            template += (
                f"{predicate}stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
                f"[${{b32:st_addr}}], {{{regs}}};\n"
            )
        lowered = PTXTemplateOp.build(
            operands=[dst, src], result_types=[], template=template,
            operand_names=["dst", "src"], result_names=[],
            description=str(op),
        )
        return [lowered], []

    def _lower_move_smem_to_wgrmem_mma(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower smem → wgrmem move to ldmatrix.sync (warp-collective).

        All 32 lanes participate: each provides an smem row address,
        and the hardware shuffles data across lanes into the mma
        fragment layout. Uses `.trans` when `attrs={trans: True}`
        (B operands needing col-major fragments from row-major smem).
        """

        # TODO: this needs unique wgrmem layouts
        raise NotImplementedError("this needs unique wgrmem layouts")

        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])
        wgrmem_type = dst_value.type
        smem_type = src_value.type
        assert isinstance(wgrmem_type, WgrmemTensorType)
        assert isinstance(smem_type, SmemTensorType)

        dtype_bytes = smem_type.dtype.byte_size
        m_stride_bytes = smem_type.strides[0] * dtype_bytes
        trans = ".trans" if op.attrs.get("trans") else ""
        n_regs = wgrmem_type.num_elements // (wgrmem_type.warps * 32 * 2)

        if n_regs == 4:
            x_count = "x4"
        elif n_regs == 2:
            x_count = "x2"
        elif n_regs == 1:
            x_count = "x1"
        else:
            raise NotImplementedError(
                f"_lower_move_smem_to_wgrmem: unsupported register count {n_regs}"
            )

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        reg_list = ", ".join(
            f"${{b32:operand.dst.r{i}}}" for i in range(n_regs)
        )

        template = ""
        base_reg = "${b32:operand.src.ptr}"

        template += (
            f"mov.u32 ${{b32:lm_lane}}, %laneid;\n"
            f"and.b32 ${{b32:lm_row}}, ${{b32:lm_lane}}, 15;\n"
            f"shr.b32 ${{b32:lm_half}}, ${{b32:lm_lane}}, 4;\n"
            f"mul.lo.u32 ${{b32:lm_off}}, ${{b32:lm_row}}, {m_stride_bytes};\n"
            f"shl.b32 ${{b32:lm_hb}}, ${{b32:lm_half}}, 4;\n"
            f"add.u32 ${{b32:lm_off}}, ${{b32:lm_off}}, ${{b32:lm_hb}};\n"
            f"add.u32 ${{b32:lm_addr}}, {base_reg}, ${{b32:lm_off}};\n"
            f"ldmatrix.sync.aligned.m8n8.{x_count}{trans}.shared.b16"
            f" {{{reg_list}}}, [${{b32:lm_addr}}];\n"
        )

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_smem_rmem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower per-thread move between smem and rmem (either direction).

        Each thread independently reads/writes its own elements via
        ld.shared / st.shared with 32-bit smem addressing.

        The smem RAW/WAR hazards (prior writers/readers of the tile) are handled
        by insert_syncs; when the producer is a TMA load, its await orders this.
        """
        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])

        is_load = isinstance(op.dst.type, RmemTensorType)
        smem_ir_type = op.src.type if is_load else op.dst.type
        rmem_type = dst_value.type if is_load else src_value.type
        assert isinstance(smem_ir_type, SmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        smem_label = "src" if is_load else "dst"
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type, smem_ir_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = f"${{b32:operand.{smem_label}.ptr}}"

        for flat_idx, idx in enumerate(
            itertools.product(
                *[range(s) for s in cast(tuple[int, ...], rmem_type.sizes)]
            )
        ):
            strides = cast(tuple[int, ...], smem_ir_type.strides)
            offset = sum(
                c * s for c, s in zip(idx, strides)
            ) * dtype_bytes
            reg = f"${{{suffix}:operand.{'dst' if is_load else 'src'}.r{flat_idx}}}"
            if is_load:
                template += f"ld.shared.{suffix} {reg}, [{base_reg}+{offset}];\n"
            else:
                template += f"st.shared.{suffix} [{base_reg}+{offset}], {reg};\n"

        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_async_tma_store(
        self, op: TmaMoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower smem → gmem TMA bulk store (block-collective).

        Thread 0 issues cp.async.bulk.tensor.2d and commits the group.
        The barrier inserted before the async orders all threads' shared-memory
        writes before thread 0 starts the store.
        """
        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])
        idx_values = [cast(Value, self.value_map[v]) for v in op.indices]
        desc_type = dst_value.type
        assert isinstance(desc_type, TmaDescType)

        operands = [dst_value, src_value] + idx_values
        operand_names = ["dst", "src"] + [f"idx{i}" for i in range(len(idx_values))]
        coordinate_lines, coordinates = _tma_coordinates(
            desc_type, [f"idx{i}" for i in range(len(idx_values))], "store",
        )
        tensor_rank = len(coordinates)
        indices = ", ".join(coordinates)
        coordinate_code = "\n".join(coordinate_lines)
        tma_instruction = self._tma_instruction(
            op,
            f"cp.async.bulk.tensor.{tensor_rank}d.global.shared::cta.bulk_group",
            f"[${{b64:operand.dst.ptr}}, {{{indices}}}], "
            f"[${{b32:operand.src.ptr}}]",
            operands,
            operand_names,
            "cache_policy",
        )

        # TODO: The logic to test for thread 0 could be expressed in index compute to cse etc.
        # The RAW (all threads finish writing the smem tile before t0 stores it) is
        # handled by insert_syncs, which emits a barrier before this async.

        template = textwrap.dedent(f"""\
            mov.u32 ${{b32:st_tid}}, %tid.x;
            mov.u32 ${{b32:st_tidy}}, %tid.y;
            or.b32 ${{b32:st_tid}}, ${{b32:st_tid}}, ${{b32:st_tidy}};
            mov.u32 ${{b32:st_tidz}}, %tid.z;
            or.b32 ${{b32:st_tid}}, ${{b32:st_tid}}, ${{b32:st_tidz}};
            setp.ne.s32 ${{pred:st_not_t0}}, ${{b32:st_tid}}, 0;
            @${{pred:st_not_t0}} bra ${{label:st_done}};
            fence.proxy.async.shared::cta;
            {coordinate_code}
            {tma_instruction}
            cp.async.bulk.commit_group;
            ${{label:st_done}}:
        """)

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []


    def _lower_async_tma_loads(
        self, op: AsyncOp, tmas: list[TmaMoveOp], lock_value: Value,
    ) -> tuple[list[Op], list[object]]:
        """Lower one or more gmem → smem TMA loads sharing a single mbarrier.

        Each load issues cp.async.bulk.tensor against the shared mbarrier, then a
        single mbarrier.arrive.expect_tx covers their summed byte count — so a
        fused lhs+rhs pair (or any run of loads) needs one mbarrier and one
        arrive instead of one per load (fastcu's single-``full``-barrier idea).
        Thread 0 issues; the await op emits the wait.
        """
        lock_type = lock_value.type
        assert isinstance(lock_type, LockType)
        assert lock_type.kind == "mbar", (
            f"_lower_async_tma_loads: expected lock kind 'mbar'; got "
            f"{lock_type.kind!r}"
        )

        operands: list[Value] = [lock_value]
        operand_names: list[str] = ["lock"]
        multicast_mask: int | None = None
        if op.is_tma_multicast:
            kernel = find_ancestor_of_type(op, KernelOp)
            assert kernel is not None
            cluster_dim = kernel.cluster_dim
            cluster_size = math.prod(dim or 1 for dim in cluster_dim)
            multicast_mask = (1 << cluster_size) - 1
        total_bytes = 0
        load_lines: list[str] = []
        for i, tma in enumerate(tmas):
            dst_value = cast(Value, self.value_map[tma.dst])
            desc_value = cast(Value, self.value_map[tma.src])
            assert isinstance(desc_value.type, TmaDescType), (
                f"_lower_async_tma_loads: load src must be a TMA descriptor; "
                f"got {desc_value.type}"
            )
            desc_type = desc_value.type
            total_bytes += (
                desc_type.tile_num_elements
                * cast(
                    BF16Type | F16Type | F32Type | F64Type,
                    desc_type.gmem_type.dtype,
                ).byte_size
            )
            idx_values = [
                cast(Value, self.value_map[v]) for v in tma.indices
            ]
            dn, sn = f"dst{i}", f"desc{i}"
            idx_names = [f"idx{i}_{j}" for j in range(len(idx_values))]
            operands += [dst_value, desc_value] + idx_values
            operand_names += [dn, sn] + idx_names
            coordinate_lines, coordinates = _tma_coordinates(
                desc_type, idx_names, f"load{i}",
            )
            tensor_rank = len(coordinates)
            indices = ", ".join(coordinates)
            load_lines.extend(coordinate_lines)
            opcode = (
                f"cp.async.bulk.tensor.{tensor_rank}d.shared::cluster.global."
                "mbarrier::complete_tx::bytes"
            )
            arguments = (
                f"[${{b32:operand.{dn}.ptr}}], "
                f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                f"[${{b32:operand.lock.val}}]"
            )
            if tma.multicast is not True:
                load_lines.append(self._tma_instruction(
                    tma,
                    opcode,
                    arguments,
                    operands,
                    operand_names,
                    f"cache_policy{i}",
                ))
            else:
                assert multicast_mask is not None
                load_lines.extend([
                    "mov.u32 ${b32:cluster_rank}, %cluster_ctarank;",
                    "setp.ne.u32 ${pred:not_cluster_leader}, "
                    "${b32:cluster_rank}, 0;",
                    f"@${{pred:not_cluster_leader}} bra "
                    f"${{label:after_multicast_{i}}};",
                    self._tma_instruction(
                        tma,
                        f"{opcode}.multicast::cluster",
                        f"{arguments}, {multicast_mask}",
                        operands,
                        operand_names,
                        f"cache_policy{i}",
                    ),
                    f"${{label:after_multicast_{i}}}:",
                ])

        # Plain-string lines (literal ${...} placeholders); only load_lines and
        # the byte count are interpolated, so we avoid f-string/dedent brace and
        # multi-line-indent pitfalls by joining a line list.
        #
        # The buffer-reuse WAR (t0's reload vs prior readers of the smem tile) is
        # handled by insert_syncs, which emits a barrier before this async.
        lines = [
            "mov.u32 ${b32:tid_x}, %tid.x;",
            "mov.u32 ${b32:tid_y}, %tid.y;",
            "or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_y};",
            "mov.u32 ${b32:tid_z}, %tid.z;",
            "or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_z};",
            "setp.ne.s32 ${pred:not_t0}, ${b32:tid_x}, 0;",
            "@${pred:not_t0} bra ${label:after_load};",
            *load_lines,
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
            f"[${{b32:operand.lock.val}}], {total_bytes};",
            "${label:after_load}:",
        ]
        template = "\n".join(lines) + "\n"

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []



def _build_wgmma_layout_0_dynamic_offset(
    row_stride_bytes: int,
    dtype_bytes: int,
) -> tuple[list[Op], Value]:
    """Emit index ops for the wgmma_layout_0 per-thread byte offset.

    warp_in_wg = (tid.x % 128) / 32
    lane = tid.x % 32
    row = warp_in_wg * 16 + lane / 4
    col = (lane % 4) * 2
    byte_offset = row * row_stride_bytes + col * dtype_bytes
    """
    ops: list[Op] = []

    tid_op = ThreadIdOp.build(CollectiveAxis.X)
    ops.append(tid_op)
    tid_x = tid_op.value

    c128 = IndexConstantOp.build(128)
    ops.append(c128)
    tid_in_wg = IndexModOp.build(tid_x, c128.result())
    ops.append(tid_in_wg)

    c32 = IndexConstantOp.build(32)
    ops.append(c32)
    warp = IndexDivOp.build(tid_in_wg.result(), c32.result())
    ops.append(warp)
    lane = IndexModOp.build(tid_in_wg.result(), c32.result())
    ops.append(lane)

    # row = warp * 16 + lane / 4
    c16 = IndexConstantOp.build(16)
    ops.append(c16)
    warp_times_16 = IndexMulOp.build(warp.result(), c16.result())
    ops.append(warp_times_16)
    c4 = IndexConstantOp.build(4)
    ops.append(c4)
    lane_div_4 = IndexDivOp.build(lane.result(), c4.result())
    ops.append(lane_div_4)
    row = IndexAddOp.build(warp_times_16.result(), lane_div_4.result())
    ops.append(row)

    # col = (lane % 4) * 2
    lane_mod_4 = IndexModOp.build(lane.result(), c4.result())
    ops.append(lane_mod_4)
    c2 = IndexConstantOp.build(2)
    ops.append(c2)
    col = IndexMulOp.build(lane_mod_4.result(), c2.result())
    ops.append(col)

    # byte_offset = row * row_stride_bytes + col * dtype_bytes
    row_stride = IndexConstantOp.build(row_stride_bytes)
    ops.append(row_stride)
    row_bytes = IndexMulOp.build(row.result(), row_stride.result())
    ops.append(row_bytes)
    col_stride = IndexConstantOp.build(dtype_bytes)
    ops.append(col_stride)
    col_bytes = IndexMulOp.build(col.result(), col_stride.result())
    ops.append(col_bytes)
    byte_offset = IndexAddOp.build(row_bytes.result(), col_bytes.result())
    ops.append(byte_offset)

    return ops, byte_offset.result()


def _wgmma_pair_static_offset(pair_index: int, n_dim: int) -> tuple[int, int]:
    """Thread-independent root offset of a WGMMA pair's first element.

    The runtime warp/lane offset selects the thread's base coordinate. This
    returns the additional logical ``(row, column)`` offset for ``pair_index``
    in that thread's register fragment.
    """
    assert n_dim % 8 == 0
    pairs_per_m_tile = n_dim // 4
    m, pair_in_m_tile = divmod(pair_index, pairs_per_m_tile)
    w, pair_in_16_columns = divmod(pair_in_m_tile, 4)
    di, dj = [
        (0, 0), (8, 0), (0, 8), (8, 8)
    ][pair_in_16_columns]
    return (m * 64 + di, w * 16 + dj)


def _static_view_origin(value: Value) -> tuple[int, ...]:
    """Logical element coordinates of a static view's origin in its root."""
    value_type = value.type
    assert isinstance(value_type, WgrmemTensorType)
    offsets = [0] * value_type.rank
    while isinstance(value.definition, TileOp):
        tile = value.definition
        for dim, index in enumerate(tile.static_indices):
            if index is None:
                raise NotImplementedError(
                    "WGRMEM views currently require static tile indices"
                )
            result_type = tile.result().type
            assert isinstance(result_type, WgrmemTensorType)
            offsets[dim] += index * cast(int, result_type.sizes[dim])
        value = tile.arg
    return tuple(offsets)


def _wgmma_view_pairs(value: Value) -> list[tuple[int, int, int]]:
    """Map a WGMMA WGRMEM value or view to pairs in its root.

    Each tuple is ``(root_pair_index, view_row_offset, view_column_offset)``.
    Pair ``p`` names F32 registers ``2*p`` and ``2*p + 1``, or the packed
    BF16/F16 b32 register ``p``. The offsets are the thread-independent
    logical coordinates of the pair's first element relative to the view;
    store lowering adds the runtime warp/lane coordinate separately.
    """
    root = root_buffer(value)
    assert root is not None
    view_type, root_type = value.type, root.type
    assert isinstance(view_type, WgrmemTensorType)
    assert isinstance(root_type, WgrmemTensorType)
    assert view_type.layout == root_type.layout == "wgmma_layout_0"
    assert view_type.rank == root_type.rank == 2
    assert view_type.warps == root_type.warps == 4
    assert view_type.dtype == root_type.dtype
    assert isinstance(view_type.dtype, (F32Type, BF16Type, F16Type)), (
        f"WGMMA views support f32, bf16, and f16; got {view_type.dtype}"
    )
    assert root_type.num_elements_per_thread % 2 == 0
    row_start, col_start = _static_view_origin(value)
    view_m, view_n = cast(tuple[int, int], view_type.sizes)
    root_n = cast(int, root_type.sizes[1])
    pairs = [
        (pair_index, row - row_start, col - col_start)
        for pair_index in range(root_type.num_elements_per_thread // 2)
        for row, col in [
            _wgmma_pair_static_offset(pair_index, root_n)
        ]
        if row_start <= row < row_start + view_m
        and col_start <= col < col_start + view_n
    ]
    assert len(pairs) == view_m * view_n // (root_type.warps * 64)
    return pairs


def _view_pair_indices(value: Value) -> list[int]:
    """Return per-thread root pair indices selected by a WGRMEM value or view."""
    assert isinstance(value.type, WgrmemTensorType)
    if value.type.layout == "wgmma_layout_0":
        return [pair for pair, _, _ in _wgmma_view_pairs(value)]
    if isinstance(value.definition, TileOp):
        raise NotImplementedError(
            f"WGRMEM views are unsupported for layout {value.type.layout!r}"
        )
    count = value.type.num_elements_per_thread
    assert count % 2 == 0, (
        f"WGRMEM pair operations require an even number of elements per thread; "
        f"got {count}"
    )
    return list(range(count // 2))


def _view_f32_register_indices(value: Value) -> list[int]:
    """Return root F32 register indices covered by a WGRMEM value or view."""
    value_type = value.type
    assert isinstance(value_type, WgrmemTensorType)
    assert isinstance(value_type.dtype, F32Type)
    return [
        2 * pair + element
        for pair in _view_pair_indices(value)
        for element in (0, 1)
    ]


def _stmatrix_store_static_offset(
    pair_index: int, n_dim: int, col_start: int, dst_m: int,
) -> int:
    """Byte offset for a pair in the selected 64-column blocked layout."""
    di, dj = _wgmma_pair_static_offset(pair_index, n_dim)
    relative_col = dj - col_start
    assert relative_col >= 0 and relative_col % 32 == 0
    block_offset = relative_col // 64 * dst_m * 64
    return (block_offset + di * 64 + relative_col % 64) * 2


def _tma_coordinates(
    desc_type: TmaDescType,
    index_names: list[str],
    prefix: str,
) -> tuple[list[str], list[str]]:
    layout = physical_layout(desc_type)
    if layout is None:
        return [], [
            f"${{b32:operand.{name}.val}}"
            for name in reversed(index_names)
        ]

    assert desc_type.gmem_type.rank == 2
    assert len(index_names) == 2
    row = f"${{b32:operand.{index_names[0]}.val}}"
    col = f"${{b32:operand.{index_names[1]}.val}}"
    block = f"${{b32:{prefix}_block}}"
    inner = f"${{b32:{prefix}_inner}}"
    block_width, order = layout
    lines = [
        f"div.u32 {block}, {col}, {block_width};",
        f"rem.u32 {inner}, {col}, {block_width};",
    ]
    if order == "row_major":
        coordinates = [inner, block, row]
    else:
        coordinates = [inner, row, block]
    return lines, coordinates


def _stmatrix_register_starts(op: MoveOp) -> list[int] | None:
    dst_type = op.dst.type
    src_type = op.src.type
    layout = (
        physical_layout(dst_type)
        if isinstance(dst_type, SmemTensorType) else None
    )
    if not (
        isinstance(dst_type, SmemTensorType)
        and isinstance(src_type, WgrmemTensorType)
        and src_type.layout == "wgmma_layout_0"
        and isinstance(src_type.dtype, (BF16Type, F16Type))
        and dst_type.dtype == src_type.dtype
        and dst_type.rank == 2
        and dst_type.strides[-1] == 1
        and dst_type.swizzle == "128B"
        and layout == (64, "block_major")
    ):
        return None

    dst_root = root_buffer(op.dst)
    assert dst_root is not None
    root_type = dst_root.type
    if (
        not isinstance(root_type, SmemTensorType)
        or root_type.rank != 2
        or not root_type.is_canonical_strides
        or cast(int, root_type.sizes[0]) % 64
    ):
        return None

    m_dim, n_dim = cast(tuple[int, int], src_type.sizes)
    src_root = root_buffer(op.src)
    assert src_root is not None
    src_root_type = src_root.type
    assert isinstance(src_root_type, WgrmemTensorType)
    root_n = cast(int, src_root_type.sizes[1])
    is_view = isinstance(op.src.definition, TileOp)
    if is_view:
        row_start, col_start = _static_view_origin(op.src)
        pairs = _wgmma_view_pairs(op.src)
    else:
        row_start = 0
        col_start = op.attrs.get("wgmma_col_start", 0)
        assert isinstance(col_start, int)
        dst_n = cast(int, dst_type.sizes[1])
        pairs = [
            (pair, row, col - col_start)
            for pair, row, col in _wgmma_view_pairs(op.src)
            if col_start <= col < col_start + dst_n
        ]
    dst_m, dst_n = cast(tuple[int, int], dst_type.sizes)
    if (
        dst_m != m_dim
        or (is_view and dst_n != n_dim)
        or row_start != 0
        or col_start < 0
        or col_start % 16
        or (not is_view and col_start + dst_n > n_dim)
        or m_dim % 64
        or dst_n % 64
        or root_n % 16
        or not (8 <= root_n <= 256)
    ):
        return None

    if len(pairs) % 4:
        return None
    register_starts = [pair for pair, _, _ in pairs if pair % 4 == 0]
    return register_starts if len(register_starts) % 2 == 0 else None


def _ptx_suffix(dtype) -> str:
    """PTX type suffix for a dtype. bf16 uses b16 (no native suffix)."""
    name = dtype.name
    return "b16" if name == "bf16" else name


def _mov_literals_ptx(result: str, dtype: Type, values) -> str:
    """Move literals into a result's registers; 16-bit floats move as bits."""
    suffix = "b16" if isinstance(dtype, (F16Type, BF16Type)) else dtype.name
    return "".join(
        f"mov.{suffix} ${{{suffix}:result.{result}.r{i}}}, "
        f"{_ptx_literal(dtype, value)};\n"
        for i, value in enumerate(values)
    )


def _ptx_literal(dtype, value) -> str:
    name = dtype.name
    if name == "f32":
        return "0f" + struct.pack(">f", float(value)).hex().upper()
    if name == "f64":
        return "0d" + struct.pack(">d", float(value)).hex().upper()
    if name == "bf16":
        bits = struct.unpack(">I", struct.pack(">f", float(value)))[0] >> 16
        return f"0x{bits:04X}"
    if name == "f16":
        return "0x" + struct.pack(">e", float(value)).hex().upper()
    raise NotImplementedError(f"_ptx_literal: dtype {name}")


def lower_to_ptx_template(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToPtxTemplate().run(module)
