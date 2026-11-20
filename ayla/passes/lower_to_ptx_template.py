"""Lower high-level GPU-dialect ops to ptx_template ops.

Implements `LowerToPtxTemplate`, a `LoweringPass` subclass that
dispatches to per-op rewriter methods. Today only `tile` is lowered;
other ops (alloc, load, store, matmul) will be added incrementally.
"""

from __future__ import annotations

from ..cluster_layout import rank_table_ptx, pair_mask, multicast_sources

from collections.abc import Callable
from functools import partial
import itertools
import math
import struct
import textwrap
from typing import cast

from ..base.ir import Block, Op, Type, Value, clone_op, find_ancestor, find_ancestor_of_type, find_ops_of_type, iter_ops
from ..base.lowering_pass import LoweringPass, lowers
from ..tmem_layout import TMEM_FRAGMENT_LAYOUTS, TmemWarpLayout
from ..gpu_dialect import (
    ForOp,
    AddrMemTensorType,
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BF16Type,
    CollectiveAxis,
    DeallocOp,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexLtOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    LockType,
    LockViewOp,
    OperandPipelineOp,
    MatmulOp,
    MmaDescriptorType,
    MmaDescriptorOp,
    MmaDescriptorLowOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    PtrAddOp,
    PtrType,
    PTXTemplateOp,
    PredicateType,
    ReinterpretOp,
    RegMemTensorType,
    ReshapeOp,
    RmemTensorType,
    execution_issuer_warp,
    execution_warps,
    ConstantOp,
    SmemTensorType,
    ThreadIdOp,
    IfOp,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    TmemTensorType,
    ToPtrOp,
    WarpIdsOp,
    WgrmemTensorType,
    is_column_major,
    is_side_effect_free,
    physical_layout,
    root_buffer,
)


def _ptx_masked_half_pair(pair, di, dj, row_stride, bounds, *, is_load, writer):
    """Bounds-check a Half pair; use a packed access only when both halves fit."""
    rows, columns = bounds
    offset = di * row_stride + dj * 2
    reg = f"${{b32:operand.fragment.r{pair}}}"
    template = textwrap.dedent(f"""
        add.u32 ${{b32:r{pair}}}, ${{b32:row}}, {di};
        add.u32 ${{b32:c{pair}}}, ${{b32:col}}, {dj};
        add.u32 ${{b32:cnext{pair}}}, ${{b32:c{pair}}}, 1;
        setp.lt.u32 ${{pred:row{pair}}}, ${{b32:r{pair}}}, {rows};
        setp.lt.u32 ${{pred:first{pair}}}, ${{b32:c{pair}}}, {columns};
        setp.lt.u32 ${{pred:second{pair}}}, ${{b32:cnext{pair}}}, {columns};
        and.pred ${{pred:first{pair}}}, ${{pred:first{pair}}}, ${{pred:row{pair}}};
        and.pred ${{pred:second{pair}}}, ${{pred:second{pair}}}, ${{pred:row{pair}}};
    """)
    if writer:
        template += f"and.pred ${{pred:first{pair}}}, ${{pred:first{pair}}}, ${{pred:writer}};\n"
        template += f"and.pred ${{pred:second{pair}}}, ${{pred:second{pair}}}, ${{pred:writer}};\n"
    template += textwrap.dedent(f"""
        and.pred ${{pred:wide{pair}}}, ${{pred:second{pair}}}, ${{pred:aligned}};
        not.pred ${{pred:narrow{pair}}}, ${{pred:wide{pair}}};
        and.pred ${{pred:low{pair}}}, ${{pred:first{pair}}}, ${{pred:narrow{pair}}};
        and.pred ${{pred:high{pair}}}, ${{pred:second{pair}}}, ${{pred:narrow{pair}}};
    """)
    lo, hi = f"${{b16:lo{pair}}}", f"${{b16:hi{pair}}}"
    if is_load:
        template += f"mov.b16 {lo}, 0;\nmov.b16 {hi}, 0;\n"
        template += f"@${{pred:low{pair}}} ld.global.b16 {lo}, [${{b64:base}}+{offset}];\n"
        template += f"@${{pred:high{pair}}} ld.global.b16 {hi}, [${{b64:base}}+{offset+2}];\n"
        template += f"mov.b32 {reg}, {{{lo}, {hi}}};\n"
        template += f"@${{pred:wide{pair}}} ld.global.b32 {reg}, [${{b64:base}}+{offset}];\n"
    else:
        template += f"mov.b32 {{{lo}, {hi}}}, {reg};\n"
        template += f"@${{pred:wide{pair}}} st.global.b32 [${{b64:base}}+{offset}], {reg};\n"
        template += f"@${{pred:low{pair}}} st.global.b16 [${{b64:base}}+{offset}], {lo};\n"
        template += f"@${{pred:high{pair}}} st.global.b16 [${{b64:base}}+{offset+2}], {hi};\n"
    return template


def _ptx_store_unaligned_wgmma_half_pairs(pairs, row_stride, columns):
    """Repack complete WGMMA rows into aligned stores, with scalar row endpoints.

    Four adjacent lanes own consecutive Half pairs. Rotate each pair by one
    Half using its next lane; lane three reads the next register group's pair.
    The participating mask is captured by the caller before divergence.
    """
    template = textwrap.dedent(r"""
        mov.u32 ${b32:lane}, %laneid;
        and.b32 ${b32:row_lane}, ${b32:lane}, 3;
        and.b32 ${b32:row_base}, ${b32:lane}, 4294967292;
        add.u32 ${b32:next_row_lane}, ${b32:row_lane}, 1;
        and.b32 ${b32:next_row_lane}, ${b32:next_row_lane}, 3;
        or.b32 ${b32:next_lane}, ${b32:row_base}, ${b32:next_row_lane};
        setp.eq.u32 ${pred:first_lane}, ${b32:row_lane}, 0;
        setp.eq.u32 ${pred:last_lane}, ${b32:row_lane}, 3;
    """)
    pair_by_coordinate = {(di, dj): pair for pair, di, dj in pairs}
    for pair, di, dj in pairs:
        offset = di * row_stride + dj * 2
        reg = f"${{b32:operand.fragment.r{pair}}}"
        neighbor_pair = pair_by_coordinate.get((di, dj + 8))
        next_reg = f"${{b32:operand.fragment.r{neighbor_pair}}}" if neighbor_pair is not None else "0"
        template += f"selp.b32 ${{b32:shift_src{pair}}}, {next_reg}, {reg}, ${{pred:first_lane}};\n"
        template += f"shfl.sync.idx.b32 ${{b32:neighbor{pair}}}, ${{b32:shift_src{pair}}}, ${{b32:next_lane}}, 31, ${{b32:narrow_mask}};\n"
        template += f"shf.r.wrap.b32 ${{b32:shift_pair{pair}}}, {reg}, ${{b32:neighbor{pair}}}, 16;\n"
        last_group = dj + 8 == columns
        predicate = "@!${pred:last_lane} " if last_group else ""
        template += f"{predicate}st.global.b32 [${{b64:base}}+{offset+2}], ${{b32:shift_pair{pair}}};\n"
        if dj == 0 or last_group:
            lo, hi = f"${{b16:fast_lo{pair}}}", f"${{b16:fast_hi{pair}}}"
            template += f"mov.b32 {{{lo}, {hi}}}, {reg};\n"
            if dj == 0:
                template += f"@${{pred:first_lane}} st.global.b16 [${{b64:base}}+{offset}], {lo};\n"
            if last_group:
                template += f"@${{pred:last_lane}} st.global.b16 [${{b64:base}}+{offset+2}], {hi};\n"
    return template


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
        raise NotImplementedError(f"lower_elementwise: mul does not support {dtype}")
    return "".join(
        f"{opcode} {dst}, {lhs}, {rhs};\n"
        for dst, lhs, rhs in zip(
            dst_regs,
            lhs_regs,
            rhs_regs,
            strict=True,
        )
    )


def _lower_elementwise_sigmoid_ptx(
    dst_regs: list[str],
    src_regs: list[str],
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
    dst_regs: list[str],
    src_regs: list[str],
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


def _register_operand_regs(value: Value, operand_name: str) -> tuple[str, list[str]]:
    """Return the PTX storage type and registers selected by a value or view.

    WGRMEM holds each element pair in one packed 16-bit register or in two F32
    registers; a view selects the pairs it covers in its root fragment.
    """
    value_type = cast(RegMemTensorType, value.type)
    suffix = _ptx_suffix(value_type.dtype)
    if isinstance(value_type, RmemTensorType):
        indices = range(cast(int, value_type.num_elements))
    elif suffix == "b16":
        suffix, indices = "b32", _view_pair_indices(value)
    else:
        indices = _view_f32_register_indices(value)
    return suffix, [f"${{{suffix}:operand.{operand_name}.r{reg}}}" for reg in indices]


def _lower_fill_ptx(op: ElementwiseOp) -> str:
    """Broadcast a scalar source into every destination register."""
    suffix, dst_regs = _register_operand_regs(op.dst, "dst")
    src_suffix, [scalar] = _register_operand_regs(op.src(0), "src")
    # Storage types differ only when a destination register packs a pair.
    if suffix != src_suffix:
        scalar = f"{{{scalar}, {scalar}}}"
    return "".join(f"mov.{suffix} {dst_reg}, {scalar};\n" for dst_reg in dst_regs)


def _zero_dst_ptx(op: MatmulOp, registers: int) -> str:
    """Zero an overwriting matmul's f32 destination ahead of accumulating."""
    if op.accumulate:
        return ""
    return "".join(
        f"mov.f32 ${{f32:operand.dst.r{reg}}}, 0f00000000;\n"
        for reg in range(registers)
    )


def _elementwise_operand_regs(value: Value, operand_name: str) -> list[str]:
    return _register_operand_regs(value, operand_name)[1]


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
            dst_regs,
            src_regs[::2],
            src_regs[1::2],
            strict=True,
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
    for index, (src, dst_lo, dst_hi) in enumerate(
        zip(
            src_regs,
            dst_regs[::2],
            dst_regs[1::2],
            strict=True,
        )
    ):
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
    if isinstance(src_dtype, F32Type) and isinstance(dst_dtype, (BF16Type, F16Type)):
        return _lower_f32_to_packed16_ptx(
            dst_regs,
            src_regs,
            src_dtype,
            dst_dtype,
        )
    raise NotImplementedError(
        f"lower_elementwise: unsupported conversion {src_dtype} -> {dst_dtype}"
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
    lower_registers: Callable[[ElementwiseOp, list[str], list[list[str]]], str],
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
            result_regs = [f"${{f32:{op.fn}_result{i}}}" for i in range(batch_size)]
        template += lower_registers(op, result_regs, batch_src_regs)
        if src_dtype != dst_dtype:
            # TODO: Have PromoteBuffers emit an explicit convert after the F32
            # epilogue; only promotion knows the register accumulator exists.
            template += _lower_dtype_conversion_ptx(
                dst_regs[start // conversion_width : end // conversion_width],
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
    if any(cast(MemTensorType, value.type).dtype != dtype for value in op.operands):
        raise NotImplementedError(
            "lower_elementwise: mul requires matching operand dtypes"
        )
    lhs_regs, rhs_regs = src_regs
    return _lower_mul_registers(dst_regs, lhs_regs, rhs_regs, dtype)


def _lower_add_register_batches(op, dst_regs, src_regs):
    if any(not isinstance(cast(MemTensorType, value.type).dtype, F32Type)
           for value in op.operands):
        raise NotImplementedError("elementwise add currently requires FP32 registers")
    lhs, rhs = src_regs
    return "".join(f"add.f32 {dst}, {a}, {b};\n"
                   for dst, a, b in zip(dst_regs, lhs, rhs, strict=True))


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
        if op.dst is src
        else dst_regs
    )
    template = _sigmoid_ptx(dst_dtype)(tmp_regs, src_regs)
    template += _lower_mul_registers(dst_regs, src_regs, tmp_regs, F32Type())
    return template


_ELEMENTWISE_LOWERERS = {
    "add": partial(_lower_elementwise_in_batches, lower_registers=_lower_add_register_batches),
    "convert": _lower_convert_elementwise_ptx,
    "fill": _lower_fill_ptx,
    "mul": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_mul_register_batches,
    ),
    "sigmoid": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_sigmoid_registers,
    ),
    "silu": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_silu_registers,
    ),
}


def _leader_warp(prefix: str, issuer_warp: int = 0, *others: int) -> str:
    """PTX setting ``${pred:<prefix>_other_warp}`` outside ``issuer_warp`` and ``others``.

    Each warp index component is broadcast from the warp's first lane
    (CUTLASS's canonical_warp_idx_sync), so ptxas proves the predicate
    warp-uniform; a broadcast of a combined value would not fold away.
    """
    return textwrap.dedent(f"""\
        mov.u32 ${{b32:{prefix}_warp}}, %tid.x;
        shr.u32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, 5;
        shfl.sync.idx.b32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, 0, 31, 0xffffffff;
        mov.u32 ${{b32:{prefix}_warp_yz}}, %tid.y;
        shfl.sync.idx.b32 ${{b32:{prefix}_warp_yz}}, ${{b32:{prefix}_warp_yz}}, 0, 31, 0xffffffff;
        or.b32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp_yz}};
        mov.u32 ${{b32:{prefix}_warp_yz}}, %tid.z;
        shfl.sync.idx.b32 ${{b32:{prefix}_warp_yz}}, ${{b32:{prefix}_warp_yz}}, 0, 31, 0xffffffff;
        or.b32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp_yz}};
        setp.ne.u32 ${{pred:{prefix}_other_warp}}, ${{b32:{prefix}_warp}}, {issuer_warp};
    """) + "".join(
        f"setp.ne.u32 ${{pred:{prefix}_not_this}}, ${{b32:{prefix}_warp}}, {warp};\n"
        f"and.pred ${{pred:{prefix}_other_warp}}, ${{pred:{prefix}_other_warp}}, ${{pred:{prefix}_not_this}};\n"
        for warp in others)


# Warp guard of an operation inside an issuer region, by _region_guard_mode.
# Each format takes the template ``prefix``.
_REGION_GUARDS = {
    "predicate": "not.pred ${{pred:{prefix}_other_warp}}, ${{pred:operand.region.val}};\n",
    "constant": "mov.pred ${{pred:{prefix}_other_warp}}, 0;\n",
}


def _region_guard_mode(kernel: KernelOp) -> str:
    """How operations inside an issuer region spell their (redundant) warp guard.

    TODO(region-guards): a measured ptxas 13.0 workaround, not a semantic
    choice; the always-false guard's spelling changes K-loop code by up to 40%.
    "predicate" (``not.pred`` of the region condition) keeps the elected issue
    code uniform but leaves BRA.U branches in paired kernels; "constant"
    (``mov.pred 0``) drops them but moves single-CTA multicast descriptor math
    to the vector datapath (+R2UR). "constant" measured better for paired MMAs
    and clusters without shared operands (.benchmarks/b200_region_guard_modes_20261112).
    """
    paired = any(op.lock.type.kind == "tcgen05" and op.lock.type.attrs.get("cta_group", 1) == 2
                 for op in find_ops_of_type(kernel, AsyncOp))
    shared = any(op.multicast for op in find_ops_of_type(kernel, TmaMoveOp))
    clustered = math.prod(d or 1 for d in kernel.cluster_dim) > 1
    return "constant" if paired or (clustered and not shared) else "predicate"


def _elect_ptx(pred: str, converge: bool = True) -> str:
    """Set ``${pred:<pred>}`` on one lane, after reconverging the warp if ``converge``."""
    return ("bar.warp.sync 0xffffffff;\n" if converge else "") + f"elect.sync _|${{pred:{pred}}}, 0xffffffff;\n"


def _leader_thread(prefix: str, before_branch: str = "", issuer_warp: int = 0,
                   guard: str | None = None, converge: bool = False) -> str:
    """PTX that lets one elected thread of warp 0 do leader work.

    Every other warp branches to ``${label:<prefix>_done}``, without diverging
    inside a warp, and ``elect.sync`` sets ``${pred:<prefix>_leader}`` on one
    thread of warp 0, so ptxas can keep the work's operands in uniform
    registers. ``before_branch`` runs in every warp, so ptxas can keep its
    special-register reads out of loops.
    """
    return (
        (guard if guard is not None else _leader_warp(prefix, issuer_warp))
        + before_branch
        + f"@${{pred:{prefix}_other_warp}} bra ${{label:{prefix}_done}};\n"
        + _elect_ptx(f"{prefix}_leader", converge)
    )


def _pair_leader_ptx(rank: str, pred: str, peer: bool = False) -> str:
    """Read the cluster rank into ``rank``; set ``pred`` on the CTA pair's even leader, or its odd ``peer``."""
    return (f"mov.u32 ${{b32:{rank}}}, %cluster_ctarank;\nand.b32 ${{b32:{rank}_parity}}, ${{b32:{rank}}}, 1;\n"
            f"setp.{'ne' if peer else 'eq'}.u32 ${{pred:{pred}}}, ${{b32:{rank}_parity}}, 0;\n")


def _try_wait_ptx(lock: str, scope: str = "cta", prefix: str = "") -> str:
    """Spin until ``operand.<lock>`` completes its current parity.

    A tick hint in a register lets ptxas select the compact fused wait.
    """
    return (f"mov.u32 ${{b32:{prefix}ticks}}, 0x989680;\n${{label:{prefix}wait}}:\n"
            f"mbarrier.try_wait.parity.acquire.{scope}.shared::cta.b64 ${{pred:{prefix}ready}}, "
            f"[${{b32:operand.{lock}.val}}], ${{b32:operand.{lock}.parity}}, ${{b32:{prefix}ticks}};\n"
            f"@!${{pred:{prefix}ready}} bra ${{label:{prefix}wait}};\n")


# A cluster-to-leader release, in every warp: one elected thread arrives at
# its own CTA's mbarrier, for the CTA leader's next write, and a non-leader
# CTA's also at the cluster leader's, for the cluster leader's next write.
_CLUSTER_RELEASE_ARRIVAL = _elect_ptx("cl_arrive") + textwrap.dedent("""\
    mov.u32 ${b32:cluster_rank}, %cluster_ctarank;
    setp.ne.u32 ${pred:cl_remote}, ${b32:cluster_rank}, 0;
    and.pred ${pred:cl_arrive_remote}, ${pred:cl_arrive}, ${pred:cl_remote};
    mov.u32 ${b32:leader_rank}, 0;
    mapa.shared::cluster.u32 ${b32:leader_mbar}, ${b32:operand.sync.val}, ${b32:leader_rank};
    @${pred:cl_arrive} mbarrier.arrive.shared::cta.b64 _, [${b32:operand.sync.val}];
    @${pred:cl_arrive_remote} mbarrier.arrive.shared::cluster.b64 _, [${b32:leader_mbar}];
    """)

# The rest of the release: the leader's warp waits on its CTA's mbarrier
# phase, while the other warps go on.
_CLUSTER_LEADER_WAIT = (
    "@${pred:cl_other_warp} bra ${label:cl_done};\n" + _try_wait_ptx("sync", "cluster")
    + "xor.b32 ${b32:operand.sync.parity}, ${b32:operand.sync.parity}, 1;\n${label:cl_done}:\n")
_CLUSTER_RELEASE_LEADER_WAIT = _leader_warp("cl") + _CLUSTER_LEADER_WAIT


def _split_release(lock: Value) -> bool:
    """Whether a cluster mbarrier releases stages with split arrive/wait."""
    return any(
        isinstance(user, BarrierOp) and "phase" in user.attrs for user in lock.users()
    )


def _sink_tmem_alloc(ops: list[Op], alloc: AllocOp) -> None:
    """Allocate a kernel-wide TMEM arena just before its first reader or the first wait.

    The allocation's rendezvous (and any retry while a previous CTA owns this
    SM's TMEM) then overlaps the operand loads instead of delaying them. Its
    views move with it; their users are the readers.
    """
    start = ops.index(alloc)
    views = [op for op in ops[start + 1:]
             if isinstance(op, TileOp) and op.operands[0] is alloc.result()]
    users = [user for user in alloc.result().users() if user not in views]
    users += [user for view in views for user in view.result().users()]
    for index in range(start + 1, len(ops)):
        op = ops[index]
        if op in views:
            continue
        if isinstance(op, AwaitOp) or any(
            user is op or find_ancestor(user, lambda parent: parent is op) is not None
            for user in users
        ):
            break
    else:
        return
    moved = [alloc, *(view for view in views if ops.index(view) < index)]
    at = index - len(moved)
    for op in moved:
        ops.remove(op)
    ops[at:at] = moved


def _published_wait(op) -> bool:
    """Whether ``op`` is an all-warp wait directly followed by a CTA barrier."""
    if not isinstance(op, AwaitOp) or op.waiter_warps is not None:
        return False
    ops = op.parent_block.ops
    index = ops.index(op) + 1
    return (index < len(ops) and isinstance(ops[index], BarrierOp)
            and ops[index].attrs.get("scope") == "block"
            and execution_warps(ops[index]) is None)


def _issuer_only(op: Op, issuer: list[int]) -> bool:
    if isinstance(op, AsyncOp) and 'participant_warps' not in op.attrs:
        # An elected TMA or tcgen05 issue runs on its issuing warp alone.
        return (cast(LockType, op.lock.type).kind in ('mbar', 'tcgen05')
                and [execution_issuer_warp(op)] == issuer)
    if isinstance(op, AsyncOp | BarrierOp):
        return op.attrs.get('participant_warps') == issuer
    if isinstance(op, AwaitOp):
        # Every thread tracks a lock's phase. Nonparticipants that skip an
        # await must never observe that lock elsewhere.
        return all(user.attrs.get('waiter_warps') == issuer
                   for user in op.lock.users() if isinstance(user, AwaitOp))
    if isinstance(op, IfOp | ForOp):
        return 'participant_warps' not in op.attrs and all(
            _issuer_only(inner, issuer) or is_side_effect_free(inner) for inner in op.body.ops)
    return False


def _group_issuer_runs(block: Block, issuer: list[int], warp_index: Value | None = None) -> None:
    """Guard each run of the issuing warp's operations once.

    Other warps then branch over a run instead of over each operation in it.
    Definitions used after the run stay outside it.
    """
    ops = list(block.ops)
    grouped: list[Op] = []
    index = 0
    while index < len(ops):
        end = index
        while end < len(ops) and (_issuer_only(ops[end], issuer) or is_side_effect_free(ops[end])):
            end += 1
        run = ops[index:end]
        while run and not _issuer_only(run[-1], issuer):
            run.pop()
        if sum(_issuer_only(op, issuer) for op in run) < 2:
            grouped.append(ops[index])
            index += 1
            continue
        later = {v for op in ops[index + len(run):] for nested in iter_ops(op) for v in nested.operands}
        inside = set(run)
        outside = set()
        changed = True
        while changed:
            changed = False
            for op in run:
                if op in inside and not _issuer_only(op, issuer) and any(r in later for r in op.results):
                    inside.discard(op)
                    outside.add(op)
                    changed = True
                    later.update(op.operands)
        if warp_index is None:
            tid = ThreadIdOp.build(CollectiveAxis.X)
            source, scale = [tid], 32
        else:
            # A warp index broadcast from each warp's first lane keeps the
            # branch, and the issue code inside it, provably warp-uniform.
            source, scale = [], 1
        bound = IndexConstantOp.build(scale * (issuer[0] + 1))
        condition = IndexLtOp.build(source[0].result() if source else warp_index, bound.result())
        region = IfOp.build(condition.result(), Block(ops=[op for op in run if op in inside]))
        region.attrs['participant_warps'] = list(issuer)
        grouped += [op for op in run if op in outside] + [*source, bound, condition, region]
        index += len(run)
    block.ops = grouped



def _group_issuer_regions(block, warp_index=None) -> None:
    """Guard each run of warp 0's own issue work once, after barriers are placed.

    TMA/tcgen05 issues and issuer-only awaits already execute on warp 0
    alone, behind separate guards that every other warp still traverses.
    One region guard lets the other warps branch over the whole run, such as a
    K loop whose waits all belong to the issuer. Barriers, which every warp
    must reach, and values used after the run stay outside it.
    """

    owner = block.parent_op
    kernel = owner if isinstance(owner, KernelOp) else (
        find_ancestor_of_type(owner, KernelOp) if owner is not None else None)
    if kernel is not None and kernel.block_dim[1:] != (1, 1):
        # Participant regions use the x warp index. A tid.x guard alone
        # cannot select one physical warp in a multidimensional CTA.
        return

    for wait in block.ops:
        # A CTA barrier right after a wait already publishes what the wait
        # observed, so the issuer alone can wait on an mbarrier phase while
        # the others park at the barrier instead of polling through the issue
        # run. (Bulk-group waits only cover the waiting thread's own copies.)
        # Every thread tracks the phase, so all its waits must be published.
        if (isinstance(wait, AwaitOp) and wait.waiter_warps is None
                and cast(LockType, wait.lock.type).kind in ("mbar", "tcgen05")
                and all(_published_wait(user) for user in wait.lock.users()
                        if not isinstance(user, AsyncOp))):
            for user in wait.lock.users():
                if isinstance(user, AwaitOp):
                    user.attrs["waiter_warps"] = [0]
    _group_issuer_runs(block, [0], warp_index)
    for op in block.ops:
        if isinstance(op, ForOp) and "participant_warps" not in op.attrs:
            _group_issuer_regions(op.body, warp_index)


# An event's mbarrier count and thread-0 initialization: ``init(dst, suffix)``
# returns the PTX for result ``dst``, suffixing its own placeholders.
EventInit = tuple[int, Callable[[str, str], str]]


def _mbarrier_init(slots: int, arrivals: int | str, setup: Callable[[str], str] = lambda s: "") -> EventInit:
    """Initialize ``slots`` mbarriers with ``arrivals`` each, a count or a register ``setup`` computes."""
    def init(dst: str, s: str) -> str:
        count = f"${{b32:{arrivals}{s}}}" if isinstance(arrivals, str) else arrivals
        if slots == 1:
            return setup(s) + f"mbarrier.init.shared::cta.b64 [${{b32:{dst}.val}}], {count};\n"
        return textwrap.dedent(f"""\
            mov.u32 ${{b32:init_index{s}}}, 0;
            ${{label:init_loop{s}}}:
            .pragma "nounroll";
            mad.lo.u32 ${{b32:init_addr{s}}}, ${{b32:init_index{s}}}, 8, ${{b32:{dst}.val}};
            """) + setup(s) + textwrap.dedent(f"""\
            mbarrier.init.shared::cta.b64 [${{b32:init_addr{s}}}], {count};
            add.u32 ${{b32:init_index{s}}}, ${{b32:init_index{s}}}, 1;
            setp.lt.u32 ${{pred:init_more{s}}}, ${{b32:init_index{s}}}, {slots};
            @${{pred:init_more{s}}} bra ${{label:init_loop{s}}};

            """)
    return slots, init


def _event_inits_ptx(issuer_warp: int, inits: list[EventInit]) -> str:
    """Declare events, then initialize them all behind one thread-0 branch and fence.

    ptxas fences each such init block against the async proxy once events
    are addressed through the shared arena (~50 cycles each), so consecutive
    events share one block.
    """
    names = ([("result.dst", "")] if len(inits) == 1
             else [(f"result.dst{n}", f"_{n}") for n in range(len(inits))])
    return ("".join(f".shared .align 8 .b64 ${{smem:bar{s}}}[{slots}];\n"
                    f"mov.u32 ${{b32:{dst}.val}}, ${{smem:bar{s}}};\nmov.u32 ${{b32:{dst}.parity}}, 0;\n"
                    for (slots, _), (dst, s) in zip(inits, names))
            + textwrap.dedent(f"""\
                mov.u32 ${{b32:tid_x}}, %tid.x;
                mov.u32 ${{b32:tid_y}}, %tid.y;
                or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_y}};
                mov.u32 ${{b32:tid_z}}, %tid.z;
                or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_z}};
                setp.ne.s32 ${{pred:not_t0}}, ${{b32:tid_x}}, {32 * issuer_warp};
                @${{pred:not_t0}} bra ${{label:after_init}};
                """)
            + "".join(init(dst, s) for (_, init), (dst, s) in zip(inits, names))
            + "fence.mbarrier_init.release.cluster;\n${label:after_init}:\n")


# Probing ahead only pays when operands are usually resident already: rings
# of at least six banks gained 0.6% geomean on the B200 MM corpus, shallower
# rings (whose waits genuinely block) were neutral to slightly slower.
_PROBE_AHEAD_MIN_BANKS = 6


def _operand_bank_depth(kernel: KernelOp) -> int:
    """Largest number of distinct SMEM banks one global operand is loaded into."""
    banks: dict[object, set[object]] = {}
    for op in iter_ops(kernel):
        if isinstance(op, TmaMoveOp) and isinstance(op.dst.type, SmemTensorType):
            banks.setdefault(root_buffer(op.src), set()).add(root_buffer(op.dst))
    return max((len(roots) for roots in banks.values()), default=0)


class LowerToPtxTemplate(LoweringPass):
    """Pass that rewrites high-level ops to ptx_template ops."""

    def __init__(self) -> None:
        super().__init__()
        self.cache_policy_values: dict[str, Value] = {}
        self.warp_ids: WarpIdsOp | None = None
        self.distinct_issuers = False
        self._early_tmem_alloc: AllocOp | None = None
        # Probe-ahead completion observations: awaits probed before the
        # preceding MMA issue, keyed by that issue, and the probe results.
        self._probe_sites: dict[int, list[AwaitOp]] = {}
        self._lowering: list[Op] = []
        self._probe_results: dict[int, Value] = {}
        self._region_guard_modes: dict[int, str] = {}
        self._bank_depths: dict[int, int] = {}
        # Lowered event allocation -> (issuing warp, initialization).
        self._event_inits: dict[int, tuple[int, EventInit]] = {}
        self._issuer_elections: dict[IfOp, Value] = {}
        self._operand_loops = set()

    def run(self, module: ModuleOp) -> ModuleOp:
        if module.body.ops[0].attrs.get("rolled_shared_ring"):
            from .pipeline_rolled_shared_ring import PrepareRolledSharedRing

            module = PrepareRolledSharedRing().run(module)
            from .verify_gpu_dialect import verify_expanded_gpu_hardware

            verify_expanded_gpu_hardware(module)
        return super().run(module)

    def lower_block(self, block) -> None:
        self._plan_probe_ahead(block)
        owner = block.parent_op
        if (isinstance(owner, IfOp) and execution_warps(owner) is not None
                and len(execution_warps(owner)) == 1
                and any(op in self._operand_loops
                        for op in iter_ops(owner))):
            # The region's branch is warp-uniform. Elect the common issuing
            # lane once; every K batch retains the same lane and predicates.
            election = PTXTemplateOp.build(operands=[], result_types=[PredicateType()],
                template=_elect_ptx('result.lane.val', converge=False),
                result_names=['lane'], description='rolled issuing lane')
            block.ops.insert(0, election)
            self._issuer_elections[owner] = election.result()
        super().lower_block(block)

    def _plan_probe_ahead(self, block) -> None:
        """Observe event completion one MMA issue ahead of each wait.

        A single issuing stream otherwise pays the full phase-check latency
        between consecutive MMA batches. A non-blocking test of the same
        event and parity is issued before the preceding MMA batch; the wait
        skips its blocking loop when that test already succeeded.

        Correctness: the probe uses the wait's own parity. No other wait on
        the same event lies between the probe and this wait, so the event's
        previous phase was already observed before the probe executes, and
        a probe cannot alias an older phase. A failed probe keeps the
        original blocking wait.
        """
        for loop in list(block.ops):
            if isinstance(loop, ForOp):
                self._plan_rolled_probes(loop)
        owner = block.parent_op
        kernel = owner if isinstance(owner, KernelOp) else (
            find_ancestor_of_type(owner, KernelOp) if owner is not None else None)
        if kernel is None:
            return
        # Measure the operand ring once, before lowering replaces its loads.
        depth = self._bank_depths.setdefault(id(kernel), _operand_bank_depth(kernel))
        if depth < _PROBE_AHEAD_MIN_BANKS:
            return
        ops = block.ops
        last_issue = None
        for index, op in enumerate(ops):
            if isinstance(op, AsyncOp) and any(isinstance(inner, MatmulOp) for inner in op.body.ops):
                last_issue = index
                continue
            if (last_issue is None or not isinstance(op, AwaitOp)
                    or not isinstance(op.lock.type, LockType)
                    or op.lock.type.kind not in ("mbar", "tcgen05")
                    or isinstance(op.lock.definition, LockViewOp)):
                continue
            between = ops[last_issue + 1:index]
            if op.lock.definition in ops[last_issue:index] or any(
                    isinstance(inner, AwaitOp) and inner.lock is op.lock
                    for between_op in between for inner in iter_ops(between_op)):
                continue
            self._probe_sites.setdefault(id(ops[last_issue]), []).append(op)

    def _plan_rolled_probes(self, loop):
        """Carry explicit-epoch observations across a rolled backedge.

        Each predicate starts false, so the first iteration performs its
        ordinary wait. A probe writes the predicate for the next iteration;
        only the final PTX has mutable registers, like implicit lock parity.
        The semantic ring proves the previous generation was observed before
        reuse. Repeated jobs need a separate proof of their carried epochs.
        """
        from ..analysis.operand_ring import operand_ring
        facts = operand_ring(loop)
        if facts is None:
            return
        self._operand_loops.add(loop)
        if facts.depth < _PROBE_AHEAD_MIN_BANKS:
            return
        issues = [op for op in loop.body.ops if isinstance(op, AsyncOp)
                  and op.lock.type.kind == 'tcgen05']
        if not issues:
            return
        waits = facts.waits
        before = []
        for wait in waits:
            if wait.lock.type.kind not in ('mbar', 'tcgen05'):
                continue
            seed = PTXTemplateOp.build([], [PredicateType()],
                'mov.pred ${pred:result.ready.val}, 0;\n', result_names=['ready'],
                description='initial rolled probe')
            loop.parent_block.ops.insert(loop.parent_block.ops.index(loop), seed)
            current = PTXTemplateOp.build([seed.result()], [PredicateType()],
                'mov.pred ${pred:result.ready.val}, ${pred:operand.previous.val};\n',
                operand_names=['previous'], result_names=['ready'],
                description='consume previous rolled probe')
            loop.body.ops.insert(0, current)
            self._probe_results[id(wait)] = current.result()
            one = IndexConstantOp.build(loop.step)
            ahead = IndexAddOp.build(loop.iv, one.result())
            before.extend([one, ahead])
            values = {loop.iv: ahead.result()}

            def copy(value):
                if value in values:
                    return values[value]
                definition = value.definition
                if (not isinstance(definition, Op)
                        or find_ancestor_of_type(definition, ForOp) is not loop):
                    return value
                if not isinstance(definition, (IndexAddOp, IndexMulOp, IndexDivOp,
                                               IndexModOp, IndexConstantOp, LockViewOp)):
                    raise ValueError('rolled probe requires an explicit affine epoch')
                for operand in definition.operands:
                    copy(operand)
                cloned = clone_op(definition, values)
                before.append(cloned)
                return cloned.result()

            next_lock = copy(wait.lock)
            before.append(PTXTemplateOp.build([next_lock, seed.result()], [],
                'mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 '
                '${pred:operand.probe.val}, [${b32:operand.lock.val}], '
                '${b32:operand.lock.parity};\n', operand_names=['lock', 'probe'],
                description='probe across rolled backedge'))
        at = loop.body.ops.index(issues[0])
        loop.body.ops[at:at] = before

    def _probe_ahead_ops(self, op: AsyncOp) -> list[Op]:
        probes = []
        for wait in self._probe_sites.pop(id(op), []):
            probe = PTXTemplateOp.build(
                operands=[cast(Value, self.value_map[wait.lock])],
                result_types=[PredicateType()],
                template=("mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 "
                          "${pred:result.ready.val}, [${b32:operand.lock.val}], ${b32:operand.lock.parity};\n"),
                operand_names=["lock"],
                result_names=["ready"],
                description=f"probe-ahead {wait}",
            )
            self._probe_results[id(wait)] = probe.result()
            probes.append(probe)
        return probes

    def _issuer_region(self, warp: int) -> IfOp | None:
        """The region only ``warp`` executes around the op being lowered.

        The op's own warp guard is then already decided by the region's branch.
        """
        if not self._lowering:
            return None
        return cast(IfOp | None, find_ancestor(
            self._lowering[-1], lambda op: isinstance(op, IfOp) and execution_warps(op) == frozenset({warp})))

    def _region_guard(self, prefix: str, warp: int) -> tuple[list[Value], list[str], str] | None:
        """Operands and PTX of an op's warp guard inside ``warp``'s issuer region."""
        region = self._issuer_region(warp)
        if region is None:
            return None
        kernel = find_ancestor_of_type(region, KernelOp)
        if id(kernel) not in self._region_guard_modes:
            self._region_guard_modes[id(kernel)] = _region_guard_mode(kernel)
        if self._region_guard_modes[id(kernel)] == "constant":
            return [], [], _REGION_GUARDS["constant"].format(prefix=prefix)
        return ([cast(Value, self.value_map.get(region.condition, region.condition))], ["region"],
                _REGION_GUARDS["predicate"].format(prefix=prefix))

    def lower_op(self, op: Op) -> tuple[list[Op], list[object]]:
        self._lowering.append(op)
        try:
            result = super().lower_op(op)
        finally:
            self._lowering.pop()
        actors = execution_warps(op)
        # A larger enclosing region still contains threads outside this
        # operation's participant set; retain the child's narrower guard.
        if actors is None or isinstance(op, (AsyncOp, AwaitOp, IfOp)) or find_ancestor(
                op, lambda a: isinstance(a, IfOp) and actors == execution_warps(a)):
            # Engine issuers and awaits already have dedicated guards. Await
            # parity must advance for nonparticipants too.
            return result
        if not isinstance(op, (MoveOp, ElementwiseOp, BarrierOp)):
            raise ValueError("participant scheduling supports copies, conversion and barriers")
        first, *rest = warps = sorted(actors)
        prefix = "mov.u32 ${b32:participant_tid}, %tid.x;\n" + "".join(
            "shr.u32 ${b32:participant_warp}, ${b32:participant_tid}, 5;\n"
            f"setp.eq.u32 ${{pred:participant_{warp}}}, ${{b32:participant_warp}}, {warp};\n" for warp in warps)
        prefix += f"mov.pred ${{pred:participant_active}}, ${{pred:participant_{first}}};\n" + "".join(
            f"or.pred ${{pred:participant_active}}, ${{pred:participant_active}}, ${{pred:participant_{warp}}};\n"
            for warp in rest) + "@!${pred:participant_active} bra ${label:participant_done};\n"
        for generated in result[0]:
            if isinstance(generated, PTXTemplateOp):
                generated.attrs['template'] = prefix + generated.template + '${label:participant_done}:\n'
        return result

    def _warp0_template(self, prefix: str, instructions: list[str], description: str, *,
                        operands=(), names=(), result_types=(), declaration: str = "",
                        epilogue: str = "") -> PTXTemplateOp:
        """Warp-collective TMEM ``instructions``, executed by warp 0 only."""
        guard_operands, guard_names, guard = self._leader_warp(prefix)
        return PTXTemplateOp.build(
            operands=[*operands, *guard_operands], result_types=list(result_types),
            template=guard + declaration
            + "".join(f"@!${{pred:{prefix}_other_warp}} {line}\n" for line in instructions) + epilogue,
            operand_names=[*names, *guard_names], result_names=["dst"] if result_types else [],
            description=description)

    def _lower_alloc_tmem(self, op: AllocOp) -> tuple[list[Op], list[object]]:
        """Allocate tensor memory from warp 0 and share its address."""
        typ = op.result().type
        assert isinstance(typ, TmemTensorType)
        group = typ.cta_group
        lowered = self._warp0_template(
            "tmem",
            [f"tcgen05.alloc.cta_group::{group}.sync.aligned.shared::cta.b32 [${{smem:tmem_addr}}], {typ.columns};"]
            + ([f"tcgen05.relinquish_alloc_permit.cta_group::{group}.sync.aligned;"]
               if op is self._early_tmem_alloc else []),
            str(op), result_types=[typ],
            declaration=".shared .align 4 .b32 ${smem:tmem_addr}[1];\n",
            epilogue="bar.sync 0;\nld.shared.b32 ${b32:result.dst.ptr}, [${smem:tmem_addr}];\n")
        return [lowered], [lowered.result()]

    @lowers(DeallocOp)
    def lower_dealloc(self, op: DeallocOp) -> tuple[list[Op], list[object]]:
        typ = op.value.type
        if isinstance(typ, LockType):
            operands, names, leader = self._leader_thread('event_free', issuer_warp=execution_issuer_warp(op))
            template = leader + (
                '@${pred:event_free_leader} mbarrier.inval.shared::cta.b64 [${b32:operand.lock.val}];\n'
                '${label:event_free_done}:\n'
            )
            lowered = PTXTemplateOp.build(
                operands=[self.value_map[op.value], *operands], result_types=[],
                template=template, operand_names=['lock', *names], result_names=[],
                description=str(op),
            )
            return [lowered], []
        assert isinstance(typ, TmemTensorType)
        return [self._warp0_template(
            "tmem_free",
            [f"tcgen05.dealloc.cta_group::{typ.cta_group}.sync.aligned.b32 ${{b32:operand.src.ptr}}, {typ.columns};"],
            str(op), operands=[cast(Value, self.value_map[op.value])], names=["src"])], []

    def _lower_fill_tmem(self, op: ElementwiseOp) -> tuple[list[Op], list[object]]:
        """Broadcast a scalar into tensor memory; each thread stores one row."""
        typ = op.dst.type
        assert isinstance(typ, TmemTensorType)
        if op.fn != "fill" or op.src(0).type != RmemTensorType.build([], F32Type()):
            raise NotImplementedError(
                "TMEM elementwise lowering supports scalar f32 fill"
            )
        columns = cast(int, typ.sizes[1])
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        ownership = TmemWarpLayout(cast(int, typ.sizes[0]), kernel.num_threads // 32, typ.cta_group)
        rows_per_warp = ownership.rows_per_warp
        layout = "tcgen05_32x32b" if rows_per_warp == 32 else "tcgen05_16x256b"
        assert self.warp_ids is not None, "lower_kernel creates warp_ids"
        return self._physical_fragment_coordinates(op, _lowered(
            op, _tmem_fragment_ptx("operand.dst.ptr", ownership, layout, columns,
                                   ["${f32:operand.src.r0}"] * (rows_per_warp * ownership.columns_per_warp(columns) // 32),
                                   is_load=False),
            [cast(Value, self.value_map[op.dst]), cast(Value, self.value_map[op.src(0)]), self.warp_ids.x],
            ["dst", "src", "warp"]))

    def _lower_move_tmem_fragment(self, op: MoveOp) -> tuple[list[Op], list[object]]:
        """Load a register fragment from tensor memory, or store one to it.

        The fragment's layout selects the ``tcgen05.ld``/``tcgen05.st`` shape;
        each warp addresses its own 32-lane quarter.
        """
        is_load = isinstance(op.dst.type, WgrmemTensorType)
        fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
        fragment_type, memory_type = fragment.type, memory.type
        if not (
            isinstance(fragment_type, WgrmemTensorType)
            and isinstance(memory_type, TmemTensorType)
            and fragment_type.layout in TMEM_FRAGMENT_LAYOUTS
            and fragment_type.warps in (4, 8)
            and fragment_type.sizes == memory_type.sizes
            and fragment_type.dtype == memory_type.dtype
            and not isinstance(fragment.definition, TileOp)
        ):
            raise NotImplementedError(
                "TMEM moves need a whole tcgen05 fragment of the same shape "
                f"and dtype; got {op.dst.type} <- {op.src.type}"
            )
        assert self.warp_ids is not None, "lower_kernel creates warp_ids"
        ownership = TmemWarpLayout(cast(int, memory_type.sizes[0]), fragment_type.warps, memory_type.cta_group)
        registers = [
            f"${{f32:operand.fragment.r{i}}}"
            for i in range(fragment_type.num_elements_per_thread)
        ]
        return _lowered(
            op, _tmem_fragment_ptx("operand.memory.ptr", ownership, fragment_type.layout,
                                   cast(int, fragment_type.sizes[1]), registers, is_load=is_load),
            [cast(Value, self.value_map[memory]), cast(Value, self.value_map[fragment]), self.warp_ids.x],
            ["memory", "fragment", "warp"])

    def _lower_move_row_fragment(self, op: MoveOp) -> tuple[list[Op], list[object]]:
        """Store row fragments using their declared TMEM copy geometry.

        Each copy atom holds a contiguous row segment in consecutive
        registers. Adjacent lanes may share a row. The same vector stores
        handle global memory and blocked, swizzled shared memory.
        """
        is_load = isinstance(op.dst.type, WgrmemTensorType)
        fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
        fragment_type, memory_type = fragment.type, memory.type
        assert isinstance(fragment_type, WgrmemTensorType)
        kernel = find_ancestor_of_type(op, KernelOp)
        if not (
            isinstance(memory_type, (GmemTensorType, SmemTensorType))
            and memory_type.rank == 2
            and memory_type.sizes == fragment_type.sizes
            and memory_type.dtype == fragment_type.dtype
            and memory_type.strides[1] == 1
            and isinstance(memory_type.dtype, (BF16Type, F16Type, F32Type))
            and not isinstance(fragment.definition, TileOp)
        ):
            raise NotImplementedError(
                "row fragments store whole to row-major memory of the same shape "
                f"and dtype; got {memory_type} <- {fragment_type}"
            )
        dtype_bytes = memory_type.dtype.byte_size
        f32 = isinstance(memory_type.dtype, F32Type)
        packing = 1 if f32 else 2
        count = fragment_type.num_elements_per_thread // packing
        geometry = TMEM_FRAGMENT_LAYOUTS[fragment_type.layout]
        ownership = TmemWarpLayout(cast(int, fragment_type.sizes[0]), fragment_type.warps,
                                   fragment_type.attrs.get("tmem_cta_group", 1))
        rows_per_warp = ownership.rows_per_warp
        # Validate the logical row coverage before emitting any accesses.
        accesses = list(geometry.accesses(ownership.columns_per_warp(cast(int, fragment_type.sizes[1])), rows_per_warp))
        if any(first % packing or size % packing for _, _, first, size, _ in accesses):
            raise NotImplementedError("row fragment Half copies require complete register pairs")
        registers = [
            f"${{{'f32' if f32 else 'b32'}:operand.fragment.r{i}}}" for i in range(count)
        ]
        template = textwrap.dedent("""\
            mov.u32 ${b32:row}, %tid.x;
            and.b32 ${b32:row}, ${b32:row}, 127;
        """)
        if geometry.lanes != 32:
            template += textwrap.dedent(f"""\
                shr.u32 ${{b32:warp}}, ${{b32:row}}, 5;
                shr.u32 ${{b32:column}}, ${{b32:row}}, {geometry.lanes.bit_length() - 1};
                and.b32 ${{b32:column}}, ${{b32:column}}, {32 // geometry.lanes - 1};
                mul.lo.u32 ${{b32:column}}, ${{b32:column}}, {geometry.registers};
                and.b32 ${{b32:row}}, ${{b32:row}}, {geometry.lanes - 1};
                mad.lo.u32 ${{b32:row}}, ${{b32:warp}}, {ownership.row_stride}, ${{b32:row}};
                mul.lo.u32 ${{b32:column_bytes}}, ${{b32:column}}, {dtype_bytes};
            """)
        else:
            template += "mov.u32 ${b32:column_bytes}, 0;\n"
        if ownership.paired_m64:
            # Partitions 2/3 repeat partitions 0/1's logical rows in the
            # second column half. Eight warps then add their 16-row half.
            template += "and.b32 ${b32:row}, ${b32:row}, 63;\n"
        if ownership.group_row_stride:
            template += textwrap.dedent(f"""\
                mov.u32 ${{b32:row_group}}, %tid.x;
                shr.u32 ${{b32:row_group}}, ${{b32:row_group}}, 7;
                mad.lo.u32 ${{b32:row}}, ${{b32:row_group}}, {ownership.group_row_stride}, ${{b32:row}};
            """)
        group_columns = ownership.group_column_stride(cast(int, fragment_type.sizes[1]))
        if group_columns:
            template += "mov.u32 ${b32:column_group}, %tid.x;\n"
            template += f"shr.u32 ${{b32:column_group}}, ${{b32:column_group}}, {6 if ownership.paired_m64 else 7};\n"
            if ownership.paired_m64:
                template += "and.b32 ${b32:column_group}, ${b32:column_group}, 1;\n"
        global_memory = isinstance(memory_type, GmemTensorType)
        if group_columns and global_memory:
            template += f"mad.lo.u32 ${{b32:column_bytes}}, ${{b32:column_group}}, {group_columns * dtype_bytes}, ${{b32:column_bytes}};\n"
        if is_load and not (global_memory and memory_type.strides[0] == 0):
            raise NotImplementedError("row fragment loads currently require a broadcast row bias")
        masked = not is_load and global_memory and kernel is not None and kernel.attrs.get("masked_global_store")
        index_ops, origins = [], []
        if masked:
            if f32:
                raise NotImplementedError("masked row fragments require Half outputs")
            from .lower_to_tma_move import _build_tile_chain_indices

            index_ops, root, origins, _ = _build_tile_chain_indices(
                memory, list(memory_type.sizes)
            )
        layout = None if global_memory else physical_layout(memory_type)
        if layout is None:
            alignment = cast(int, memory_type.strides[0]) * dtype_bytes
            if global_memory:
                template += textwrap.dedent(f"""\
                    mul.wide.u32 ${{b64:row_offset}}, ${{b32:row}}, {alignment};
                    add.u64 ${{b64:base}}, ${{b64:operand.memory.ptr}}, ${{b64:row_offset}};
                    cvt.u64.u32 ${{b64:column_offset}}, ${{b32:column_bytes}};
                    add.u64 ${{b64:base}}, ${{b64:base}}, ${{b64:column_offset}};
                """)
            else:
                template += (
                    f"mad.lo.u32 ${{b32:base}}, ${{b32:row}}, {alignment}, "
                    "${b32:operand.memory.ptr};\n"
                    "add.u32 ${b32:base}, ${b32:base}, ${b32:column_bytes};\n"
                )
            base = f"${{{'b64' if global_memory else 'b32'}:base}}"

            def address(row: int, column: int) -> tuple[str, str]:
                return "", f"[{base}+{row * alignment + column * dtype_bytes}]"

        else:
            assert isinstance(memory_type, SmemTensorType)
            block_width, order = layout
            root = _layout_view_root(memory)
            assert root is not None and isinstance(root.type, SmemTensorType)
            if order != "block_major" or memory_type.swizzle not in (
                None,
                _block_swizzle(block_width),
            ):
                raise NotImplementedError(
                    f"row fragments store to block-major smem; got {memory_type}"
                )
            alignment = block_width * dtype_bytes
            if block_width < geometry.columns:
                raise NotImplementedError("row fragment copy atoms must fit a shared-memory block")
            root_rows = cast(int, root.type.sizes[0])
            template += f"mul.lo.u32 ${{b32:row_offset}}, ${{b32:row}}, {alignment};\n"
            template += "add.u32 ${b32:row_offset}, ${b32:row_offset}, ${b32:column_bytes};\n"

            def address(row: int, column: int) -> tuple[str, str]:
                static = _block_major_offset(row, column, block_width, root_rows) * dtype_bytes
                group_setup = ""
                if group_columns:
                    group_static = _block_major_offset(row, column + group_columns, block_width, root_rows) * dtype_bytes
                    group_setup = f"mad.lo.u32 ${{b32:offset}}, ${{b32:column_group}}, {group_static - static}, ${{b32:offset}};\n"
                swizzle = (
                    _swizzle_ptx("address", "offset_swizzle", int(memory_type.swizzle[:-1]))
                    if memory_type.swizzle is not None
                    else ""
                )
                return (
                    f"add.u32 ${{b32:offset}}, ${{b32:row_offset}}, {static};\n"
                    + group_setup
                    + "add.u32 ${b32:address}, ${b32:operand.memory.ptr}, "
                    "${b32:offset};\n" + swizzle
                ), "[${b32:address}]"

        space = "global" if global_memory else "shared"
        if masked:
            template += textwrap.dedent(f"""\
                add.u32 ${{b32:tile_row_end}}, ${{b32:operand.origin_row.val}}, {memory_type.sizes[0]};
                add.u32 ${{b32:tile_col_end}}, ${{b32:operand.origin_col.val}}, {memory_type.sizes[1]};
                setp.le.u32 ${{pred:complete_rows}}, ${{b32:tile_row_end}}, {root.type.sizes[0]};
                setp.le.u32 ${{pred:complete_cols}}, ${{b32:tile_col_end}}, {root.type.sizes[1]};
                and.b64 ${{b64:alignment}}, ${{b64:base}}, 3;
                setp.eq.u64 ${{pred:aligned}}, ${{b64:alignment}}, 0;
                and.pred ${{pred:complete}}, ${{pred:complete_rows}}, ${{pred:complete_cols}};
                and.pred ${{pred:complete}}, ${{pred:complete}}, ${{pred:aligned}};
                @!${{pred:complete}} bra ${{label:fragment_edge}};
            """)
        # A row pitch limits vector widths, but does not establish the
        # alignment of a global pointer or an interior shared-memory view.
        # Check the actual per-thread base when allocation/view provenance
        # cannot prove it. Copy atoms and swizzles preserve these low bits.
        vector_alignment = math.gcd(16, max(4, alignment) if alignment else 16)
        for row, column, _, _, _ in accesses:
            offset = (row * alignment + column * dtype_bytes if layout is None else
                      _block_major_offset(row, column, block_width, root_rows) * dtype_bytes)
            vector_alignment = math.gcd(vector_alignment, offset)

        def copy_body(bytes_aligned: int) -> str:
            body = ""
            for row, column, first, size, _ in accesses:
                if bytes_aligned >= 4:
                    for start, width in _vector_groups(size // packing, bytes_aligned):
                        setup, location = address(row, column + start * packing)
                        begin = first // packing + start
                        vector, regs = _vector_operand(registers[begin : begin + width])
                        body += setup + (f"ld.{space}{vector}.b32 {regs}, {location};\n" if is_load
                                         else f"st.{space}{vector}.b32 {location}, {regs};\n")
                else:
                    assert not f32
                    for start in range(size // packing):
                        reg = registers[first // packing + start]
                        lo, hi = "${b16:copy_lo}", "${b16:copy_hi}"
                        if not is_load:
                            body += f"mov.b32 {{{lo}, {hi}}}, {reg};\n"
                        for half, half_reg in enumerate((lo, hi)):
                            setup, location = address(row, column + start * packing + half)
                            body += setup + (f"ld.{space}.b16 {half_reg}, {location};\n" if is_load
                                             else f"st.{space}.b16 {location}, {half_reg};\n")
                        if is_load:
                            body += f"mov.b32 {reg}, {{{lo}, {hi}}};\n"
            return body

        guaranteed = dtype_bytes if global_memory else _row_fragment_shared_alignment(memory)
        if not global_memory and guaranteed and guaranteed < dtype_bytes:
            raise NotImplementedError("row fragment copies require element-aligned shared views")
        if not global_memory and guaranteed:
            guaranteed = math.gcd(guaranteed, alignment,
                geometry.registers * dtype_bytes if geometry.lanes != 32 else 0,
                group_columns * dtype_bytes)
        minimum = max(dtype_bytes, guaranteed)
        if masked:
            minimum = max(4, minimum)  # The complete-tile path already checked this.
        maximum = max(4 * width for _, _, _, size, _ in accesses
                      for _, width in _vector_groups(size // packing, max(4, vector_alignment)))
        candidates = [width for width in (16, 8, 4) if minimum < width <= min(maximum, vector_alignment)]
        if not global_memory and guaranteed:
            candidates = []  # A known interior shared view uses its proven width.
        for width in candidates:
            setup, location = address(0, 0)
            pointer = location[1:-1].removesuffix("+0")
            bits = 64 if global_memory else 32
            template += setup + (
                f"and.b{bits} ${{b{bits}:copy_alignment}}, {pointer}, {width - 1};\n"
                f"setp.eq.u{bits} ${{pred:copy_aligned}}, ${{b{bits}:copy_alignment}}, 0;\n"
                f"@!${{pred:copy_aligned}} bra ${{label:copy_below_{width}}};\n"
                + copy_body(width)
                + f"bra ${{label:copy_done}};\n${{label:copy_below_{width}}}:\n"
            )
        template += copy_body(min(vector_alignment, minimum))
        if candidates:
            template += "${label:copy_done}:\n"
        if masked:
            template += textwrap.dedent("""\
                bra ${label:fragment_end};
                ${label:fragment_edge}:
                add.u32 ${b32:row}, ${b32:row}, ${b32:operand.origin_row.val};
                shr.u32 ${b32:col}, ${b32:column_bytes}, 1;
                add.u32 ${b32:col}, ${b32:col}, ${b32:operand.origin_col.val};
            """)
            for row, column, first, size, _ in accesses:
                for start in range(size // packing):
                    template += _ptx_masked_half_pair(
                        first // packing + start, row, column + start * packing,
                        alignment, root.type.sizes, is_load=False, writer=False,
                    )
            template += "${label:fragment_end}:\n"
        lowered, _ = _lowered(op, template, [cast(Value, self.value_map[memory]), cast(Value, self.value_map[fragment]), *origins],
                              ["memory", "fragment", *(["origin_row", "origin_col"] if masked else [])])
        return [*index_ops, *lowered], []

    @lowers(ConstantOp)
    def lower_constant(
        self,
        op: ConstantOp,
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
        self,
        op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        self.distinct_issuers = any(inner.attrs.get('issuer_warp', 0) for inner in iter_ops(op))
        # TODO: Materialize cache policies in an earlier pass, potentially
        # alongside buffer promotion, so this pass only renders their PTX.
        cache_hints = sorted(
            {
                hint
                for tma in find_ops_of_type(op, TmaMoveOp)
                if (hint := tma.l2_cache_hint) is not None
            }
        )
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
                )
                + "\n",
                result_names=names,
                description="create TMA L2 cache policies",
            )
            op.body.ops.insert(0, policy_op)
            self.cache_policy_values = dict(zip(cache_hints, policy_op.results))
        # The leader's warp (warp 0) waits at gather barriers and issues TMA.
        # Compute the warp index once, rather than at every use.
        self.warp_ids = WarpIdsOp.build()
        op.body.ops.insert(0, self.warp_ids)
        tmem_allocs = [alloc for alloc in find_ops_of_type(op, AllocOp)
            if isinstance(alloc.result().type, TmemTensorType)]
        # A single unconditional arena is the last allocation for this CTA.
        # Release its allocation permit immediately so another CTA can make
        # progress while this one still owns its TMEM and drains output.
        # Multiple sites, conditional allocation and repeated allocation keep
        # the permit until exit; counting static sites alone is insufficient.
        self._early_tmem_alloc = (
            tmem_allocs[0] if len(tmem_allocs) == 1
            and tmem_allocs[0].parent_block is op.body
            and (not op.is_clustered or tmem_allocs[0].result().type.cta_group == 2)
            else None
        )
        if len(tmem_allocs) == 1 and tmem_allocs[0].parent_block is op.body:
            _sink_tmem_alloc(op.body.ops, tmem_allocs[0])
        module = find_ancestor_of_type(op, ModuleOp)
        if module is not None and module.target == "sm_100a":
            # Participant regions are SM100 execution scopes (see the
            # verifier). A one-dimensional CTA's warp index is its x component.
            _group_issuer_regions(op.body, self.warp_ids.x if op.block_dim[1:] == (1, 1) else None)
        self.lower_block(op.body)
        self._group_event_inits(op.body)
        if tmem_allocs and self._early_tmem_alloc is None:
            # Keep the permit through every allocation, including sequential
            # GEMMs or loop iterations. Relinquish once at kernel exit.
            op.body.ops.append(self._warp0_template(
                "tmem_exit",
                [f"tcgen05.relinquish_alloc_permit.cta_group::{tmem_allocs[0].result().type.cta_group}.sync.aligned;"],
                "relinquish TMEM allocation permit"))
        return [op], []

    def _leader_warp(self, prefix: str, issuer_warp: int = 0) -> tuple[list[Value], list[str], str]:
        """Operands and PTX setting ``${pred:<prefix>_other_warp}`` outside warp 0.

        Warp 0 holds the leader, thread 0. The kernel's warp index is broadcast
        from each warp's first thread, so ptxas proves this predicate
        warp-uniform.
        """
        assert self.warp_ids is not None, "lower_kernel creates warp_ids"
        guard = self._region_guard(prefix, issuer_warp)
        if guard is not None:
            return guard
        return (
            list(self.warp_ids.results),
            ["warp_x", "warp_y", "warp_z"],
            (
                f"or.b32 ${{b32:{prefix}_warp}}, ${{b32:operand.warp_x.val}}, "
                "${b32:operand.warp_y.val};\n"
                f"or.b32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, "
                "${b32:operand.warp_z.val};\n"
                f"setp.ne.u32 ${{pred:{prefix}_other_warp}}, ${{b32:{prefix}_warp}}, {issuer_warp};\n"
            ),
        )

    def _leader_thread(self, prefix: str, before_branch: str = "", issuer_warp: int = 0,
                       converge: bool = False, cached: bool = True) -> tuple[list[Value], list[str], str]:
        """Operands and PTX of :func:`_leader_thread` behind :meth:`_leader_warp`."""
        operands, names, guard = self._leader_warp(prefix, issuer_warp)
        region = self._issuer_region(issuer_warp)
        if cached and region in self._issuer_elections:
            operands.append(self._issuer_elections[region])
            names.append('issuing_lane')
            return operands, names, (guard + before_branch
                + f"mov.pred ${{pred:{prefix}_leader}}, ${{pred:operand.issuing_lane.val}};\n")
        return operands, names, _leader_thread(prefix, before_branch, guard=guard, converge=converge)

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
        self,
        op: TileOp,
    ) -> tuple[list[Op], list[object]] | None:
        """Lower an address tile to a pointer, or erase a WGRMEM view.

        A WGRMEM tile is a zero-cost view of its parent's distributed register
        fragment. Consumers inspect the original TileOp chain to select the
        corresponding root registers; mapping the result to the root value
        prevents the view from allocating another register tensor.
        """
        arg = cast(Value, self.value_map[op.arg])
        new_ops: list[Op] = []

        if isinstance(op.arg.type, TmemTensorType):
            # A column view offsets the address's lower 16 bits. Fragments
            # assign lanes to warps, so a view keeps all 128 lanes.
            row, column = op.static_indices
            columns = cast(int, op.result().type.sizes[1])
            if row != 0 or column is None:
                raise NotImplementedError("TMEM views need static column offsets")
            if column == 0:
                return [], [arg]
            return _lowered(op, f"add.u32 ${{b32:result.dst.ptr}}, ${{b32:operand.src.ptr}}, {column * columns};\n",
                            [arg], ["src"], [op.result().type])

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
            raise NotImplementedError(f"lower_tile: unexpected arg type {type(arg)}")

        view_root = op.arg
        while isinstance(view_root.definition, TileOp):
            view_root = view_root.definition.arg
        view_root_type = view_root.type
        assert isinstance(view_root_type, AddrMemTensorType)
        dtype_bytes = cast(
            BF16Type | F16Type | F32Type | F64Type,
            view_root_type.dtype,
        ).byte_size
        result_type = op.result().type
        assert isinstance(result_type, AddrMemTensorType)
        result_sizes = cast(tuple[int, ...], result_type.sizes)
        layout = (
            physical_layout(view_root_type)
            if isinstance(view_root_type, SmemTensorType)
            else None
        )
        # Blocks split the contiguous dimension; the other one is the row.
        inner_dim = 0 if is_column_major(view_root_type) else 1
        outer_rows = view_root_type.sizes[1 - inner_dim]

        if layout is not None and layout[1] == "block_major":
            block_width, _ = layout
            parent_type = op.arg.type
            assert isinstance(parent_type, SmemTensorType)
            if (not _block_aligned_view(op.arg, block_width, inner_dim)
                    or (result_sizes[inner_dim] % block_width
                        and not isinstance(op.indices[inner_dim], int))):
                lowered, results = self._lower_block_major_subtile(
                    op, ptr, view_root, block_width, inner_dim, dtype_bytes)
                return [*new_ops, *lowered], results

        # Accumulate byte offset for each dimension.
        zero = IndexConstantOp.build(0)
        new_ops.append(zero)
        byte_offset = zero.result()

        for dim, idx in enumerate(op.indices):
            if isinstance(idx, int) and idx == 0:
                # Changing a view's extent at its origin adds no offset,
                # including a view of an existing sub-block tile.
                continue
            if layout is not None and layout[1] == "block_major" and dim == inner_dim:
                block_width, _ = layout
                if result_sizes[dim] % block_width:
                    parent_type = op.arg.type
                    assert isinstance(parent_type, AddrMemTensorType)
                    parent_width = parent_type.sizes[dim]
                    assert isinstance(parent_width, int)
                    assert parent_width % block_width == 0
                    assert isinstance(idx, int), (
                        "block-major sub-block tiles require a static index"
                    )
                    byte_delta = _block_major_offset(
                        0, idx * result_sizes[dim], block_width, cast(int, outer_rows)) * dtype_bytes
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
                stride = outer_rows if dim == inner_dim else block_width

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

    def _lower_block_major_subtile(
        self,
        op: TileOp,
        parent: Value,
        root: Value,
        block_width: int,
        inner_dim: int,
        dtype_bytes: int,
    ) -> tuple[list[Op], list[object]]:
        """Compose a sub-block view in logical coordinates, then map it back.

        Shared tile pointers carry physical, unswizzled offsets. Adding a
        logical column offset directly loses the carry between K blocks.
        """
        root_type = root.type
        assert isinstance(root_type, SmemTensorType) and root_type.rank == 2
        rows = root_type.sizes[1 - inner_dim]
        operands = [parent, cast(Value, self.value_map[root])]
        names = ["parent", "root"]
        template = (
            "{\n.reg .b32 tile_element, tile_inner, tile_row, tile_column, tile_block;\n"
            "sub.u32 tile_element, ${b32:operand.parent.ptr}, ${b32:operand.root.ptr};\n"
            + _constant_divmod_ptx("div", "tile_element", "tile_element", dtype_bytes)
            + _constant_divmod_ptx("rem", "tile_inner", "tile_element", block_width)
            + _constant_divmod_ptx("div", "tile_element", "tile_element", block_width)
            + _constant_divmod_ptx("rem", "tile_row", "tile_element", rows)
            + _constant_divmod_ptx("div", "tile_block", "tile_element", rows)
            + f"mad.lo.u32 tile_column, tile_block, {block_width}, tile_inner;\n"
        )
        for dim, index in enumerate(op.indices):
            coordinate = "tile_column" if dim == inner_dim else "tile_row"
            extent = op.result().type.sizes[dim]
            if isinstance(index, int):
                if index:
                    template += f"add.u32 {coordinate}, {coordinate}, {index * extent};\n"
            else:
                name = f"index{dim}"
                operands.append(cast(Value, self.value_map[index]))
                names.append(name)
                template += f"mad.lo.u32 {coordinate}, ${{b32:operand.{name}.val}}, {extent}, {coordinate};\n"
        template += (
            _constant_divmod_ptx("div", "tile_block", "tile_column", block_width)
            + _constant_divmod_ptx("rem", "tile_inner", "tile_column", block_width)
            + f"mad.lo.u32 tile_element, tile_block, {rows}, tile_row;\n"
            f"mad.lo.u32 tile_element, tile_element, {block_width}, tile_inner;\n"
            f"mad.lo.u32 ${{b32:result.dst.ptr}}, tile_element, {dtype_bytes}, ${{b32:operand.root.ptr}};\n"
            "}\n"
        )
        lowered, results = _lowered(op, template, operands, names, [PtrType.build("smem")])
        lowered[0].attrs["pure"] = True
        return lowered, results

    @lowers(AllocOp)
    def lower_alloc(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        if isinstance(result_type, TmemTensorType):
            return self._lower_alloc_tmem(op)
        if isinstance(result_type, SmemTensorType):
            return self._lower_alloc_smem(op)
        if isinstance(result_type, (RmemTensorType, WgrmemTensorType)):
            return self._lower_alloc_registers(op)
        if isinstance(result_type, LockType):
            return self._lower_alloc_lock(op)
        raise NotImplementedError(
            f"lower_alloc: unsupported result type {type(result_type).__name__}"
        )

    @lowers(ReinterpretOp)
    def lower_reinterpret(
        self,
        op: ReinterpretOp,
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
        self,
        op: ReshapeOp,
    ) -> tuple[list[Op], list[object]]:
        return [], [cast(Value, self.value_map[op.arg])]

    @lowers(OperandPipelineOp)
    def lower_operand_pipeline(self, op):
        from .pipeline_rolled_k import LowerRolledK
        from ..base.ir import Block

        ops, _ = LowerRolledK().lower_pipeline(op)
        body = Block(ops=ops)
        op.blocks = [body]
        # Expand while views are still ordinary IR: otherwise dead pointer
        # preparation becomes an opaque PTX template and cannot be removed.
        from .cse import _cse_block
        from .dce import DCE

        _cse_block(body, {})
        DCE().run_block(body)
        from .verify_gpu_dialect import verify_expanded_gpu_hardware

        verify_expanded_gpu_hardware(op)
        # Kernel-level issuer grouping ran before this semantic region was
        # expanded. Group its new loops now so other warps skip the whole
        # issuer run instead of traversing every bank's arithmetic and guards.
        _group_issuer_regions(body, self.warp_ids.x if self.warp_ids is not None else None)
        self.lower_block(body)
        return list(body.ops), []

    @lowers(LockViewOp)
    def lower_lock_view(self, op):
        arena, index, phase = [self.value_map[v] for v in op.operands]
        template = "mad.lo.u32 ${b32:result.dst.val}, ${b32:operand.index.val}, 8, ${b32:operand.arena.val};\nmov.b32 ${b32:result.dst.parity}, ${b32:operand.phase.val};\n"
        lowered = PTXTemplateOp.build(
            operands=[arena, index, phase],
            result_types=[op.result().type],
            template=template,
            operand_names=["arena", "index", "phase"],
            result_names=["dst"],
        )
        return [lowered], [lowered.result()]

    def _lower_alloc_lock(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        assert isinstance(lock_type, LockType)
        issuer = 0
        if lock_type.kind == "cluster_mbar":
            init = self._cluster_mbar_init(op)
        elif lock_type.kind in ("mbar", "tcgen05"):
            # InsertSyncs represents publication explicitly with BarrierOps.
            release_prior = any(isinstance(u, AsyncOp) and u.lock is op.result()
                                and u.attrs.get("release_prior")
                                for u in op.result().users())
            arrivals = lock_type.attrs.get("arrival_count", lock_type.attrs.get("arrivals", 2 if release_prior else 1))
            counts = lock_type.attrs.get("arrival_counts")
            setup = lambda s: ""
            if counts is not None and len(set(counts)) == 1:
                arrivals = counts[0]
            elif counts is not None:
                arrivals = "init_arrivals"
                setup = lambda s: (f"mov.u32 ${{b32:init_rank{s}}}, %cluster_ctarank;\n"
                                   + rank_table_ptx(f"init_arrivals{s}", counts, rank=f"init_rank{s}"))
            init = _mbarrier_init(lock_type.attrs.get("count", 1), arrivals, setup)
            issuer = execution_issuer_warp(op)
        else:
            return [], [None]
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[lock_type],
            template=_event_inits_ptx(issuer, [init]),
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        self._event_inits[id(new_op)] = (issuer, init)
        return [new_op], [new_op.result()]

    def _cluster_mbar_init(self, op: AllocOp) -> EventInit:
        lock_type = op.result().type
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None and kernel.num_threads is not None
        cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
        warps = kernel.num_threads // 32
        if _split_release(op.result()):
            # Every CTA's leader waits for one arrival per warp of the
            # cluster: any CTA may write the released stage.
            masks = lock_type.attrs.get('release_cta_masks')
            if masks is not None:
                from ..cluster_layout import arrival_counts
                counts = arrival_counts(masks)
                if len(set(counts)) != 1:
                    raise ValueError('software release currently requires uniform recipient counts')
                cluster_size = counts[0]
            return _mbarrier_init(1, len(lock_type.attrs.get("release_warps", range(warps))) * cluster_size)
        # Every warp of the CTA arrives, and at the cluster leader every warp
        # of the cluster.
        return _mbarrier_init(1, "arrivals", lambda s: textwrap.dedent(f"""\
            mov.u32 ${{b32:cluster_rank{s}}}, %cluster_ctarank;
            setp.eq.u32 ${{pred:rank0{s}}}, ${{b32:cluster_rank{s}}}, 0;
            mov.u32 ${{b32:arrivals{s}}}, {warps};
            @${{pred:rank0{s}}} mov.u32 ${{b32:arrivals{s}}}, {warps * cluster_size};
            """))

    def _group_event_inits(self, block) -> None:
        """Initialize each run of consecutive event allocations in one block (see _event_inits_ptx)."""
        def key(op):
            entry = self._event_inits.get(id(op))
            return entry and (entry[0], op.extra_attrs)

        ops, out, i = block.ops, [], 0
        while i < len(ops):
            first = ops[i]
            run = list(itertools.takewhile(lambda op: key(op) and key(op) == key(first), ops[i:]))
            if len(run) < 2:
                out.append(first)
                i += 1
                continue
            merged = PTXTemplateOp.build(
                operands=[],
                result_types=[op.results[0].type for op in run],
                template=_event_inits_ptx(self._event_inits[id(first)][0],
                                          [self._event_inits[id(op)][1] for op in run]),
                operand_names=[],
                result_names=[f"dst{n}" for n in range(len(run))],
                description=f"{first.description} (+{len(run) - 1} events initialized together)",
                attrs=dict(first.extra_attrs),
            )
            for op, result in zip(run, merged.results):
                for user, index in list(op.results[0].uses):
                    user.operands[index] = result
            out.append(merged)
            i += len(run)
        block.ops = out

    def _lower_alloc_registers(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        return _lowered(op, "", result_types=[op.result().type])

    def _lower_alloc_smem(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        smem_type = op.result().type
        assert isinstance(smem_type, SmemTensorType)
        num_elements = smem_type.num_elements
        assert num_elements is not None
        n_bytes = (
            num_elements
            * cast(
                BF16Type | F16Type | F32Type | F64Type,
                smem_type.dtype,
            ).byte_size
        )
        # TMA swizzling and the corresponding software address swizzle operate
        # within 1 KiB regions and require the shared-memory base to match.
        alignment = (
            1024
            if smem_type.swizzle is not None and physical_layout(smem_type) is not None
            else 128
        )
        alignment = op.attrs.get("allocation_alignment", alignment)
        template = textwrap.dedent(f"""\
            .shared .align {alignment} .b8 ${{smem:sym}}[{n_bytes}];
            mov.u32 ${{b32:result.dst.ptr}}, ${{smem:sym}};
        """)
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[smem_type],
            template=template,
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    @lowers(ElementwiseOp)
    def lower_elementwise(
        self,
        op: ElementwiseOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower an element-wise op on register tensors, one function per element.

        Source and destination dtypes are independent. Function lowering runs
        in the source dtype and a supported conversion writes the result to the
        destination dtype.
        """
        if isinstance(op.dst.type, TmemTensorType):
            return self._lower_fill_tmem(op)
        sources = [op.src(i) for i in range(op.num_srcs)]
        dst_value = cast(Value, self.value_map[op.dst])
        source_values = [cast(Value, self.value_map[src]) for src in sources]

        for value in op.operands:
            if not isinstance(value.type, (RmemTensorType, WgrmemTensorType)):
                raise NotImplementedError(
                    f"lower_elementwise: operands must be registers; got {value.type}"
                )
        lower = _ELEMENTWISE_LOWERERS.get(op.fn)
        if lower is None:
            raise NotImplementedError(f"lower_elementwise: unsupported fn {op.fn!r}")
        template = lower(op)

        source_names = _elementwise_source_names(op)
        assert len(source_names) == op.num_srcs
        return _lowered(op, template, [dst_value, *source_values], ["dst", *source_names])

    @lowers(MatmulOp)
    def lower_matmul(
        self,
        op: MatmulOp,
    ) -> tuple[list[Op], list[object]]:
        lhs_type = op.lhs.type
        rhs_type = op.rhs.type
        dst_type = op.dst.type
        if (
            isinstance(lhs_type, RmemTensorType)
            and isinstance(rhs_type, RmemTensorType)
            and isinstance(dst_type, RmemTensorType)
        ):
            return self._lower_matmul_fma(op)
        if (
            isinstance(lhs_type, WgrmemTensorType)
            and isinstance(rhs_type, WgrmemTensorType)
            and isinstance(dst_type, WgrmemTensorType)
        ):
            return self._lower_matmul_mma_sync(op)
        raise NotImplementedError(
            f"lower_matmul: unsupported operand types; got "
            f"{lhs_type} @ {rhs_type} -> {dst_type}"
        )

    def _lower_matmul_fma(
        self,
        op: MatmulOp,
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
                f"lower_async_matmul fma: k-dim mismatch; got {lhs.type} @ {rhs.type}"
            )
        if tuple(dst_type.sizes) != (m_dim, n_dim):
            raise NotImplementedError(
                f"lower_async_matmul fma: dst shape must be {m_dim}x{n_dim}; got {dst.type}"
            )
        template = _zero_dst_ptx(op, m_dim * n_dim)
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
        self,
        op: MatmulOp,
    ) -> tuple[
        list[Op], list[object]
    ]:  # TODO: rename "mma" to "mma_layout_0, 1, 2" and impl it the right way
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
        if not (
            isinstance(lhs_type.dtype, (BF16Type, F16Type))
            and isinstance(rhs_type.dtype, (BF16Type, F16Type))
            and isinstance(dst_type.dtype, F32Type)
        ):
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
        template = _zero_dst_ptx(op, n_dim // 8 * 4)
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
        self,
        op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        body_ops = op.body.ops
        if op.lock.type.kind == "bulk" and body_ops:
            stores = []
            conditions = []
            for inner in body_ops:
                if isinstance(inner, TmaMoveOp):
                    stores.append(inner)
                    conditions.append(None)
                elif (isinstance(inner, IfOp) and len(inner.body.ops) == 1
                      and isinstance(inner.body.ops[0], TmaMoveOp)):
                    stores.append(inner.body.ops[0])
                    conditions.append(inner.condition)
                else:
                    break
            else:
                return self._lower_async_tma_stores(stores, conditions)
        tmas = [body_op for body_op in body_ops if isinstance(body_op, TmaMoveOp)]
        if body_ops and len(tmas) == len(body_ops):
            lock_type = op.lock.type
            assert isinstance(lock_type, LockType)
            kind = lock_type.kind
            if kind == "mbar":  # one or more gmem→smem loads sharing the mbar
                lock_value = cast(Value, self.value_map[op.lock])
                return self._lower_async_tma_loads(op, tmas, lock_value)
            if kind == "bulk":
                return self._lower_async_tma_stores(tmas)
            raise NotImplementedError(f"lower_async: tma_move with lock kind {kind!r}")
        mms = [body_op for body_op in body_ops if isinstance(body_op, MatmulOp)]
        if body_ops and len(mms) == len(body_ops):
            probes = self._probe_ahead_ops(op)
            new_ops, results = self._lower_async_matmul(op, mms)
            return [*probes, *new_ops], results
        raise NotImplementedError(
            f"lower_async: unsupported body ops {[type(o).__name__ for o in body_ops]}"
        )

    def _lower_async_matmul(
        self,
        op: AsyncOp,
        mms: list[MatmulOp],
    ) -> tuple[list[Op], list[object]]:
        """Lower a group of smem × smem matmuls to wgmma or tcgen05 MMAs.

        Into a WGRMEM accumulator, the whole group is one wgmma commit group: a
        single ``wgmma.fence``, all the ``mma_async`` (accumulating into the
        shared dst), and a single ``wgmma.commit_group``. The hardware
        serializes same-accumulator mmas, so the await's lone
        ``wgmma.wait_group`` covers them all (this is the batched form nvcc
        emits, cf. ``fastcu_5_1``). The group's await op (lock kind ``wgmma``)
        emits the wait plus a ``bar.sync`` that closes the TMA-reload vs
        warpgroup-read WAR (see ``_lower_await_wgmma``).

        Into a TMEM accumulator, one elected thread issues the group's
        ``tcgen05.mma`` ops and commits them to the lock's mbarrier. Both
        families read the same shared-memory matrix descriptors; tcgen05's
        carry a version field, and one instruction covers the logical M64/M128 tile.
        """
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        assert lock_type.kind in ("wgmma", "tcgen05"), (
            f"_lower_async_matmul: unexpected lock kind {lock_type.kind!r}"
        )
        tcgen05 = lock_type.kind == "tcgen05"
        cta_group = lock_type.attrs.get("cta_group", 1)

        _swizzle_to_bits = {None: 0, "128B": 1, "64B": 2, "32B": 3}

        def _build_static_desc(lbo, sbo, swizzle):
            version = 1 << 46 if tcgen05 else 0
            return (lbo >> 4) << 16 | (sbo >> 4) << 32 | version | swizzle << 62

        # Default preparation stays near the MMA to limit register lifetimes.
        # SM100 can instead expose the variable low half to late LICM.
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        clustered_addresses = any((dim or 1) > 1 for dim in kernel.cluster_dim)

        def _build_desc(name, ptr_reg, desc_const, offset=None, *, low_half=False):
            t = ""
            if offset is not None:
                t += f"add.u32 ${{b32:{name}_addr}}, ${{b32:{ptr_reg}}}, {offset};\n"
                t += f"shr.b32 ${{b32:{name}_addr}}, ${{b32:{name}_addr}}, 4;\n"
            else:
                t += f"shr.b32 ${{b32:{name}_addr}}, ${{b32:{ptr_reg}}}, 4;\n"
            # A cluster shared pointer includes nonlocal CTA bits. Only the
            # 14-bit address field belongs in the matrix descriptor; higher
            # bits would corrupt its leading byte offset for MN-major B.
            if clustered_addresses:
                t += f"and.b32 ${{b32:{name}_addr}}, ${{b32:{name}_addr}}, 16383;\n"
            if low_half:
                return t + (
                    f"or.b32 ${{b32:{name}}}, ${{b32:{name}_addr}}, "
                    f"{desc_const & 0xffffffff};\n"
                )
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
            # Reuse exact static blocked offsets only in compact SM100 batches.
            if typ.attrs.get("physical_layout") is not None:
                if not compact_mma:
                    return None
                layout = physical_layout(typ)
                if layout is None or layout[1] != 'block_major' or typ.rank != 2:
                    return None
                root = _layout_view_root(value)
                if root is None or not isinstance(root.type, SmemTensorType) or root.type.rank != 2:
                    return None
                width = layout[0]
                inner = 0 if is_column_major(root.type) else 1
                rows = root.type.sizes[1 - inner]
                parent = tile.arg.type
                if (not isinstance(rows, int) or not isinstance(parent, SmemTensorType)
                        or parent.sizes[inner] % width):
                    return None
                # Static siblings share every preceding view and descriptor
                # field. Evaluate the actual blocked pointer offsets; no
                # affine assumption about logical coordinates is needed.
                def offset(coords):
                    starts = [i * size for i, size in zip(coords, typ.sizes)]
                    if any(start < 0 or start + size > extent
                           for start, size, extent in zip(starts, typ.sizes, parent.sizes)):
                        return None
                    return _block_major_offset(starts[1 - inner], starts[inner], width, rows)
                first, origin = offset(indices), offset(base_indices)
                if first is None or origin is None:
                    return None
                byte_delta = (first - origin) * typ.dtype.byte_size
                return byte_delta if byte_delta >= 0 and byte_delta % 16 == 0 else None
            delta = 0
            for index, base_index, size, stride in zip(
                indices,
                base_indices,
                typ.sizes,
                typ.strides,
            ):
                assert index is not None and base_index is not None
                assert size is not None and stride is not None
                delta += (index - base_index) * size * stride
            byte_delta = (
                delta
                * cast(
                    BF16Type | F16Type | F32Type | F64Type,
                    typ.dtype,
                ).byte_size
            )
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

        descriptor_ops: list[Op] = []
        rolled_mma = tcgen05 and any(ancestor in self._operand_loops for ancestor in self._lowering)
        hoist_descriptors = tcgen05 and (kernel.attrs.get("hoist_mma_descriptors", False)
            or rolled_mma)
        descriptor_lows = {}
        local_lows = {}

        def _desc(name, base, delta, value, hint, const, offset=None):
            if hoist_descriptors and delta is None:
                # The high descriptor fields are constant. Keeping their
                # 64-bit join near the MMA avoids a long-lived 64-bit table.
                preparation = _build_desc(
                    "desc", "operand.ptr.ptr", const, offset, low_half=True
                ).replace("${b32:desc}", "${b32:result.desc.val}")
                prepared = MmaDescriptorLowOp.build(
                    operands=[cast(Value, self.value_map[value])],
                    result_types=[IndexType()],
                    template=preparation,
                    operand_names=["ptr"],
                    result_names=["desc"],
                    description="pure MMA descriptor low half",
                    attrs={"pure": True},
                )
                descriptor_ops.append(prepared)
                bound = _operand(prepared.result(), "descriptor_low")
                descriptor_lows[name] = bound, const
                if rolled_mma:
                    full = MmaDescriptorOp.build([prepared.result()], [MmaDescriptorType()],
                        f'mov.b64 ${{b64:result.desc.val}}, {{${{b32:operand.low.val}}, {const >> 32}}};\n',
                        operand_names=['low'], result_names=['desc'],
                        description='pure rolled MMA descriptor', attrs={'pure': True})
                    descriptor_ops.append(full)
                    complete = _operand(full.result(), 'descriptor')
                    return f'mov.b64 ${{b64:{name}}}, ${{b64:operand.{complete}.val}};\n'
                return (
                    f"mov.b64 ${{b64:{name}}}, "
                    f"{{${{b32:operand.{bound}.val}}, {const >> 32}}};\n"
                )
            if delta is not None:
                if rolled_mma and base in descriptor_lows:
                    bound, base_const = descriptor_lows[base]
                    full = MmaDescriptorOp.build([operands[operand_names.index(bound)]], [MmaDescriptorType()],
                        f'add.u32 ${{b32:low}}, ${{b32:operand.base.val}}, {delta // 16};\n'
                        f'mov.b64 ${{b64:result.desc.val}}, {{${{b32:low}}, {base_const >> 32}}};\n',
                        operand_names=['base'], result_names=['desc'],
                        description='pure rolled MMA descriptor', attrs={'pure': True})
                    descriptor_ops.append(full)
                    complete = _operand(full.result(), 'descriptor')
                    return f'mov.b64 ${{b64:{name}}}, ${{b64:operand.{complete}.val}};\n'
                if base in local_lows:
                    # Valid start addresses never carry out of the low word.
                    # A 64-bit add lets ptxas split each sibling into a
                    # carry chain plus a rematerialized high half.
                    base_low, base_const = local_lows[base]
                    return (f"add.u32 ${{b32:{name}_low}}, ${{b32:{base_low}}}, {delta // 16};\n"
                            f"mov.b64 ${{b64:{name}}}, {{${{b32:{name}_low}}, {base_const >> 32}}};\n")
                return f"add.u64 ${{b64:{name}}}, ${{b64:{base}}}, {delta // 16};\n"
            operand = _operand(cast(Value, self.value_map[value]), hint)
            if tcgen05:
                local_lows[name] = f"{name}_low", const
                return _build_desc(f"{name}_low", f"operand.{operand}.ptr", const, offset,
                                   low_half=True) + (
                    f"mov.b64 ${{b64:{name}}}, {{${{b32:{name}_low}}, {const >> 32}}};\n")
            return _build_desc(
                name,
                f"operand.{operand}.ptr",
                const,
                offset,
            )

        desc_template = ""
        mma_template = ""
        kernel = find_ancestor_of_type(op, KernelOp)
        compact_mma = tcgen05 and (
            (cta_group == 2 and len(mms) >= 8
             and all(mm.dst.type.sizes[0] == 64 for mm in mms))
            or (kernel.attrs.get("compact_mma_batch", False) and len(mms) >= 2
                and all(mm.dst.type.sizes[0] in (64, 128) for mm in mms))
        )
        # Descriptor lifetime and elected-lane control are independent choices.
        # Preparing descriptors near their MMA avoids keeping every sibling
        # live together, without moving that preparation into a lane branch.
        interleave_descriptors = rolled_mma or compact_mma or (
            tcgen05 and kernel.attrs.get('interleave_mma_descriptors', False)
        )
        elected_batch = compact_mma

        def row_stride_bytes(typ: SmemTensorType) -> int:
            # A blocked buffer's rows are one block wide.
            layout = physical_layout(typ)
            stride = layout[0] if layout is not None and layout[1] == "block_major" else typ.strides[0]
            return cast(int, stride) * typ.dtype.byte_size

        collector_a_modes = [""] * len(mms)
        if tcgen05 and kernel.attrs.get("reuse_mma_collector_a", False):
            # Reuse only consecutive reads of exactly the same A view in one
            # elected instruction batch. End each run before any other A read.
            start = 0
            while start < len(mms):
                end = start + 1
                while end < len(mms) and mms[end].dst.type.sizes == mms[start].dst.type.sizes and (
                    mms[end].lhs is mms[start].lhs
                    or _tile_delta(mms[end].lhs, mms[start].lhs, end) == 0
                ):
                    end += 1
                if end - start > 1:
                    collector_a_modes[start] = ".collector::a::fill"
                    for index in range(start + 1, end - 1):
                        collector_a_modes[index] = ".collector::a::use"
                    collector_a_modes[end - 1] = ".collector::a::lastuse"
                start = end

        for j, mm in enumerate(mms):
            descriptor_start = len(desc_template)
            mma_start = len(mma_template)
            lhs_type = mm.lhs.type
            rhs_type = mm.rhs.type
            dst_type = mm.dst.type
            assert (
                isinstance(lhs_type, SmemTensorType)
                and isinstance(rhs_type, SmemTensorType)
                and (
                    isinstance(dst_type, TmemTensorType)
                    if tcgen05
                    else isinstance(dst_type, WgrmemTensorType)
                    and dst_type.warps == 4
                    and dst_type.layout == "wgmma_layout_0"
                )
            ), (
                f"_lower_async_matmul: unsupported matmul operand types; "
                f"got {lhs_type} @ {rhs_type} -> {dst_type}"
            )
            assert lhs_type.rank == 2 and rhs_type.rank == 2 and dst_type.rank == 2
            lhs_column = is_column_major(lhs_type)
            rhs_column = is_column_major(rhs_type)
            assert lhs_column or lhs_type.strides[-1] == 1
            assert rhs_column or rhs_type.strides[-1] == 1
            m_dim, k_dim = cast(tuple[int, int], lhs_type.sizes)
            n_dim, k_dim2 = cast(tuple[int, int], rhs_type.sizes)
            if tcgen05 and cta_group == 2:
                n_dim *= 2
            assert k_dim == k_dim2
            assert k_dim == 16, f"_lower_async_matmul: k must be 16; got {k_dim}"
            assert m_dim % 64 == 0, (
                f"_lower_async_matmul: m must be multiple of 64; got {m_dim}"
            )
            assert n_dim % 8 == 0 and 8 <= n_dim <= 256, (
                f"_lower_async_matmul: n must be a multiple of 8 in [8, 256]; got {n_dim}"
            )
            assert list(dst_type.sizes) == [m_dim, n_dim]

            m_iters = 1 if tcgen05 else m_dim // 64
            lhs_row_stride_bytes = row_stride_bytes(lhs_type)
            rhs_row_stride_bytes = row_stride_bytes(rhs_type)

            assert lhs_type.swizzle in _swizzle_to_bits, (
                f"_lower_async_matmul: lhs has unknown swizzle {lhs_type.swizzle!r}"
            )
            assert rhs_type.swizzle in _swizzle_to_bits, (
                f"_lower_async_matmul: rhs has unknown swizzle {rhs_type.swizzle!r}"
            )

            def descriptor(value, typ, column, row_stride_bytes):
                if not column:
                    return _build_static_desc(
                        16, row_stride_bytes * 8, _swizzle_to_bits[typ.swizzle]
                    )
                layout = physical_layout(typ)
                assert layout is not None and layout[1] == "block_major", (
                    "MN-major WGMMA needs compact swizzle atoms"
                )
                width = layout[0]
                root = _layout_view_root(value)
                assert root is not None and isinstance(root.type, SmemTensorType)
                reduction_tile = root.type.sizes[-1]
                assert isinstance(reduction_tile, int)
                byte_size = typ.dtype.byte_size
                lbo = reduction_tile * width * byte_size
                sbo = 8 * width * byte_size
                return _build_static_desc(lbo, sbo, _swizzle_to_bits[typ.swizzle])

            desc_const_lhs = descriptor(
                mm.lhs, lhs_type, lhs_column, lhs_row_stride_bytes
            )
            desc_const_rhs = descriptor(
                mm.rhs, rhs_type, rhs_column, rhs_row_stride_bytes
            )

            lhs_delta = _tile_delta(mm.lhs, mms[0].lhs, j)
            rhs_delta = _tile_delta(mm.rhs, mms[0].rhs, j)
            dst_name = _operand(cast(Value, self.value_map[mm.dst]), "dst")

            desc_rhs = "desc_rhs0" if rolled_mma else f"desc_rhs{j}"
            desc_template += _desc(
                desc_rhs,
                "desc_rhs0",
                rhs_delta,
                mm.rhs,
                "rhs",
                desc_const_rhs,
            )
            for m_it in range(m_iters):
                lhs_offset = m_it * 64 * lhs_row_stride_bytes
                if lhs_column:
                    lhs_root = _layout_view_root(mm.lhs)
                    lhs_width, _ = physical_layout(lhs_type)
                    lhs_offset = (
                        m_it
                        * 64
                        // lhs_width
                        * lhs_root.type.sizes[-1]
                        * lhs_width
                        * lhs_type.dtype.byte_size
                    )
                desc_lhs = f"desc_lhs0_{m_it}" if rolled_mma else f"desc_lhs{j}_{m_it}"
                desc_template += _desc(
                    desc_lhs,
                    f"desc_lhs0_{m_it}",
                    lhs_delta,
                    mm.lhs,
                    "lhs",
                    desc_const_lhs,
                    lhs_offset if m_it > 0 else None,
                )

            input_dtype = lhs_type.dtype.name
            if rhs_type.dtype.name != input_dtype:
                raise NotImplementedError(
                    "wgmma inputs must have the same dtype; got "
                    f"{lhs_type.dtype} and {rhs_type.dtype}"
                )
            if tcgen05:
                if input_dtype not in ("bf16", "f16"):
                    raise NotImplementedError(
                        f"tcgen05 matmuls support f16 and bf16 inputs; got {input_dtype}"
                    )
                # kind::f16 instruction descriptor: F32 accumulator, input
                # format (f16 0, bf16 1) for A and B, MN-major operand bits,
                # then N / 8 and M / 16.
                input_format = int(input_dtype == "bf16")
                idesc = (
                    1 << 4
                    | input_format << 7
                    | input_format << 10
                    | int(lhs_column) << 15
                    | int(rhs_column) << 16
                    | n_dim >> 3 << 17
                    | (m_dim * cta_group) >> 4 << 24
                )
                if rolled_mma:
                    instruction = PTXTemplateOp.build([], [IndexType(), PredicateType()],
                        f'mov.b32 ${{b32:result.idesc.val}}, {idesc};\n'
                        f'mov.pred ${{pred:result.accumulate.val}}, {int(mm.accumulate)};\n',
                        result_names=['idesc', 'accumulate'], attrs={'pure': True},
                        description='rolled MMA constants')
                    descriptor_ops.append(instruction)
                    idesc_name = _operand(instruction.results[0], 'idesc')
                    accumulate_name = _operand(instruction.results[1], 'accumulate')
                    setup = ''
                    idesc_ptx = f'${{b32:operand.{idesc_name}.val}}'
                    accumulate_ptx = f'${{pred:operand.{accumulate_name}.val}}'
                else:
                    setup = (f'mov.b32 ${{b32:idesc}}, {idesc};\n'
                             f'mov.pred ${{pred:accumulate}}, {int(mm.accumulate)};\n')
                    idesc_ptx, accumulate_ptx = '${b32:idesc}', '${pred:accumulate}'
                mma_template += setup + textwrap.dedent(f"""\
                    @${{pred:tcgen05_leader}} tcgen05.mma.cta_group::{cta_group}.kind::f16{collector_a_modes[j]}
                      [${{b32:operand.{dst_name}.ptr}}], ${{b64:{'desc_lhs0_0' if rolled_mma else f'desc_lhs{j}_0'}}}, ${{b64:{desc_rhs}}},
                      {idesc_ptx}, {accumulate_ptx};
                """)
                if interleave_descriptors:
                    mma_template = (mma_template[:mma_start] + desc_template[descriptor_start:]
                                    + mma_template[mma_start:])
                    desc_template = desc_template[:descriptor_start]
                continue
            num_regs_per_wgmma = 64 * n_dim // (dst_type.warps * 32)
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
                      {int(mm.accumulate)}, 1, 1, {int(lhs_column)}, {int(rhs_column)};
                """)
                reg_offset += num_regs_per_wgmma

        # One commit group for the whole batch: single fence, all mma_async, one
        # commit. Hardware serializes same-accumulator mmas, so the await's lone
        # wait_group covers them all. The await also emits an asymmetric named
        # barrier that closes the cross-warp WAR (thread-0 TMA reload vs
        # warpgroup-wide wgmma reads).
        if tcgen05:
            release_prior = op.attrs.get("release_prior", False)
            uniform_pair = rolled_mma and cta_group == 2 and not release_prior
            before_election = ''
            if uniform_pair:
                rank = PTXTemplateOp.build([], [IndexType(), PredicateType()],
                    _pair_leader_ptx('rank', 'leader').replace('${b32:rank}', '${b32:result.rank.val}')
                    .replace('${pred:leader}', '${pred:result.leader.val}'),
                    result_names=['rank', 'leader'], attrs={'pure': True},
                    description='rolled CTA pair identity')
                descriptor_ops.append(rank)
                rank_name = _operand(rank.results[0], 'pair_rank')
                rank_pred = _operand(rank.results[1], 'pair_leader')
                before_election = (f'mov.b32 ${{b32:tcgen05_rank}}, ${{b32:operand.{rank_name}.val}};\n'
                    f'@!${{pred:operand.{rank_pred}.val}} bra ${{label:tcgen05_peer_done}};\n')
            leader_operands, leader_names, leader = self._leader_thread("tcgen05",
                before_branch=before_election,
                issuer_warp=execution_issuer_warp(op),
                converge=release_prior, cached=not rolled_mma)
            operands.extend([*leader_operands, cast(Value, self.value_map[op.lock])])
            operand_names.extend([*leader_names, "lock"])
            if cta_group == 2 and not uniform_pair:
                # The peer CTA branches past every issue that tests the
                # elected lane. Keeping that lane unconditioned on the pair
                # rank leaves ptxas a uniform issue predicate (no VOTEU).
                leader += ("mov.pred ${pred:tcgen05_local_leader}, ${pred:tcgen05_leader};\n"
                           + _pair_leader_ptx("tcgen05_rank", "tcgen05_rank0")
                           + "@!${pred:tcgen05_rank0} bra ${label:tcgen05_peer_done};\n")
            kernel = find_ancestor_of_type(op, KernelOp)
            size = math.prod(d or 1 for d in kernel.cluster_dim)
            if cta_group == 1 and any('completion_cta_masks' in event.type.attrs for event in op.completion_locks):
                leader += "mov.u32 ${b32:tcgen05_rank}, %cluster_ctarank;\n"
            commit = ""
            for index, event in enumerate(op.completion_locks):
                name = "lock" if index == 0 else f"completion{index-1}"
                if index:
                    operands.append(cast(Value, self.value_map[event]))
                    operand_names.append(name)
                masks = event.type.attrs.get('completion_cta_masks')
                if masks is None and cta_group == 2:
                    masks = [pair_mask(rank) for rank in range(size)]
                if masks is not None:
                    mask_name = f"tcgen05_mask{index}"
                    commit += rank_table_ptx(mask_name, masks, bits=16, rank='tcgen05_rank')
                    suffix = f".shared::cluster.multicast::cluster.b64 [${{b32:operand.{name}.val}}], ${{b16:{mask_name}}};\n"
                else:
                    suffix = f".shared::cluster.b64 [${{b32:operand.{name}.val}}];\n"
                commit += f"@${{pred:tcgen05_leader}} tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one" + suffix
            release_pred = "tcgen05_local_leader" if cta_group == 2 else "tcgen05_leader"
            if elected_batch:
                leader += "@!${pred:tcgen05_leader} bra ${label:tcgen05_issue_done};\n"
                mma_template = mma_template.replace("@${pred:tcgen05_leader} ", "")
                commit = commit.replace("@${pred:tcgen05_leader} ", "")
            template = (
                leader
                + desc_template
                + mma_template
                + commit
                + ("${label:tcgen05_issue_done}:\n" if elected_batch else "")
                + ("${label:tcgen05_peer_done}:\n" if cta_group == 2 else "")
                + (f"@${{pred:{release_pred}}} mbarrier.arrive.release.cta.shared::cta.b64 "
                   "_, [${b32:operand.lock.val}];\n" if release_prior else "")
                + "${label:tcgen05_done}:\n"
            )
        else:
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
        return [*descriptor_ops, new_op], []

    @lowers(MoveOp)
    def lower_move(
        self,
        op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower synchronous MoveOp to PTX."""
        return self._physical_fragment_coordinates(op, self._lower_move_impl(op))

    @staticmethod
    def _physical_fragment_coordinates(op, result):
        """TMEM copy lanes and warp partitions use CUDA's flattened thread ID.

        Logical x coordinates remain unchanged for ordinary scalar transfers.
        Keep the existing one-dimensional fragment code byte-for-byte.
        """
        if not any(isinstance(v.type, TmemTensorType)
                   or isinstance(v.type, WgrmemTensorType) and v.type.layout in TMEM_FRAGMENT_LAYOUTS
                   for v in op.operands):
            return result
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        bx, by, bz = kernel.block_dim
        if (by, bz) == (1, 1):
            return result
        for index, generated in enumerate(result[0]):
            if isinstance(generated, ThreadIdOp) and generated.axis == CollectiveAxis.X:
                # M16N8 writeback constructs its physical lane arithmetic as
                # index IR rather than inside the final memory template.
                linear = PTXTemplateOp.build(
                    operands=[], result_types=[IndexType()],
                    template=textwrap.dedent(f"""\
                        mov.u32 ${{b32:result.dst.val}}, %tid.x;
                        mov.u32 ${{b32:physical_y}}, %tid.y;
                        mad.lo.u32 ${{b32:result.dst.val}}, ${{b32:physical_y}}, {bx}, ${{b32:result.dst.val}};
                        mov.u32 ${{b32:physical_z}}, %tid.z;
                        mad.lo.u32 ${{b32:result.dst.val}}, ${{b32:physical_z}}, {bx * by}, ${{b32:result.dst.val}};
                    """), operand_names=[], result_names=['dst'],
                    description='physical fragment thread index', attrs={'pure': True})
                for user, operand in list(generated.result().uses):
                    user.operands[operand] = linear.result()
                result[0][index] = linear
                continue
            if not isinstance(generated, PTXTemplateOp):
                continue
            template = generated.template
            if '%tid.x' not in template and '${b32:operand.warp.val}' not in template:
                continue
            prefix = textwrap.dedent(f"""\
                mov.u32 ${{b32:physical_thread}}, %tid.x;
                mov.u32 ${{b32:physical_y}}, %tid.y;
                mad.lo.u32 ${{b32:physical_thread}}, ${{b32:physical_y}}, {bx}, ${{b32:physical_thread}};
                mov.u32 ${{b32:physical_z}}, %tid.z;
                mad.lo.u32 ${{b32:physical_thread}}, ${{b32:physical_z}}, {bx * by}, ${{b32:physical_thread}};
            """)
            template = template.replace('%tid.x', '${b32:physical_thread}')
            if '${b32:operand.warp.val}' in template:
                prefix += 'shr.u32 ${b32:physical_warp}, ${b32:physical_thread}, 5;\n'
                template = template.replace('${b32:operand.warp.val}', '${b32:physical_warp}')
            generated.attrs['template'] = prefix + template
        return result

    def _lower_move_impl(self, op):
        dst_type = op.dst.type
        src_type = op.src.type

        if isinstance(dst_type, TmemTensorType) or isinstance(src_type, TmemTensorType):
            return self._lower_move_tmem_fragment(op)
        fragment_type = dst_type if isinstance(dst_type, WgrmemTensorType) else src_type
        if (isinstance(fragment_type, WgrmemTensorType)
            and fragment_type.layout in TMEM_FRAGMENT_LAYOUTS
            and TMEM_FRAGMENT_LAYOUTS[fragment_type.layout].row_fragment):
            return self._lower_move_row_fragment(op)
        if {type(dst_type), type(src_type)} == {RmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_rmem(op)
        if {type(dst_type), type(src_type)} == {RmemTensorType, SmemTensorType}:
            return self._lower_move_smem_rmem(op)
        if {type(dst_type), type(src_type)} == {SmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_smem(op)
        if (
            isinstance(dst_type, WgrmemTensorType)
            and isinstance(src_type, AddrMemTensorType)
            and dst_type.layout in _M16N8_FRAGMENT_ROWS
        ):
            return self._lower_move_wgrmem_wgmma_addrmem(op)
        if (
            isinstance(dst_type, WgrmemTensorType)
            and isinstance(src_type, SmemTensorType)
            and dst_type.layout == "mma"
        ):
            return self._lower_move_smem_to_wgrmem_mma(op)
        if (
            isinstance(dst_type, SmemTensorType)
            and isinstance(src_type, WgrmemTensorType)
            and src_type.layout == "mma"
        ):
            return self._lower_move_wgrmem_mma_to_smem(op)
        if (
            isinstance(dst_type, (SmemTensorType, GmemTensorType))
            and isinstance(src_type, WgrmemTensorType)
            and src_type.layout in _M16N8_FRAGMENT_ROWS
        ):
            return self._lower_move_wgrmem_wgmma(op)
        raise NotImplementedError(
            f"lower_move: unsupported; got "
            f"{type(dst_type).__name__} <- {type(src_type).__name__}"
        )

    @lowers(AwaitOp)
    def lower_await(
        self,
        op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        kind = lock_type.kind
        if kind in ("mbar", "tcgen05"):
            lock = cast(Value, self.value_map[op.lock])
            return self._lower_await_mbar(op, lock)
        if kind == "wgmma":
            return self._lower_await_wgmma(op)
        if kind == "bulk":
            return self._lower_await_bulk(op)
        raise NotImplementedError(f"lower_await: unsupported lock kind {kind!r}")

    @lowers(BarrierOp)
    def lower_barrier(
        self,
        op: BarrierOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower a cross-thread ordering fence."""
        scope = op.attrs.get("scope")
        operands: list[Value] = []
        operand_names: list[str] = []
        if scope == "thread":
            return [], []  # per-thread ordering is implicit in program order
        if scope == "block":
            template = (f"bar.sync {op.attrs.get('id', 15)}, {32 * len(execution_warps(op))};\n"
                        if execution_warps(op) is not None else "bar.sync 0;\n")
        elif scope == "cluster_to_leader" and op.attrs.get("phase") == "arrive":
            # Each warp arrives once at every cluster CTA's mbarrier and
            # continues.
            assert op.sync is not None
            kernel = find_ancestor_of_type(op, KernelOp)
            assert kernel is not None
            cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
            template = (
                _elect_ptx("release")
                + "".join(
                    textwrap.dedent(f"""\
                        mov.u32 ${{b32:rank}}, {rank};
                        mapa.shared::cluster.u32 ${{b32:remote}}, ${{b32:operand.sync.val}}, ${{b32:rank}};
                        @${{pred:release}} mbarrier.arrive.shared::cluster.b64 _, [${{b32:remote}}];
                        """)
                    for rank in range(cluster_size)
                )
            )
            masks = op.sync.type.attrs.get('release_cta_masks')
            if masks is not None:
                prefix = ("mov.u32 ${b32:release_source}, %cluster_ctarank;\n"
                          + rank_table_ptx('release_mask', masks, rank='release_source'))
                arrivals = ""
                for rank in range(cluster_size):
                    arrivals += (
                        f"and.b32 ${{b32:release_bit}}, ${{b32:release_mask}}, {1 << rank};\n"
                        "setp.ne.u32 ${pred:release_target}, ${b32:release_bit}, 0;\n"
                        "and.pred ${pred:release_target}, ${pred:release_target}, ${pred:release};\n"
                        f"mov.u32 ${{b32:rank}}, {rank};\n"
                        "mapa.shared::cluster.u32 ${b32:remote}, ${b32:operand.sync.val}, ${b32:rank};\n"
                        "@${pred:release_target} mbarrier.arrive.shared::cluster.b64 _, [${b32:remote}];\n")
                template = _elect_ptx("release") + prefix + arrivals
            operands = [cast(Value, self.value_map[op.sync])]
            operand_names = ["sync"]
        elif scope == "cluster_to_leader" and op.attrs.get("phase") == "wait":
            # Only the leader's warp, which issues the next write, waits for
            # every warp of the cluster.
            assert op.sync is not None
            template = _leader_warp("cl", execution_issuer_warp(op)) + _CLUSTER_LEADER_WAIT
            operands = [cast(Value, self.value_map[op.sync])]
            operand_names = ["sync"]
        elif scope == "cluster_to_leader":
            # Every warp releases; only the leaders wait, so the other warps
            # are not held back by the slowest CTA.
            assert op.sync is not None
            template = _CLUSTER_RELEASE_ARRIVAL + _CLUSTER_RELEASE_LEADER_WAIT
            operands = [cast(Value, self.value_map[op.sync])]
            operand_names = ["sync"]
        elif scope in ("leader_to_cluster", "cluster"):
            # A release arrive waits for every outstanding write, including
            # the last TMA stores. A barrier that only precedes the kernel's
            # final deallocations, or that orders only reads before writes
            # (insert_syncs marks it ``publishes: False``), publishes no data.
            block = op.parent_block
            retires = (block is not None and isinstance(block.parent_op, KernelOp)
                       and all(isinstance(later, DeallocOp)
                               for later in block.ops[block.ops.index(op) + 1:]))
            arrive = (
                "relaxed" if (op.attrs.get("initialization_publication") or retires
                              or op.attrs.get("publishes") is False) else "release"
            )
            template = textwrap.dedent(f"""\
                barrier.cluster.arrive.{arrive};
                barrier.cluster.wait.acquire;
                """)
        elif scope == "block_leader" and op.sync is None:
            # Warp 0, which holds the leader t0, waits; the other warps arrive
            # and continue, on the named barrier id chosen by insert_syncs
            # (unique among concurrently-live block-leader barriers). Warps
            # are told apart by their warp-uniform index, and arrive first so
            # an arrival never waits behind a skipped bar.sync.
            bar_id = op.attrs.get("id", 1)
            num_threads = op.attrs["num_threads"]
            template = _leader_warp("bl", execution_issuer_warp(op)) + textwrap.dedent(f"""\
                @${{pred:bl_other_warp}} bar.arrive {bar_id}, {num_threads};
                @!${{pred:bl_other_warp}} bar.sync {bar_id}, {num_threads};
                """)
        elif scope == "block_leader":
            assert op.sync is not None
            operands = [cast(Value, self.value_map[op.sync])]
            operand_names = ["sync"]
            arrival = (_elect_ptx("bl_arrive") + "@${pred:bl_arrive} mbarrier.arrive.release.cta"
                       ".shared::cta.b64 _, [${b32:operand.sync.val}];\n")
            wait = ("@${pred:bl_other_warp} bra ${label:bl_done};\n" + _try_wait_ptx("sync", prefix="bl_")
                    + "${label:bl_done}:\n"
                    "// All actors advance: a later handoff may elect another warp.\n"
                    "xor.b32 ${b32:operand.sync.parity}, ${b32:operand.sync.parity}, 1;\n")
            phase = op.attrs.get('phase')
            template = arrival if phase == 'arrive' else (
                _leader_warp("bl", execution_issuer_warp(op))
                + (arrival if phase is None else '') + wait)
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
            if tuple(d or 1 for d in ancestor.block_dim[1:]) != (1, 1):
                # A grid ticket belongs to exactly one physical CTA thread.
                template = template.replace(
                    "mov.u32 ${b32:grid_tid}, %tid.x;\n",
                    "mov.u32 ${b32:grid_tid}, %tid.x;\n"
                    "mov.u32 ${b32:grid_yz}, %tid.y;\n"
                    "or.b32 ${b32:grid_tid}, ${b32:grid_tid}, ${b32:grid_yz};\n"
                    "mov.u32 ${b32:grid_yz}, %tid.z;\n"
                    "or.b32 ${b32:grid_tid}, ${b32:grid_tid}, ${b32:grid_yz};\n",
                )
            operands = [sync]
            operand_names = ["sync"]
        else:
            raise NotImplementedError(
                f"lower_barrier: unsupported barrier scope {scope!r}"
            )
        if self.distinct_issuers:
            # tcgen05 operations require explicit code-motion ordering when
            # thread synchronization hands an accumulator between warps.
            template = ('tcgen05.fence::before_thread_sync;\n' + template
                        + 'tcgen05.fence::after_thread_sync;\n')
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
        self,
        op: AwaitOp,
        lock: Value,
    ) -> tuple[list[Op], list[object]]:
        paired_load = op.lock.type.kind == "mbar" and op.lock.type.attrs.get("cta_group", 1) == 2
        # Completed only by async-proxy transactions or local release.cta
        # arrivals, so a CTA-scope acquire suffices (as in CUTLASS); .cluster
        # would make ptxas invalidate L1 after every wait.
        template = "\n" + _try_wait_ptx("lock")
        operands, operand_names = [lock], ["lock"]
        probe = self._probe_results.pop(id(op), None)
        if probe is not None:
            template = ("@${pred:operand.probe.val} bra ${label:probe_done};\n" + template
                        + "${label:probe_done}:\n")
            operands.append(probe)
            operand_names.append("probe")
        if op.waiter_warps is not None:
            guard = (self._region_guard("await", op.waiter_warps[0]) if len(op.waiter_warps) == 1
                     else None) or ([], [], _leader_warp("await", *op.waiter_warps))
            operands.extend(guard[0])
            operand_names.extend(guard[1])
            template = guard[2] + "@${pred:await_other_warp} bra ${label:await_done};\n" + template + "${label:await_done}:\n"
        if paired_load:
            template = (_pair_leader_ptx("await_rank", "await_peer", peer=True)
                        + "@${pred:await_peer} bra ${label:paired_await_done};\n"
                        + template + "${label:paired_await_done}:\n")
        if not isinstance(op.lock.definition, LockViewOp):
            template += "xor.b32 ${b32:operand.lock.parity}, ${b32:operand.lock.parity}, 1;\n"
        if self.distinct_issuers and op.lock.type.kind == 'tcgen05':
            template += 'tcgen05.fence::after_thread_sync;\n'
        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_await_wgmma(
        self,
        op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        # wait_group drains the commit group so the warpgroup's wgmma reads of
        # the smem operands have retired. The buffer-reuse WAR (thread-0's TMA
        # reload vs those reads) is handled by insert_syncs, which emits a barrier
        # before the reload — so no barrier is needed here.
        pending = op.pending_groups
        template = f"wgmma.wait_group.sync.aligned {pending};\n"
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
        self,
        op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        read = ".read" if op.bulk_read_only else ""
        template = textwrap.dedent(f"""
            cp.async.bulk.wait_group{read} {op.pending_groups};
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
        self,
        op: MoveOp,
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
            BF16Type | F16Type | F32Type | F64Type,
            gmem_ir_type.dtype,
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
            offset = sum(c * s for c, s in zip(idx, strides)) * dtype_bytes
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
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_gmem_smem(
        self,
        op: MoveOp,
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
            BF16Type | F16Type | F32Type | F64Type,
            gmem_type.dtype,
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
            gmem_seq_stride_bytes = cast(int, gmem_type.strides[seq_dim]) * dtype_bytes
            smem_seq_stride_bytes = cast(int, smem_type.strides[seq_dim]) * dtype_bytes
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
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_wgrmem_mma_to_smem(
        self,
        op: MoveOp,
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

        return _lowered(op, template, operands, operand_names)

    def _lower_move_wgrmem_wgmma(
        self,
        op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        register_starts = _stmatrix_register_starts(op)
        if register_starts is None:
            return self._lower_move_wgrmem_wgmma_addrmem(op)
        return self._lower_move_wgrmem_wgmma_to_stmatrix(op, register_starts)

    def _lower_move_wgrmem_wgmma_addrmem(
        self,
        op: MoveOp,
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
        warp_rows, _ = _m16n8_fragment_rows(root_type)
        root_n = _m16n8_fragment_columns(root_type)
        pairs = _m16n8_move_pairs(op)
        paired_fragment = _paired_fragment(root_type)
        row_partitions = 2 if paired_fragment else 4
        assert m_dim % (row_partitions * warp_rows) == 0, (
            f"fragment store: m must be a multiple of {row_partitions * warp_rows}; got {m_dim}"
        )
        assert root_n % 8 == 0 and 8 <= root_n <= 256, (
            f"wgmma store: n must be a multiple of 8 in [8, 256]; got {root_n}"
        )
        assert isinstance(memory_type.dtype, (BF16Type, F16Type, F32Type, F64Type))
        kernel = find_ancestor_of_type(op, KernelOp)
        if (isinstance(memory_type, GmemTensorType)
                and isinstance(memory_type.dtype, (BF16Type, F16Type))
                and kernel is not None and kernel.attrs.get("masked_global_store")):
            return self._lower_masked_half_fragment_move(
                op, fragment, memory, pairs, warp_rows
            )
        dtype_bytes = memory_type.dtype.byte_size
        row_stride_bytes = cast(int, memory_type.strides[0]) * dtype_bytes
        column_stride_bytes = cast(int, memory_type.strides[1]) * dtype_bytes
        column_broadcast = is_load and column_stride_bytes == 0
        strided_f32 = isinstance(memory_type.dtype, F32Type) and row_stride_bytes == dtype_bytes
        assert column_broadcast or strided_f32 or (
            column_stride_bytes == dtype_bytes
        ), "wgmma element pairs must be contiguous"
        assert not (
            isinstance(memory_type, SmemTensorType)
            and physical_layout(memory_type) is not None
        ), f"wgmma move: strided addressing cannot store {memory_type}"

        new_ops, byte_offset = _build_wgmma_layout_0_dynamic_offset(
            row_stride_bytes,
            column_stride_bytes,
            warp_rows=warp_rows,
            warp_group_rows=_m16n8_fragment_group_rows(root_type),
            warp_group_columns=_m16n8_fragment_group_columns(root_type),
            paired_columns=root_n if paired_fragment else 0,
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
        # Column-warpgroup gathers reuse one staging buffer across slices;
        # honor the same writer ownership as the STMatrix store path.
        writer_wg = op.attrs.get("writer_wg") if not is_load else None
        predicate = ""
        if writer_wg is not None:
            template += textwrap.dedent(f"""\
                mov.u32 ${{b32:store_tid}}, %tid.x;
                shr.u32 ${{b32:store_wg}}, ${{b32:store_tid}}, 7;
                setp.eq.u32 ${{pred:store_writer}}, ${{b32:store_wg}}, {writer_wg};
            """)
            predicate = "@${pred:store_writer} "
        # A pair is one packed BF16/F16 register or two F32 registers.
        f32 = isinstance(wgrmem_type.dtype, F32Type)
        instruction = f"{'ld' if is_load else 'st'}.{space}{'.v2' if f32 else ''}.b32"
        pair_bytes = 8 if f32 else 4
        scalar_pairs = strided_f32 and column_stride_bytes != dtype_bytes
        known_alignment = 0
        if isinstance(memory_type, SmemTensorType):
            known_alignment = _row_fragment_shared_alignment(memory)
            if known_alignment and known_alignment % dtype_bytes:
                raise ValueError("fragment shared-memory view violates element alignment")
            if known_alignment:
                # The dynamic lane address includes an arbitrary row and a
                # column pair; static fragment offsets use those same strides.
                known_alignment = math.gcd(known_alignment, row_stride_bytes,
                                          2 * column_stride_bytes,
                                          *(di * row_stride_bytes + dj * column_stride_bytes
                                            for _, di, dj in pairs))
                scalar_pairs |= known_alignment < pair_bytes
        guarded_pairs = (not column_broadcast and not scalar_pairs
                         and dtype_bytes in (2, 4) and not known_alignment)
        if guarded_pairs:
            # Typed global pointers promise element alignment, not pair
            # alignment. Odd row pitches can also differ between lanes, so
            # these ordinary copies use a potentially divergent branch and
            # reconverge before subsequent collective work.
            address_bits = 64 if space == "global" else 32
            template += textwrap.dedent(f"""\
                and.b{address_bits} ${{b{address_bits}:alignment}}, {address}, {pair_bytes - 1};
                setp.eq.u{address_bits} ${{pred:pair_aligned}}, ${{b{address_bits}:alignment}}, 0;
                @!${{pred:pair_aligned}} bra ${{label:unaligned_pair}};
            """)

        # Broadcast columns load one scalar per distinct row and duplicate its
        # bits into the fragment pair. Other columns reuse that register value.
        # This reads the original strided view; it does not materialize a bias.
        loaded_rows: dict[int, str] = {}
        for pair, di, dj in pairs:
            offset = di * row_stride_bytes + dj * column_stride_bytes
            if column_broadcast:
                first = (
                    f"${{f32:operand.fragment.r{2 * pair}}}"
                    if f32
                    else f"${{b32:operand.fragment.r{pair}}}"
                )
                if offset in loaded_rows:
                    template += f"mov.b32 {first}, {loaded_rows[offset]};\n"
                elif f32:
                    template += f"ld.{space}.b32 {first}, [{address}+{offset}];\n"
                else:
                    scalar = f"${{b16:broadcast{pair}}}"
                    template += f"ld.{space}.b16 {scalar}, [{address}+{offset}];\n"
                    template += f"mov.b32 {first}, {{{scalar}, {scalar}}};\n"
                loaded_rows.setdefault(offset, first)
                if f32:
                    second = f"${{f32:operand.fragment.r{2 * pair + 1}}}"
                    template += f"mov.b32 {second}, {first};\n"
                continue
            if scalar_pairs and f32:
                # Column-major FP32 workspaces retain the logical fragment
                # ownership; a column pair becomes two scalar accesses.
                for element in range(2):
                    register = f"${{f32:operand.fragment.r{2 * pair + element}}}"
                    location = f"[{address}+{offset + element * column_stride_bytes}]"
                    template += (
                        f"ld.{space}.b32 {register}, {location};\n" if is_load
                        else f"{predicate}st.{space}.b32 {location}, {register};\n"
                    )
                continue
            if scalar_pairs and dtype_bytes == 2:
                register = f"${{b32:operand.fragment.r{pair}}}"
                lo, hi = f"${{b16:half_lo{pair}}}", f"${{b16:half_hi{pair}}}"
                if is_load:
                    template += f"ld.{space}.b16 {lo}, [{address}+{offset}];\n"
                    template += f"ld.{space}.b16 {hi}, [{address}+{offset + 2}];\n"
                    template += f"mov.b32 {register}, {{{lo}, {hi}}};\n"
                else:
                    template += f"mov.b32 {{{lo}, {hi}}}, {register};\n"
                    template += f"{predicate}st.{space}.b16 [{address}+{offset}], {lo};\n"
                    template += f"{predicate}st.{space}.b16 [{address}+{offset + 2}], {hi};\n"
                continue
            registers = (
                f"{{${{f32:operand.fragment.r{2 * pair}}}, "
                f"${{f32:operand.fragment.r{2 * pair + 1}}}}}"
                if f32
                else f"${{b32:operand.fragment.r{pair}}}"
            )
            location = f"[{address}+{offset}]"
            template += (
                f"{instruction} {registers}, {location};\n"
                if is_load
                else f"{predicate}{instruction} {location}, {registers};\n"
            )

        if guarded_pairs:
            template += "bra ${label:pair_copy_done};\n${label:unaligned_pair}:\n"
            for pair, di, dj in pairs:
                offset = di * row_stride_bytes + dj * column_stride_bytes
                if f32:
                    for element in range(2):
                        register = f"${{f32:operand.fragment.r{2 * pair + element}}}"
                        location = f"[{address}+{offset + element * column_stride_bytes}]"
                        template += (f"ld.{space}.b32 {register}, {location};\n" if is_load
                                     else f"{predicate}st.{space}.b32 {location}, {register};\n")
                    continue
                lo, hi = f"${{b16:half_lo{pair}}}", f"${{b16:half_hi{pair}}}"
                register = f"${{b32:operand.fragment.r{pair}}}"
                if is_load:
                    template += f"ld.{space}.b16 {lo}, [{address}+{offset}];\n"
                    template += f"ld.{space}.b16 {hi}, [{address}+{offset + 2}];\n"
                    template += f"mov.b32 {register}, {{{lo}, {hi}}};\n"
                else:
                    template += f"mov.b32 {{{lo}, {hi}}}, {register};\n"
                    template += f"{predicate}st.{space}.b16 [{address}+{offset}], {lo};\n"
                    template += f"{predicate}st.{space}.b16 [{address}+{offset + 2}], {hi};\n"
            template += "${label:pair_copy_done}:\n"

        lowered, _ = _lowered(op, template, [cast(Value, self.value_map[memory]),
                                             cast(Value, self.value_map[fragment]), byte_offset],
                              ["memory", "fragment", "offset"])
        return new_ops + lowered, []

    def _lower_masked_half_fragment_move(self, op, fragment, memory, pairs, warp_rows):
        """Lower fragment addressing, then emit bounded or complete-tile accesses."""
        from .lower_to_tma_move import _build_tile_chain_indices

        memory_type = memory.type
        assert isinstance(memory_type, GmemTensorType)
        assert memory_type.rank == 2 and memory_type.strides[1] == 1
        index_ops, root, origins, _ = _build_tile_chain_indices(
            memory, list(memory_type.sizes),
        )
        assert root.type.rank == 2
        row_stride = int(memory_type.strides[0]) * 2
        offset_ops, byte_offset, lane_row, lane_col = (
            _build_wgmma_layout_0_dynamic_offset(
                row_stride, 2, coordinates=True, warp_rows=warp_rows,
                warp_group_rows=_m16n8_fragment_group_rows(cast(WgrmemTensorType, fragment.type)),
                warp_group_columns=_m16n8_fragment_group_columns(cast(WgrmemTensorType, fragment.type)),
                paired_columns=_m16n8_fragment_columns(cast(WgrmemTensorType, fragment.type)) if _paired_fragment(cast(WgrmemTensorType, fragment.type)) else 0,
            )
        )
        is_load = isinstance(op.dst.type, WgrmemTensorType)
        instruction = "ld" if is_load else "st"
        template = textwrap.dedent(r"""
            cvt.u64.u32 ${b64:off64}, ${b32:operand.offset.val};
            add.u64 ${b64:base}, ${b64:operand.memory.ptr}, ${b64:off64};
            add.u32 ${b32:row}, ${b32:operand.origin_row.val}, ${b32:operand.lane_row.val};
            add.u32 ${b32:col}, ${b32:operand.origin_col.val}, ${b32:operand.lane_col.val};
            and.b64 ${b64:alignment}, ${b64:base}, 3;
            setp.eq.u64 ${pred:aligned}, ${b64:alignment}, 0;
        """)
        writer_wg = op.attrs.get("writer_wg") if not is_load else None
        if writer_wg is not None:
            template += textwrap.dedent(f"""
                mov.u32 ${{b32:writer_tid}}, %tid.x;
                shr.u32 ${{b32:writer_wg}}, ${{b32:writer_tid}}, 7;
                setp.eq.u32 ${{pred:writer}}, ${{b32:writer_wg}}, {writer_wg};
                @!${{pred:writer}} bra ${{label:fragment_end}};
            """)
        template += textwrap.dedent(f"""
            add.u32 ${{b32:tile_row_end}}, ${{b32:operand.origin_row.val}}, {memory_type.sizes[0]};
            add.u32 ${{b32:tile_col_end}}, ${{b32:operand.origin_col.val}}, {memory_type.sizes[1]};
            setp.le.u32 ${{pred:complete_rows}}, ${{b32:tile_row_end}}, {root.type.sizes[0]};
            setp.le.u32 ${{pred:complete_cols}}, ${{b32:tile_col_end}}, {root.type.sizes[1]};
            and.pred ${{pred:complete}}, ${{pred:complete_rows}}, ${{pred:complete_cols}};
            and.pred ${{pred:complete_wide}}, ${{pred:complete}}, ${{pred:aligned}};
        """)
        if not is_load:
            # The shuffle participants must be recorded before alignment paths diverge.
            template += textwrap.dedent(r"""
                not.pred ${pred:unaligned}, ${pred:aligned};
                and.pred ${pred:narrow_members}, ${pred:complete}, ${pred:unaligned};
                activemask.b32 ${b32:all_members};
                vote.ballot.sync.b32 ${b32:narrow_mask}, ${pred:narrow_members}, ${b32:all_members};
            """)
        template += "@${pred:complete_wide} bra ${label:fragment_wide};\n"
        template += "@${pred:complete} bra ${label:fragment_narrow};\n"
        for path in ("edge", "wide", "narrow"):
            if path == "narrow" and not is_load:
                template += _ptx_store_unaligned_wgmma_half_pairs(
                    pairs, row_stride, memory_type.sizes[1],
                )
                break
            loaded_offsets = {}
            for pair, di, dj in pairs:
                offset = di * row_stride + dj * 2
                reg = f"${{b32:operand.fragment.r{pair}}}"
                if is_load and row_stride == 0 and offset in loaded_offsets:
                    template += f"mov.b32 {reg}, {loaded_offsets[offset]};\n"
                    continue
                if path == "edge":
                    template += _ptx_masked_half_pair(
                        pair, di, dj, row_stride, root.type.sizes,
                        is_load=is_load, writer=writer_wg is not None,
                    )
                elif path == "wide":
                    address = f"[${{b64:base}}+{offset}]"
                    operands = f"{reg}, {address}" if is_load else f"{address}, {reg}"
                    template += f"{instruction}.global.b32 {operands};\n"
                else:
                    # Unaligned stores were handled above; only scalar loads remain.
                    lo, hi = f"${{b16:fast_lo{pair}}}", f"${{b16:fast_hi{pair}}}"
                    template += f"ld.global.b16 {lo}, [${{b64:base}}+{offset}];\n"
                    template += f"ld.global.b16 {hi}, [${{b64:base}}+{offset+2}];\n"
                    template += f"mov.b32 {reg}, {{{lo}, {hi}}};\n"
                if is_load and row_stride == 0:
                    loaded_offsets[offset] = reg
            next_label = {"edge": "fragment_wide", "wide": "fragment_narrow"}.get(path)
            if next_label:
                template += f"bra ${{label:fragment_end}};\n${{label:{next_label}}}:\n"
        template += "${label:fragment_end}:\n"
        lowered, _ = _lowered(op, template, [cast(Value, self.value_map[memory]), cast(Value, self.value_map[fragment]),
                                             byte_offset, origins[0], origins[1], lane_row, lane_col],
                              ["memory", "fragment", "offset", "origin_row", "origin_col", "lane_row", "lane_col"])
        return index_ops + offset_ops + lowered, []

    def _lower_move_wgrmem_wgmma_to_stmatrix(
        self,
        op: MoveOp,
        register_starts: list[int],
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
        dst_root = _layout_view_root(op.dst)
        assert src_root is not None and dst_root is not None
        src_root_type = src_root.type
        assert isinstance(src_root_type, WgrmemTensorType)
        paired_dynamic = _paired_fragment(src_root_type)
        n_dim = _m16n8_fragment_columns(src_root_type)
        warp_rows, tile_rows = _m16n8_fragment_rows(src_root_type)
        root_type = dst_root.type
        assert isinstance(root_type, SmemTensorType)
        root_sizes = cast(tuple[int, ...], root_type.sizes)
        transposed = is_column_major(root_type)
        layout = physical_layout(dst_type)
        assert layout is not None
        block_width = layout[0]
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
        # Lane l addresses row l % 8 of matrix l / 8. Untransposed, the x4
        # matrices sit side by side along the fragment row. Transposed, they
        # are the 2x2 matrices of a 16x16 fragment block, and a stored row is
        # a fragment column: lane l writes physical row
        # (l & 7) + 8 * (l >> 4) at column warp_rows * warp + 8 * ((l >> 3) & 1).
        lane_coordinates = (
            textwrap.dedent(f"""\
            shr.u32 ${{b32:st_hi}}, ${{b32:st_lane}}, 4;
            mad.lo.u32 ${{b32:st_row}}, ${{b32:st_hi}}, 8, ${{b32:st_row}};
            shr.u32 ${{b32:st_col}}, ${{b32:st_lane}}, 3;
            and.b32 ${{b32:st_col}}, ${{b32:st_col}}, 1;
            shl.b32 ${{b32:st_col}}, ${{b32:st_col}}, 3;
            mad.lo.u32 ${{b32:st_col}}, ${{b32:st_warp}}, {warp_rows}, ${{b32:st_col}};
        """)
            if transposed
            else textwrap.dedent(f"""\
            mad.lo.u32 ${{b32:st_row}}, ${{b32:st_warp}}, {warp_rows}, ${{b32:st_row}};
            shr.u32 ${{b32:st_col}}, ${{b32:st_lane}}, 3;
            shl.b32 ${{b32:st_col}}, ${{b32:st_col}}, 3;
        """)
        )
        template = (
            writer_setup
            + textwrap.dedent("""\
            mov.u32 ${b32:st_lane}, %laneid;
            mov.u32 ${b32:st_tid}, %tid.x;
            and.b32 ${b32:st_tid}, ${b32:st_tid}, 127;
            shr.u32 ${b32:st_warp}, ${b32:st_tid}, 5;
            and.b32 ${b32:st_row}, ${b32:st_lane}, 7;
        """)
            + ("shr.u32 ${b32:st_pair_col}, ${b32:st_warp}, 1;\nand.b32 ${b32:st_warp}, ${b32:st_warp}, 1;\n" if paired_dynamic else "")
            + lane_coordinates
        )
        group_rows = _m16n8_fragment_group_rows(src_root_type)
        if group_rows:
            coordinate = "st_col" if transposed else "st_row"
            template += textwrap.dedent(f"""\
                mov.u32 ${{b32:st_group}}, %tid.x;
                shr.u32 ${{b32:st_group}}, ${{b32:st_group}}, 7;
                mad.lo.u32 ${{b32:{coordinate}}}, ${{b32:st_group}}, {group_rows}, ${{b32:{coordinate}}};
            """)
        # A lane's matrix starts up to 24 columns into the 32 an x4 covers;
        # transposed, its column is the warp's fragment row. A block narrower
        # than that reach puts the matrix in a later block.
        if not paired_dynamic and not _m16n8_fragment_group_columns(src_root_type) and block_width <= (3 * warp_rows + 8 if transposed else 24):
            physical_rows = root_sizes[1] if transposed else root_sizes[0]
            template += textwrap.dedent(f"""\
                shr.u32 ${{b32:st_block}}, ${{b32:st_col}}, {block_width.bit_length() - 1};
                and.b32 ${{b32:st_col}}, ${{b32:st_col}}, {block_width - 1};
                mad.lo.u32 ${{b32:st_row}}, ${{b32:st_block}}, {physical_rows}, ${{b32:st_row}};
            """)
        template += textwrap.dedent(f"""\
            mad.lo.u32 ${{b32:st_off}}, ${{b32:st_row}}, {block_width}, ${{b32:st_col}};
            shl.b32 ${{b32:st_off}}, ${{b32:st_off}}, 1;
        """)
        groups = (
            _stmatrix_transposed_groups(
                register_starts, n_dim, col_start, root_sizes[1], tile_rows
            )
            if transposed
            else _stmatrix_groups(
                register_starts,
                {pair for pair, _, _ in _stmatrix_pairs(op)},
                n_dim,
                col_start,
                root_sizes[0],
                block_width,
                tile_rows,
            )
        )
        group_columns = src_root_type.sizes[1] // 2 if paired_dynamic else _m16n8_fragment_group_columns(src_root_type)
        if paired_dynamic:
            template += "mov.u32 ${b32:st_column_group}, ${b32:st_pair_col};\n"
        elif group_columns:
            template += "mov.u32 ${b32:st_column_group}, %tid.x;\nshr.u32 ${b32:st_column_group}, ${b32:st_column_group}, 7;\n"
        for registers, physical_offset in groups:
            regs = ", ".join(f"${{b32:operand.src.r{i}}}" for i in registers)
            template += (
                f"add.u32 ${{b32:st_addr_off}}, ${{b32:st_off}}, {physical_offset};\n"
                + _swizzle_ptx("st_addr_off", "st_swizzle_mask", 2 * block_width)
                + "add.u32 ${b32:st_addr}, ${b32:operand.dst.ptr}, ${b32:st_addr_off};\n"
                if block_width > 8
                else textwrap.dedent(f"""\
                add.u32 ${{b32:st_addr}}, ${{b32:operand.dst.ptr}},
                    ${{b32:st_off}};
                add.u32 ${{b32:st_addr}}, ${{b32:st_addr}}, {physical_offset};
            """)
            )
            if group_columns:
                di, dj = _wgmma_pair_static_offset(registers[0], n_dim, tile_rows)
                dj -= col_start
                static_row, static_col = (dj, di) if transposed else (di, dj)
                coordinate = "st_access_row" if transposed else "st_access_col"
                # Compute the complete blocked address after adding both the
                # static register offset and the warp's column ownership.
                template += textwrap.dedent(f"""\
                    add.u32 ${{b32:st_access_row}}, ${{b32:st_row}}, {static_row};
                    add.u32 ${{b32:st_access_col}}, ${{b32:st_col}}, {static_col};
                    mad.lo.u32 ${{b32:{coordinate}}}, ${{b32:st_column_group}}, {group_columns}, ${{b32:{coordinate}}};
                    div.u32 ${{b32:st_access_block}}, ${{b32:st_access_col}}, {block_width};
                    rem.u32 ${{b32:st_access_inner}}, ${{b32:st_access_col}}, {block_width};
                    mad.lo.u32 ${{b32:st_access_row}}, ${{b32:st_access_block}}, {root_sizes[1] if transposed else root_sizes[0]}, ${{b32:st_access_row}};
                    mad.lo.u32 ${{b32:st_addr_off}}, ${{b32:st_access_row}}, {block_width}, ${{b32:st_access_inner}};
                    shl.b32 ${{b32:st_addr_off}}, ${{b32:st_addr_off}}, 1;
                """)
                template += _swizzle_ptx("st_addr_off", "st_swizzle_mask", 2 * block_width) if block_width > 8 else ""
                template += "add.u32 ${b32:st_addr}, ${b32:operand.dst.ptr}, ${b32:st_addr_off};\n"
            shape = f"x{len(registers)}" + (".trans" if transposed else "")
            template += (
                f"{predicate}stmatrix.sync.aligned.m8n8.{shape}.shared::cta.b16 "
                f"[${{b32:st_addr}}], {{{regs}}};\n"
            )
        return _lowered(op, template, [dst, src], ["dst", "src"])

    def _lower_move_smem_to_wgrmem_mma(
        self,
        op: MoveOp,
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

        reg_list = ", ".join(f"${{b32:operand.dst.r{i}}}" for i in range(n_regs))

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

        return _lowered(op, template, operands, operand_names)

    def _lower_move_smem_rmem(
        self,
        op: MoveOp,
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
            BF16Type | F16Type | F32Type | F64Type,
            smem_ir_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = f"${{b32:operand.{smem_label}.ptr}}"
        vectorize = op.attrs.get("vectorize_scalar_shared_loads", False)
        if type(vectorize) is not bool:
            raise ValueError("vectorize_scalar_shared_loads must be a boolean")
        vectorize = (vectorize and is_load and isinstance(rmem_type.dtype, F32Type)
                     and _scalar_shared_vector_aligned(op.src))
        swizzle = smem_ir_type.swizzle
        shared_view = op.src if is_load else op.dst
        root = _layout_view_root(shared_view)
        layout = physical_layout(cast(SmemTensorType, root.type))
        blocked = layout is not None and layout[1] == "block_major"
        dynamic_inner = False
        if blocked:
            assert isinstance(root.type, SmemTensorType) and root.type.rank == 2
            block_width = layout[0]
            inner_dim = 0 if is_column_major(root.type) else 1
            root_rows = cast(int, root.type.sizes[1 - inner_dim])
            dynamic_inner = rmem_type.sizes[inner_dim] > 1 and not _block_aligned_view(
                shared_view, block_width, inner_dim
            )
        if swizzle is not None or dynamic_inner:
            template = (
                "{\n.reg .b32 scalar_addr, scalar_xor, scalar_inner, scalar_blocks;\n"
            )
        if swizzle is not None:
            if swizzle not in ("32B", "64B", "128B"):
                raise NotImplementedError(f"scalar shared move swizzle: {swizzle}")
        if dynamic_inner:
            operands.append(cast(Value, self.value_map[root]))
            operand_names.append("root")
            template += (
                f"sub.u32 scalar_inner, {base_reg}, ${{b32:operand.root.ptr}};\n"
                + _constant_divmod_ptx(
                    "rem", "scalar_inner", "scalar_inner", block_width * dtype_bytes
                )
            )

        skipped = 0
        for flat_idx, idx in enumerate(
            itertools.product(*[range(s) for s in cast(tuple[int, ...], rmem_type.sizes)])
        ):
            if skipped:
                skipped -= 1
                continue
            width = (4 if vectorize and idx[-1] % 4 == 0
                     and idx[-1] + 4 <= rmem_type.sizes[-1] else 1)
            skipped = width - 1
            strides = cast(tuple[int, ...], smem_ir_type.strides)
            offset = sum(c * s for c, s in zip(idx, strides)) * dtype_bytes
            if blocked:
                row, column = idx[1 - inner_dim], idx[inner_dim]
                offset = (
                    _block_major_offset(row, column, block_width, root_rows) * dtype_bytes
                )
            reg = f"${{{suffix}:operand.{'dst' if is_load else 'src'}.r{flat_idx}}}"
            address = f"{base_reg}+{offset}"
            if dynamic_inner:
                # Column increments can cross a block from a sub-block origin.
                # Crossing skips the other rows of the old physical block.
                offset = (row * block_width + column) * dtype_bytes
                template += (
                    f"add.u32 scalar_blocks, scalar_inner, {column * dtype_bytes};\n"
                    + _constant_divmod_ptx(
                        "div", "scalar_blocks", "scalar_blocks", block_width * dtype_bytes
                    )
                    + f"mad.lo.u32 scalar_addr, scalar_blocks, {(root_rows - 1) * block_width * dtype_bytes}, {base_reg};\n"
                    f"add.u32 scalar_addr, scalar_addr, {offset};\n"
                )
                address = "scalar_addr"
            if swizzle is not None:
                # Tile pointers retain unswizzled offsets. TMA's swizzle
                # uses the shared address bits, including the allocation's
                # position within its 1 KiB swizzle span.
                mask = (int(swizzle[:-1]) // 16 - 1) << 7
                if not dynamic_inner:
                    template += f"add.u32 scalar_addr, {base_reg}, {offset};\n"
                template += (
                    f"and.b32 scalar_xor, scalar_addr, {mask};\n"
                    "shr.u32 scalar_xor, scalar_xor, 3;\n"
                    "xor.b32 scalar_addr, scalar_addr, scalar_xor;\n"
                )
                address = "scalar_addr"
            if is_load:
                if width == 4:
                    registers = ", ".join(
                        f"${{{suffix}:operand.dst.r{flat_idx + i}}}" for i in range(width))
                    template += f"ld.shared.v4.{suffix} {{{registers}}}, [{address}];\n"
                else:
                    template += f"ld.shared.{suffix} {reg}, [{address}];\n"
            else:
                template += f"st.shared.{suffix} [{address}], {reg};\n"

        if swizzle is not None or dynamic_inner:
            template += "}\n"

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_async_tma_stores(
        self, moves: list[TmaMoveOp], conditions: list[Value | None] | None = None
    ) -> tuple[list[Op], list[object]]:
        """Issue optional atom stores from one elected thread and commit one group."""
        issuer = execution_issuer_warp(moves[0])
        if any(execution_issuer_warp(move) != issuer for move in moves):
            raise ValueError('TMA store group requires one issuing warp')
        leader_operands, leader_names, _ = self._leader_thread("st", issuer_warp=issuer)
        operands = list(leader_operands)
        operand_names = list(leader_names)
        lines = ["fence.proxy.async.shared::cta;"]
        for i, op in enumerate(moves):
            dst_value = cast(Value, self.value_map[op.dst])
            src_value = cast(Value, self.value_map[op.src])
            indices = [cast(Value, self.value_map[v]) for v in op.indices]
            desc_type = dst_value.type
            assert isinstance(desc_type, TmaDescType)
            dn, sn = f"dst{i}", f"src{i}"
            names = [f"idx{i}_{j}" for j in range(len(indices))]
            operands += [dst_value, src_value, *indices]
            operand_names += [dn, sn, *names]
            coordinate_lines, coordinates = _tma_coordinates(
                desc_type, names, f"store{i}"
            )
            instruction = self._tma_instruction(
                op,
                f"cp.async.bulk.tensor.{len(coordinates)}d.global.shared::cta.bulk_group",
                f"[${{b64:operand.{dn}.ptr}}, {{{', '.join(coordinates)}}}], [${{b32:operand.{sn}.ptr}}]",
                operands,
                operand_names,
                f"cache_policy{i}",
            )
            predicate = "${pred:st_leader}"
            condition = conditions[i] if conditions is not None else None
            if condition is not None:
                name = f"condition{i}"
                operands.append(cast(Value, self.value_map[condition]))
                operand_names.append(name)
                lines.append(
                    f"and.pred ${{pred:store{i}_enabled}}, ${{pred:st_leader}}, "
                    f"${{pred:operand.{name}.val}};"
                )
                predicate = f"${{pred:store{i}_enabled}}"
            lines += [*coordinate_lines, f"@{predicate} {instruction}"]
        lines += ["@${pred:st_leader} cp.async.bulk.commit_group;", "${label:st_done}:"]
        return _lowered(moves[0], _leader_thread("st", issuer_warp=issuer) + "\n".join(lines) + "\n",
                        operands, operand_names, description="Grouped TMA stores")

    def _lower_async_tma_loads(
        self,
        op: AsyncOp,
        tmas: list[TmaMoveOp],
        lock_value: Value,
    ) -> tuple[list[Op], list[object]]:
        """Lower one or more gmem → smem TMA loads sharing a single mbarrier.

        Each load issues cp.async.bulk.tensor against the shared mbarrier, then a
        single mbarrier.arrive.expect_tx covers their summed byte count — so a
        fused lhs+rhs pair (or any run of loads) needs one mbarrier and one
        arrive instead of one per load (fastcu's single-``full``-barrier idea).
        One elected thread of warp 0 issues; the await op emits the wait.
        """
        lock_type = lock_value.type
        assert isinstance(lock_type, LockType)
        assert lock_type.kind == "mbar", (
            f"_lower_async_tma_loads: expected lock kind 'mbar'; got {lock_type.kind!r}"
        )
        cta_group = lock_type.attrs.get("cta_group", 1)

        before_branch = (
            _pair_leader_ptx("cluster_rank", "cluster_leader") if cta_group == 2
            else "mov.u32 ${b32:cluster_rank}, %cluster_ctarank;\n"
            "setp.eq.u32 ${pred:cluster_leader}, ${b32:cluster_rank}, 0;\n")
        leader_thread = _leader_thread(
            "ld",
            before_branch=before_branch,
            issuer_warp=execution_issuer_warp(op),
            guard=(guard := self._region_guard("ld", execution_issuer_warp(op)) or ([], [], None))[2],
        )
        if self._issuer_region(execution_issuer_warp(op)) in self._issuer_elections:
            elected_operands, elected_names, leader_thread = self._leader_thread(
                'ld', before_branch=before_branch, issuer_warp=execution_issuer_warp(op))
            guard = elected_operands, elected_names, None
        operands: list[Value] = [lock_value, *guard[0]]
        operand_names: list[str] = ["lock", *guard[1]]
        multicast_mask: int | None = None
        cluster_size = 1
        if op.is_tma_multicast:
            kernel = find_ancestor_of_type(op, KernelOp)
            assert kernel is not None
            cluster_dim = kernel.cluster_dim
            cluster_size = math.prod(dim or 1 for dim in cluster_dim)
            multicast_mask = (1 << cluster_size) - 1
        total_bytes = 0
        load_lines: list[str] = []
        multicast_count = sum(tma.multicast is True for tma in tmas)
        multicasts = 0
        group_ordinals = {}
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
            idx_values = [cast(Value, self.value_map[v]) for v in tma.indices]
            dn, sn = f"dst{i}", f"desc{i}"
            idx_names = [f"idx{i}_{j}" for j in range(len(idx_values))]
            operands += [dst_value, desc_value] + idx_values
            operand_names += [dn, sn] + idx_names
            coordinate_lines, coordinates = _tma_coordinates(
                desc_type,
                idx_names,
                f"load{i}",
            )
            tensor_rank = len(coordinates)
            indices = ", ".join(coordinates)
            load_lines.extend(coordinate_lines)
            opcode = (
                f"cp.async.bulk.tensor.{tensor_rank}d."
                + ("cta_group::2." if cta_group == 2 else "")
                + "shared::cluster.global."
                "mbarrier::complete_tx::bytes"
            )
            arguments = (
                f"[${{b32:operand.{dn}.ptr}}], "
                f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                f"[${{b32:operand.lock.val}}]"
            )
            if cta_group == 2:
                load_lines.extend([
                    "and.b32 ${b32:tma_pair_leader_rank}, ${b32:cluster_rank}, 0xfffffffe;",
                    "mapa.shared::cluster.u32 ${b32:tma_pair_bar}, ${b32:operand.lock.val}, ${b32:tma_pair_leader_rank};",
                    f"mapa.shared::cluster.u32 ${{b32:tma_pair_dst{i}}}, ${{b32:operand.{dn}.ptr}}, ${{b32:cluster_rank}};",
                ])
                arguments = (f"[${{b32:tma_pair_dst{i}}}], "
                             f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                             "[${b32:tma_pair_bar}]")
            issue = "ld_leader"
            if tma.multicast is True:
                assert multicast_mask is not None
                opcode = f"{opcode}.multicast::cluster"
                masks = tma.attrs.get('multicast_cta_masks')
                if masks is not None:
                    key = tuple(masks)
                    ordinal = group_ordinals.get(key, 0)
                    group_ordinals[key] = ordinal + 1
                    source_table = multicast_sources(masks, ordinal)
                    load_lines += rank_table_ptx(f'multicast_mask{i}', masks, bits=16).splitlines()
                    load_lines += rank_table_ptx(f'multicast_source{i}', source_table).splitlines()
                    arguments += f", ${{b16:multicast_mask{i}}}"
                    issue = f"multicast_issue{i}"
                    load_lines += [
                        f"setp.eq.u32 ${{pred:issuer{i}}}, ${{b32:cluster_rank}}, ${{b32:multicast_source{i}}};",
                        f"and.pred ${{pred:{issue}}}, ${{pred:ld_leader}}, ${{pred:issuer{i}}};",
                    ]
                else:
                    arguments = f"{arguments}, {multicast_mask}"
                    issue = "multicast_issue"
                    if multicast_count > 1:
                        issue = f"multicast_issue{i}"
                        load_lines += [
                            f"setp.eq.u32 ${{pred:issuer{i}}}, ${{b32:cluster_rank}}, {multicasts % cluster_size};",
                            f"and.pred ${{pred:{issue}}}, ${{pred:ld_leader}}, ${{pred:issuer{i}}};",
                        ]
                    multicasts += 1
            load_lines.append(
                f"@${{pred:{issue}}} "
                + self._tma_instruction(
                    tma,
                    opcode,
                    arguments,
                    operands,
                    operand_names,
                    f"cache_policy{i}",
                )
            )

        # Plain-string lines (literal ${...} placeholders); only load_lines and
        # the byte count are interpolated, so we avoid f-string/dedent brace and
        # multi-line-indent pitfalls by joining a line list.
        #
        # The buffer-reuse WAR (t0's reload vs prior readers of the smem tile) is
        # handled by insert_syncs, which emits a barrier before this async.
        lines = [
            "and.pred ${pred:multicast_issue}, ${pred:ld_leader}, "
            "${pred:cluster_leader};",
            *load_lines,
            ("@${pred:multicast_issue} " if cta_group == 2 else "@${pred:ld_leader} ")
            + "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
            f"[${{b32:operand.lock.val}}], {total_bytes * cta_group};",
            "${label:ld_done}:",
        ]
        template = leader_thread + "\n".join(lines) + "\n"

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []


def _lowered(op: Op, template: str, operands=(), names=(), result_types=(),
             description: str | None = None) -> tuple[list[Op], list[object]]:
    """``op`` lowered to one PTX template, whose results are named ``dst``."""
    lowered = PTXTemplateOp.build(
        operands=list(operands), result_types=list(result_types), template=template,
        operand_names=list(names), result_names=["dst"] if result_types else [],
        description=str(op) if description is None else description)
    return [lowered], list(lowered.results)


def _layout_view_root(value: Value) -> Value:
    while isinstance(value.definition, (TileOp, ReshapeOp)):
        value = value.definition.arg
    return value


def _block_aligned_view(value: Value, block_width: int, inner_dim: int) -> bool:
    """Whether every tile in a view chain starts at a whole physical block."""
    while isinstance(value.definition, TileOp):
        tile = value.definition
        index = tile.indices[inner_dim]
        extent = tile.result().type.sizes[inner_dim]
        if isinstance(index, int):
            if index * extent % block_width:
                return False
        elif extent % block_width:
            return False
        value = tile.arg
    return not isinstance(value.definition, ReshapeOp)


def _scalar_shared_vector_aligned(value: Value) -> bool:
    """Prove a row-major FP32 view starts on a 16-byte chunk.

    Shared allocations have at least 128-byte alignment. Every tile offset
    must preserve four-element alignment, including dynamic indices. A
    blocked row stride and block boundary must preserve that alignment too.
    Swizzles permute whole 16-byte chunks, so an aligned vector stays within
    one chunk. Reshapes, casts, column-major blocks and unknown roots retain
    scalar lowering rather than inventing an alignment promise.
    """
    typ = value.type
    if (not isinstance(typ, SmemTensorType) or not isinstance(typ.dtype, F32Type)
            or not typ.rank or typ.strides[-1] != 1
            or any(s is None or s % 4 for s in typ.strides[:-1])):
        return False
    while isinstance(value.definition, TileOp):
        tile = value.definition
        parent = tile.arg.type
        if (not isinstance(parent, SmemTensorType) or parent.rank != typ.rank
                or parent.dtype != typ.dtype or parent.strides[-1] != 1
                or any(s is None or s % 4 for s in parent.strides[:-1])):
            return False
        for index, extent, stride in zip(tile.indices, value.type.sizes, parent.strides):
            if extent is None or stride is None:
                return False
            coefficient = extent * stride
            if isinstance(index, int):
                if index * coefficient % 4:
                    return False
            elif coefficient % 4:
                return False
        value = tile.arg
    if not isinstance(value.definition, AllocOp):
        return False
    layout = physical_layout(value.type)
    if layout is not None and (layout[1] != "block_major"
                               or layout[0] % 4 or is_column_major(value.type)):
        return False
    return True


def _row_fragment_shared_alignment(value: Value) -> int:
    """Conservative pointer alignment, capped at the largest copy vector.

    Block-major mapping and swizzling preserve the low four offset bits
    when a physical block is a multiple of 16 bytes. Unknown provenance
    leaves alignment to the runtime copy guard.
    """
    alignment = 16
    while True:
        op = value.definition
        if isinstance(op, ReinterpretOp):
            alignment = math.gcd(alignment, op.byte_offset)
            value = op.arg
        elif isinstance(op, ReshapeOp):
            value = op.arg
        elif isinstance(op, TileOp):
            parent = op.arg.type
            if not isinstance(parent, SmemTensorType):
                return 0
            layout = physical_layout(parent)
            if layout is not None and (layout[1] != "block_major" or
                    is_column_major(parent) or layout[0] * parent.dtype.byte_size % 16):
                return 0
            for dim, (index, extent, stride) in enumerate(zip(op.indices, value.type.sizes, parent.strides)):
                if extent is None or stride is None:
                    return 0
                # Only the intra-block column and row offsets affect low bits.
                stride = (layout[0] if dim == 0 else 1) if layout is not None else stride
                offset = extent * stride * parent.dtype.byte_size
                alignment = math.gcd(alignment, offset * index if isinstance(index, int) else offset)
            value = op.arg
        elif isinstance(op, AllocOp) and isinstance(value.type, SmemTensorType):
            return math.gcd(alignment, cast(int, op.attrs.get("allocation_alignment", 128)))
        else:
            return 0


def _constant_divmod_ptx(kind: str, dst: str, src: str, divisor: int) -> str:
    """Exact unsigned arithmetic without reciprocal work for powers of two."""
    assert kind in ("div", "rem") and divisor > 0
    if divisor & (divisor - 1) == 0:
        return (
            f"shr.u32 {dst}, {src}, {divisor.bit_length() - 1};\n"
            if kind == "div"
            else f"and.b32 {dst}, {src}, {divisor - 1};\n"
        )
    return f"{kind}.u32 {dst}, {src}, {divisor};\n"


def _swizzle_ptx(offset: str, scratch: str, row_bytes: int) -> str:
    """XOR the 16-byte chunk of ``${b32:offset}`` with the row bits above it.

    CuTe's ``Swizzle<log2(row_bytes / 16), 4, 3>``, the SMEM swizzle whose
    span is one ``row_bytes`` row of a blocked buffer.
    """
    mask = (row_bytes // 16 - 1) << 7
    return textwrap.dedent(f"""\
        and.b32 ${{b32:{scratch}}}, ${{b32:{offset}}}, {mask};
        shr.u32 ${{b32:{scratch}}}, ${{b32:{scratch}}}, 3;
        xor.b32 ${{b32:{offset}}}, ${{b32:{offset}}}, ${{b32:{scratch}}};
    """)


def _vector_groups(count: int, alignment_bytes: int) -> list[tuple[int, int]]:
    """Split ``count`` contiguous 32-bit registers into aligned vectors.

    Each group is ``(first register, width)``; a vector is at most 16 bytes
    and never straddles a multiple of ``alignment_bytes``.
    """
    groups = []
    first = 0
    while first < count:
        width = next(
            width
            for width in (4, 2, 1)
            if first % width == 0
            and first + width <= count
            and alignment_bytes % (4 * width) == 0
        )
        groups.append((first, width))
        first += width
    return groups


def _vector_operand(registers: list[str]) -> tuple[str, str]:
    """The ``.vN`` suffix and operand of a register group."""
    if len(registers) == 1:
        return "", registers[0]
    return f".v{len(registers)}", "{" + ", ".join(registers) + "}"


# ``tcgen05.ld``/``tcgen05.st`` shape for each fragment layout: lanes one
# instruction covers, then the columns and registers of one repetition.
def _tmem_warp_address_ptx(pointer: str, group_row_stride: int = 0, group_column_stride: int = 0) -> str:
    """PTX setting ``${b32:taddr}`` to the warp's 32-lane TMEM quarter.

    A tensor-memory address holds the lane in its upper 16 bits, and warp
    ``w`` of a warp group can only access lanes ``32 * (w % 4)`` onward. The
    template's ``warp`` operand is the kernel's warp index.
    """
    template = textwrap.dedent(f"""\
        and.b32 ${{b32:tmem_lane}}, ${{b32:operand.warp.val}}, 3;
        shl.b32 ${{b32:tmem_lane}}, ${{b32:tmem_lane}}, 21;
        add.u32 ${{b32:taddr}}, ${{b32:{pointer}}}, ${{b32:tmem_lane}};
    """)
    if group_row_stride:
        template += textwrap.dedent(f"""\
            shr.u32 ${{b32:tmem_group}}, ${{b32:operand.warp.val}}, 2;
            mad.lo.u32 ${{b32:taddr}}, ${{b32:tmem_group}}, {group_row_stride << 16}, ${{b32:taddr}};
        """)
    if group_column_stride:
        template += textwrap.dedent(f"""\
            shr.u32 ${{b32:tmem_column_group}}, ${{b32:operand.warp.val}}, 2;
            mad.lo.u32 ${{b32:taddr}}, ${{b32:tmem_column_group}}, {group_column_stride}, ${{b32:taddr}};
        """)
    return template


def _tmem_fragment_ptx(pointer: str, ownership: TmemWarpLayout, layout: str, columns: int,
                       registers: list[str], *, is_load: bool) -> str:
    """Address the warp's TMEM rows and columns, then move ``registers`` to or from them."""
    return _tmem_warp_address_ptx(
        pointer, ownership.group_row_stride, 0 if ownership.paired_m64 else ownership.group_column_stride(columns)
    ) + _tmem_access_ptx(layout, ownership.columns_per_warp(columns), registers, is_load=is_load,
                         rows_per_warp=ownership.rows_per_warp)


def _tmem_access_ptx(
    layout: str,
    columns: int,
    registers: list[str],
    *,
    is_load: bool,
    rows_per_warp: int = 32,
) -> str:
    """Move a fragment between ``${b32:taddr}`` and ``registers``, then wait.

    Each warp covers 16 or 32 rows in its physical 32-row partition. Along the
    columns, instructions repeat their shape as widely as PTX allows, in
    powers of two of at most 128 registers.
    """
    geometry = TMEM_FRAGMENT_LAYOUTS[layout]
    assert len(registers) == rows_per_warp * columns // 32
    template = ""
    for row, column, first, size, repetitions in geometry.accesses(columns, rows_per_warp):
        regs = ", ".join(registers[first : first + size])
        address = row << 16 | column
        template += f"add.u32 ${{b32:tmem_access}}, ${{b32:taddr}}, {address};\n"
        count = size if geometry.split_offset is not None else repetitions
        instruction = f"tcgen05.{'ld' if is_load else 'st'}.sync.aligned.{geometry.shape}.x{count}.b32"
        suffix = f", {geometry.split_offset}" if geometry.split_offset is not None else ""
        template += (
            f"{instruction} {{{regs}}}, [${{b32:tmem_access}}]{suffix};\n"
            if is_load
            else f"{instruction} [${{b32:tmem_access}}], {{{regs}}}{suffix};\n"
        )
    return template + f"tcgen05.wait::{'ld' if is_load else 'st'}.sync.aligned;\n"


# Accumulator fragments built from m16n8 blocks differ only in where a
# thread's rows sit: the row distance between consecutive warps, and between
# one thread's 16-row tiles. Tensor memory confines each warp to 32 lanes.
_M16N8_FRAGMENT_ROWS = {"wgmma_layout_0": (16, 64), "tcgen05_16x256b": (32, 16)}


def _paired_fragment(typ: WgrmemTensorType) -> bool:
    """Whether a local M64 fragment uses the paired column-half mapping."""
    return typ.attrs.get("tmem_cta_group") == 2 and typ.sizes[0] == 64


def _m16n8_ownership(typ: WgrmemTensorType) -> TmemWarpLayout | None:
    """Warp ownership of a TMEM-backed m16n8 fragment; None for WGMMA's."""
    if typ.layout != "tcgen05_16x256b":
        return None
    return TmemWarpLayout(cast(int, typ.sizes[0]), typ.warps, typ.attrs.get("tmem_cta_group", 1))


def _m16n8_fragment_rows(typ: WgrmemTensorType) -> tuple[int, int]:
    ownership = _m16n8_ownership(typ)
    return (ownership.row_stride, 16) if ownership else _M16N8_FRAGMENT_ROWS[typ.layout]


def _m16n8_fragment_columns(typ: WgrmemTensorType) -> int:
    ownership = _m16n8_ownership(typ)
    columns = cast(int, typ.sizes[1])
    return ownership.columns_per_warp(columns) if ownership else columns


def _m16n8_fragment_group_columns(typ: WgrmemTensorType) -> int:
    ownership = _m16n8_ownership(typ)
    return ownership.group_column_stride(cast(int, typ.sizes[1])) if ownership and not _paired_fragment(typ) else 0


def _m16n8_fragment_group_rows(typ: WgrmemTensorType) -> int:
    ownership = _m16n8_ownership(typ)
    return ownership.group_row_stride if ownership else 0


def _build_wgmma_layout_0_dynamic_offset(
    row_stride_bytes: int,
    column_stride_bytes: int,
    *, coordinates: bool = False,
    warp_rows: int = 16,
    warp_group_rows: int = 0,
    warp_group_columns: int = 0,
    paired_columns: int = 0,
) -> tuple[list[Op], Value] | tuple[list[Op], Value, Value, Value]:
    """Emit index ops for an m16n8 fragment's per-thread byte offset.

    warp_in_wg = (tid.x % 128) / 32
    lane = tid.x % 32
    row = warp_in_wg * warp_rows + lane / 4
    col = (lane % 4) * 2
    byte_offset = row * row_stride_bytes + col * column_stride_bytes
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
    column_pair = None
    if paired_columns:
        # Paired M64 owns two row partitions and two column halves. Warps
        # 2/3 share the rows of 0/1 and drain the other N half.
        c2 = IndexConstantOp.build(2)
        column_pair = IndexDivOp.build(warp.result(), c2.result())
        row_pair = IndexModOp.build(warp.result(), c2.result())
        ops.extend([c2, column_pair, row_pair])
        warp = row_pair
    lane = IndexModOp.build(tid_in_wg.result(), c32.result())
    ops.append(lane)

    # row = warp * warp_rows + lane / 4
    c_warp_rows = IndexConstantOp.build(warp_rows)
    ops.append(c_warp_rows)
    warp_row = IndexMulOp.build(warp.result(), c_warp_rows.result())
    ops.append(warp_row)
    c4 = IndexConstantOp.build(4)
    ops.append(c4)
    lane_div_4 = IndexDivOp.build(lane.result(), c4.result())
    ops.append(lane_div_4)
    row = IndexAddOp.build(warp_row.result(), lane_div_4.result())
    ops.append(row)
    if warp_group_rows:
        group = IndexDivOp.build(tid_x, c128.result())
        stride = IndexConstantOp.build(warp_group_rows)
        group_row = IndexMulOp.build(group.result(), stride.result())
        row = IndexAddOp.build(row.result(), group_row.result())
        ops.extend([group, stride, group_row, row])

    # col = (lane % 4) * 2
    lane_mod_4 = IndexModOp.build(lane.result(), c4.result())
    ops.append(lane_mod_4)
    c2 = IndexConstantOp.build(2)
    ops.append(c2)
    col = IndexMulOp.build(lane_mod_4.result(), c2.result())
    ops.append(col)
    if column_pair is not None:
        pair_stride = IndexConstantOp.build(paired_columns)
        pair_col = IndexMulOp.build(column_pair.result(), pair_stride.result())
        col = IndexAddOp.build(col.result(), pair_col.result())
        ops.extend([pair_stride, pair_col, col])

    if warp_group_columns:
        group = IndexDivOp.build(tid_x, c128.result())
        stride = IndexConstantOp.build(warp_group_columns)
        group_col = IndexMulOp.build(group.result(), stride.result())
        col = IndexAddOp.build(col.result(), group_col.result())
        ops.extend([group, stride, group_col, col])

    # byte_offset = row * row_stride_bytes + col * column_stride_bytes
    row_stride = IndexConstantOp.build(row_stride_bytes)
    ops.append(row_stride)
    row_bytes = IndexMulOp.build(row.result(), row_stride.result())
    ops.append(row_bytes)
    col_stride = IndexConstantOp.build(column_stride_bytes)
    ops.append(col_stride)
    col_bytes = IndexMulOp.build(col.result(), col_stride.result())
    ops.append(col_bytes)
    byte_offset = IndexAddOp.build(row_bytes.result(), col_bytes.result())
    ops.append(byte_offset)

    if coordinates:
        return ops, byte_offset.result(), row.result(), col.result()
    return ops, byte_offset.result()


def _wgmma_pair_static_offset(
    pair_index: int, n_dim: int, tile_rows: int = 64
) -> tuple[int, int]:
    """Thread-independent root offset of an m16n8 fragment pair's first element.

    The runtime warp/lane offset selects the thread's base coordinate. This
    returns the additional logical ``(row, column)`` offset for ``pair_index``
    in that thread's register fragment.
    """
    assert n_dim % 8 == 0
    pairs_per_m_tile = n_dim // 4
    m, pair_in_m_tile = divmod(pair_index, pairs_per_m_tile)
    w, pair_in_16_columns = divmod(pair_in_m_tile, 4)
    di, dj = [(0, 0), (8, 0), (0, 8), (8, 8)][pair_in_16_columns]
    return (m * tile_rows + di, w * 16 + dj)


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
    """Map an m16n8 fragment WGRMEM value or view to pairs in its root.

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
    assert view_type.layout == root_type.layout in _M16N8_FRAGMENT_ROWS
    assert view_type.rank == root_type.rank == 2
    assert view_type.warps == root_type.warps
    if root_type.layout == "wgmma_layout_0":
        assert root_type.warps == 4
    else:
        _m16n8_ownership(root_type)
    assert view_type.dtype == root_type.dtype
    assert isinstance(view_type.dtype, (F32Type, BF16Type, F16Type)), (
        f"WGMMA views support f32, bf16, and f16; got {view_type.dtype}"
    )
    assert root_type.num_elements_per_thread % 2 == 0
    if root_type.layout != "wgmma_layout_0":
        if view_type.attrs.get("tmem_cta_group", 1) != root_type.attrs.get("tmem_cta_group", 1):
            raise ValueError("WGRMEM views must preserve TMEM CTA group metadata")
    row_start, col_start = _static_view_origin(value)
    view_m, view_n = cast(tuple[int, int], view_type.sizes)
    root_m, logical_n = cast(tuple[int, int], root_type.sizes)
    if (view_n <= 0 or view_n % 8 or col_start % 8
            or row_start < 0 or col_start < 0
            or row_start + view_m > root_m or col_start + view_n > logical_n):
        raise ValueError("m16n8 views require complete eight-column packets within their root")
    ownership = _m16n8_ownership(root_type)
    if (ownership is not None and ownership.column_groups == 2
            and (row_start != 0 or col_start != 0 or view_type.sizes != root_type.sizes)):
        raise NotImplementedError("column-sharing TMEM fragments require whole register values")
    root_n = _m16n8_fragment_columns(root_type)
    _, tile_rows = _M16N8_FRAGMENT_ROWS[root_type.layout]
    pairs = [
        (pair_index, row - row_start, col - col_start)
        for pair_index in range(root_type.num_elements_per_thread // 2)
        for row, col in [_wgmma_pair_static_offset(pair_index, root_n, tile_rows)]
        if row_start <= row < row_start + view_m
        and col_start <= col < col_start + view_n
    ]
    assert len(pairs) == view_m * view_n // (root_type.warps * 64)
    return pairs


def _view_pair_indices(value: Value) -> list[int]:
    """Return per-thread root pair indices selected by a WGRMEM value or view."""
    assert isinstance(value.type, WgrmemTensorType)
    if value.type.layout in _M16N8_FRAGMENT_ROWS:
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
        2 * pair + element for pair in _view_pair_indices(value) for element in (0, 1)
    ]


def _stmatrix_transposed_groups(
    register_starts: list[int],
    n_dim: int,
    col_start: int,
    dst_rows: int,
    tile_rows: int,
) -> list[tuple[list[int], int]]:
    """Registers and static offsets of each transposed STMatrix.

    A group is the four pairs of one 16x16 fragment block, or the two pairs
    of a trailing 8-column block (an x2).
    """
    pairs_per_m_tile = n_dim // 4
    groups: list[tuple[list[int], int]] = []
    for first in register_starts:
        count = min(4, pairs_per_m_tile - first % pairs_per_m_tile)
        offset = _stmatrix_store_static_offset(
            first,
            n_dim,
            col_start,
            dst_rows,
            tile_rows=tile_rows,
            transposed=True,
        )
        groups.append((list(range(first, first + count)), offset))
    return groups


def _stmatrix_groups(
    register_starts: list[int],
    pairs: set[int],
    n_dim: int,
    col_start: int,
    dst_m: int,
    block_width: int,
    tile_rows: int,
) -> list[tuple[list[int], int]]:
    """Registers and static offsets of each STMatrix.

    Two adjacent 16-column groups of a fragment tile form an x4 per 8-row
    half; a trailing 16 or 8 columns form an x2 or an x1.
    """
    pairs_per_m_tile = n_dim // 4

    def in_tile(pair: int, step: int) -> bool:
        return pair % pairs_per_m_tile + step < pairs_per_m_tile and (
            pair + step in pairs
        )

    chunks: list[list[int]] = []
    for start in register_starts:
        if (
            chunks
            and len(chunks[-1]) == 1
            and start == chunks[-1][0] + 4
            and start % pairs_per_m_tile
            and in_tile(start, 2)
        ):
            chunks[-1].append(start)
        else:
            chunks.append([start])
    groups: list[tuple[list[int], int]] = []
    for chunk in chunks:
        columns = [
            pair + step for pair in chunk for step in (0, 2) if in_tile(pair, step)
        ]
        for half in (0, 1):
            groups.append(
                (
                    [pair + half for pair in columns],
                    _stmatrix_store_static_offset(
                        chunk[0] + half,
                        n_dim,
                        col_start,
                        dst_m,
                        block_width,
                        tile_rows,
                    ),
                )
            )
    return groups


def _stmatrix_store_static_offset(
    pair_index: int,
    n_dim: int,
    col_start: int,
    dst_rows: int,
    block_width: int = 64,
    tile_rows: int = 64,
    transposed: bool = False,
) -> int:
    """Byte offset for a fragment pair in the destination's blocked layout."""
    di, dj = _wgmma_pair_static_offset(pair_index, n_dim, tile_rows)
    row, col = di, dj - col_start
    if transposed:
        row, col = col, row
    assert row >= 0 and col >= 0 and col % 8 == 0
    return _block_major_offset(row, col, block_width, dst_rows) * 2


def _block_major_offset(row: int, column: int, block_width: int, rows: int) -> int:
    """Element offset of (row, column) in a block-major buffer of ``rows`` rows."""
    block, inner = divmod(column, block_width)
    return (block * rows + row) * block_width + inner


def _block_swizzle(block_width: int) -> str | None:
    return f"{2 * block_width}B" if block_width > 8 else None


def _tma_coordinates(
    desc_type: TmaDescType,
    index_names: list[str],
    prefix: str,
) -> tuple[list[str], list[str]]:
    layout = physical_layout(desc_type)
    if layout is None:
        return [], [f"${{b32:operand.{name}.val}}" for name in reversed(index_names)]

    assert desc_type.gmem_type.rank >= 2
    assert len(index_names) == desc_type.gmem_type.rank
    row = f"${{b32:operand.{index_names[-2]}.val}}"
    col = f"${{b32:operand.{index_names[-1]}.val}}"
    outer = [f"${{b32:operand.{name}.val}}" for name in reversed(index_names[:-2])]
    block = f"${{b32:{prefix}_block}}"
    inner = f"${{b32:{prefix}_inner}}"
    block_width, order = layout
    # ptxas expands div/rem by an immediate into a reciprocal sequence on the
    # issuing thread's critical path; a power-of-two width is a shift and mask.
    if block_width & (block_width - 1) == 0:
        lines = [
            f"shr.u32 {block}, {col}, {block_width.bit_length() - 1};",
            f"and.b32 {inner}, {col}, {block_width - 1};",
        ]
    else:
        lines = [
            f"div.u32 {block}, {col}, {block_width};",
            f"rem.u32 {inner}, {col}, {block_width};",
        ]
    if order == "row_major":
        coordinates = [inner, block, row, *outer]
    else:
        coordinates = [inner, row, block, *outer]
    return lines, coordinates


def _stmatrix_register_starts(op: MoveOp) -> list[int] | None:
    dst_type = op.dst.type
    src_type = op.src.type
    dst_root = _layout_view_root(op.dst)
    layout = physical_layout(dst_type) if isinstance(dst_type, SmemTensorType) else None
    if not (
        isinstance(dst_type, SmemTensorType)
        and isinstance(src_type, WgrmemTensorType)
        and src_type.layout in _M16N8_FRAGMENT_ROWS
        and isinstance(src_type.dtype, (BF16Type, F16Type))
        and dst_type.dtype == src_type.dtype
        and dst_type.rank == 2
        and (dst_type.strides[-1] == 1 or is_column_major(dst_type))
        and layout is not None
        and layout[1] == "block_major"
        and dst_type.swizzle == _block_swizzle(layout[0])
    ):
        return None
    block_width = layout[0]

    dst_root = _layout_view_root(op.dst)
    assert dst_root is not None
    root_type = dst_root.type
    if not isinstance(root_type, SmemTensorType) or root_type.rank != 2:
        return None
    transposed = root_type.is_column_major_2d
    dense = (
        list(root_type.strides) == [1, root_type.sizes[0]]
        if transposed
        else root_type.is_canonical_strides
    )
    if (
        root_type.swizzle != _block_swizzle(block_width)
        or physical_layout(root_type) != (block_width, "block_major")
        or cast(int, root_type.sizes[0]) % 64
        or not dense
    ):
        return None

    m_dim, n_dim = cast(tuple[int, int], src_type.sizes)
    src_root = root_buffer(op.src)
    assert src_root is not None
    src_root_type = src_root.type
    assert isinstance(src_root_type, WgrmemTensorType)
    root_n = _m16n8_fragment_columns(src_root_type)
    is_view = isinstance(op.src.definition, TileOp)
    row_start, col_start = (
        _static_view_origin(op.src)
        if is_view
        else (0, op.attrs.get("wgmma_col_start", 0))
    )
    assert isinstance(col_start, int)
    pairs = _stmatrix_pairs(op)
    dst_m, dst_n = cast(tuple[int, int], dst_type.sizes)
    if (
        dst_m != m_dim
        or (is_view and dst_n != n_dim)
        or row_start != 0
        or col_start < 0
        or col_start % 16
        or (not is_view and col_start + dst_n > n_dim)
        or m_dim % ((2 if _paired_fragment(src_root_type) else 4) * _m16n8_fragment_rows(src_root_type)[0])
        or dst_n % (8 if transposed else block_width)
        or root_n % 8
        or not (8 <= root_n <= 256)
    ):
        return None

    if transposed:
        # A trailing 8-column group uses x2; other groups must contain all
        # four pairs of a 16-column fragment block.
        if dst_n % 16 and col_start + dst_n != root_n:
            return None
        pairs_per_m_tile = root_n // 4
        return [pair for pair, _, _ in pairs if pair % pairs_per_m_tile % 4 == 0]
    # Each start opens a 16-column group of one 64-row fragment tile.
    pairs_per_m_tile = root_n // 4
    return [pair for pair, _, _ in pairs if pair % pairs_per_m_tile % 4 == 0]


def _m16n8_move_pairs(op: MoveOp) -> list[tuple[int, int, int]]:
    """Validate complete lane packets and select root registers for a move."""
    fragment, memory = ((op.dst, op.src) if isinstance(op.dst.type, WgrmemTensorType)
                        else (op.src, op.dst))
    fragment_type, memory_type = fragment.type, memory.type
    assert isinstance(fragment_type, WgrmemTensorType)
    assert isinstance(memory_type, AddrMemTensorType)
    if (fragment_type.rank != 2 or memory_type.rank != 2
            or fragment_type.dtype != memory_type.dtype):
        raise ValueError("m16n8 moves require rank-2 tensors with matching dtypes")
    memory_m, memory_n = cast(tuple[int, int], memory_type.sizes)
    col_start = op.attrs.get("wgmma_col_start", 0)
    if type(col_start) is not int or col_start < 0 or col_start % 8:
        raise ValueError("wgmma_col_start must be a nonnegative integer multiple of eight")
    if memory_n <= 0 or memory_n % 8:
        raise ValueError("m16n8 memory tiles must cover complete eight-column packets")
    pairs = _wgmma_view_pairs(fragment)
    if isinstance(fragment.definition, TileOp):
        if memory_type.sizes != fragment_type.sizes:
            raise ValueError("m16n8 moves require matching memory and fragment view shapes")
        if col_start != 0:
            raise ValueError("explicit WGRMEM views cannot also specify wgmma_col_start")
        return pairs
    fragment_m, fragment_n = cast(tuple[int, int], fragment_type.sizes)
    if memory_m != fragment_m or col_start + memory_n > fragment_n:
        raise ValueError("m16n8 move slice exceeds its register fragment")
    ownership = _m16n8_ownership(fragment_type)
    if (ownership is not None and ownership.column_groups == 2
            and (col_start != 0 or memory_n != fragment_n)):
        raise NotImplementedError("column-sharing TMEM fragments require whole register values")
    return [(pair, row, col - col_start) for pair, row, col in pairs
            if col_start <= col < col_start + memory_n]


def _stmatrix_pairs(op: MoveOp) -> list[tuple[int, int, int]]:
    """The WGMMA pairs a fragment store writes, relative to its destination."""
    return _m16n8_move_pairs(op)


def _ptx_suffix(dtype) -> str:
    """PTX storage suffix for a dtype; 16-bit floats move as raw bits."""
    name = dtype.name
    return "b16" if name in ("bf16", "f16") else name


def _mov_literals_ptx(result: str, dtype: Type, values) -> str:
    """Move literals into a result's registers; 16-bit floats move as bits."""
    suffix = _ptx_suffix(dtype)
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
