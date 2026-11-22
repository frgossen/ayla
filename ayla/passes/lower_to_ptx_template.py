"""Lower GPU-dialect operations to PTX templates.

Scalar instructions, hardware transfers, matmul, and synchronization are
lowered here. LowerToScalar exposes reduction loops, ownership guards, and
shuffles in IR before this pass selects instructions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import itertools
import math
import struct
import textwrap
from typing import cast

from ..register_layout import (
    RegisterLayout,
    wgmma_pair_offset as _wgmma_pair_static_offset,
    static_view_origin as _static_view_origin,
    wgmma_view_pairs as _wgmma_view_pairs,
)
from ..base.ir import Op, Type, Value, find_ancestor_of_type, find_ops_of_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BF16Type,
    E4M3Type,
    E5M2Type,
    CollectiveAxis,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    LoadIndexOp,
    IndexAddOp,
    IndexEqOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    MatmulOp,
    ReduceOp,
    RegisterViewOp,
    ShuffleOp,
    PredicateType,
    WrmemTensorType,
    MemTensorType,
    ModuleOp,
    MoveOp,
    PtrAddOp,
    PtrType,
    PTXTemplateOp,
    ReinterpretOp,
    RegMemTensorType,
    ReshapeOp,
    RmemTensorType,
    ConstantOp,
    SmemTensorType,
    ThreadIdOp,
    TileOp,
    TileSyncOp,
    TmaDescType,
    TmaMoveOp,
    ToPtrOp,
    WarpIdsOp,
    WgrmemTensorType,
    is_column_major,
    physical_layout,
    lock_has_tma_multicast,
    root_buffer,
)


def _warpgroup_copies(op: Op) -> int:
    """Copies of a 128-thread CTA program resident in one CTA, one per warpgroup."""
    kernel = find_ancestor_of_type(op, KernelOp)
    return 1 if kernel is None else int(kernel.attrs.get("warpgroup_ctas", 1))


def _copy_offset(name: str, stride: int, copies: int) -> str:
    """Offset a private allocation to the executing copy's instance."""
    if copies == 1:
        return ""
    return f"mad.lo.u32 ${{b32:{name}}}, %vwg, {stride}, ${{b32:{name}}};"


def _ptx_reg(kind: str, name: str) -> str:
    return f"${{{kind}:{name}}}"


def _register_operand_regs(value: Value, name: str) -> tuple[str, list[str]]:
    """Format IR-selected storage registers using PTX's scalar/packed types."""
    typ = cast(RegMemTensorType, value.type)
    suffix = "b16" if typ.dtype.name in ("bf16", "f16") else typ.dtype.name
    packed = isinstance(typ, WgrmemTensorType) and suffix == "b16"
    if packed:
        suffix = "b32"
    indices = typ.register_indices(value, elements_per_register=2 if packed else 1)
    return suffix, [_ptx_reg(suffix, f"operand.{name}.r{i}") for i in indices]


FUNCTIONS = {
    "add": ("add.f32", (0, 1)),
    "sub": ("sub.f32", (0, 1)),
    "max": ("max.NaN.f32", (0, 1)),
    "div": ("div.rn.f32", (0, 1)),
    "mul": ("mul.f32", (0, 1)),
    "square": ("mul.f32", (0, 0)),
    "rsqrt": ("rsqrt.approx.f32", (0,)),
    "neg": ("neg.f32", (0,)),
}


def _lower_register_arithmetic(
    dst: Value, srcs: Sequence[Value], fn: str, names: Sequence[str]
) -> str:
    native_bf16 = (
        isinstance(dst.type, RmemTensorType)
        and isinstance(dst.type.dtype, BF16Type)
        and fn in ("add", "sub", "mul")
        and all(value.type == dst.type for value in srcs)
    )
    if not native_bf16 and any(cast(MemTensorType, v.type).dtype != F32Type() for v in (dst, *srcs)):
        raise NotImplementedError("scalar arithmetic requires F32 operands")
    if native_bf16:
        opcode, operands = f"{fn}.rn.bf16", (0, 1)
    else:
        opcode, operands = ("", (0,)) if fn == "exp" else FUNCTIONS[fn]
    if len(srcs) != max(operands) + 1 or len(names) != len(srcs):
        raise ValueError(f"{fn} operand arity does not match its scalar function")
    if any(source.type != dst.type for source in srcs):
        raise NotImplementedError('decompose register broadcasts with LowerToScalar before PTX lowering')
    storage = [_register_operand_regs(value, name)[1]
               for value, name in zip((dst, *srcs), ('dst', *names), strict=True)]
    if fn == "exp":
        return ''.join(f"mul.f32 ${{f32:exp_tmp}}, {source}, 0f3FB8AA3B;\n"
                       f"ex2.approx.f32 {destination}, ${{f32:exp_tmp}};\n"
                       for destination, source in zip(*storage, strict=True))
    return ''.join(f"{opcode} {destination}, {', '.join(arguments[i] for i in operands)};\n"
                   for destination, *arguments in zip(*storage, strict=True))


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
    if all(isinstance(v.type, (RmemTensorType, WrmemTensorType)) for v in op.operands):
        dst_type = cast(MemTensorType, op.dst.type)
        src_type = cast(MemTensorType, op.src(0).type)
        dst_regs = _elementwise_operand_regs(op.dst, "dst")
        src_regs = _elementwise_operand_regs(op.src(0), "src")
        if len(dst_regs) != len(src_regs):
            raise NotImplementedError("convert requires identical register ownership")
        opcode = f"cvt.rn.{dst_type.dtype.name}.{src_type.dtype.name}"
        if isinstance(dst_type.dtype, F32Type):
            opcode = f"cvt.f32.{src_type.dtype.name}"
        return "".join(f"{opcode} {d}, {s};\n" for d, s in zip(dst_regs, src_regs))
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
    if isinstance(op.dst.type, RmemTensorType) and src_dtype != dst_dtype:
        if not isinstance(src_dtype, F32Type) or not isinstance(dst_dtype, (BF16Type, F16Type)):
            raise NotImplementedError("scalar fused arithmetic requires F32 to 16-bit conversion")
        scratch = [f"${{f32:fused_result{i}}}" for i in range(register_count)]
        template = lower_registers(op, scratch, src_regs)
        return template + ''.join(
            f"cvt.rn.{dst_dtype.name}.f32 {dst}, {src};\n"
            for dst, src in zip(dst_regs, scratch, strict=True)
        )
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


def _lower_scalar_arithmetic(op: ElementwiseOp) -> str:
    sources = [op.src(i) for i in range(op.num_srcs)]
    names = _elementwise_source_names(op)
    dst_type = cast(RegMemTensorType, op.dst.type)
    if dst_type.dtype == F32Type() or (
        isinstance(dst_type, RmemTensorType)
        and isinstance(dst_type.dtype, BF16Type)
        and op.fn in ("add", "sub", "mul")
    ):
        return _lower_register_arithmetic(op.dst, sources, op.fn, names)
    if not isinstance(dst_type, WgrmemTensorType) or not isinstance(dst_type.dtype, (BF16Type, F16Type)):
        raise NotImplementedError('mixed arithmetic outputs require a packed float fragment')
    # Compute in F32 and round once into the requested output representation.
    # Broadcasts have already been decomposed into scalar IR.
    compute = Value(name=None, type=dst_type.with_shape(dst_type.static_shape, F32Type()))
    template = _lower_register_arithmetic(compute, sources, op.fn, names)
    scratch = []
    for i, register in enumerate(_elementwise_operand_regs(compute, 'dst')):
        replacement = f'${{f32:arithmetic_result{i}}}'
        template = template.replace(register, replacement)
        scratch.append(replacement)
    return template + _lower_dtype_conversion_ptx(
        _elementwise_operand_regs(op.dst, 'dst'), scratch, F32Type(), dst_type.dtype)


def _lower_mul_elementwise(op: ElementwiseOp) -> str:
    if isinstance(op.dst.type, RmemTensorType) or all(
        cast(MemTensorType, v.type).dtype == F32Type() for v in op.operands[1:]
    ):
        return _lower_scalar_arithmetic(op)
    return _lower_elementwise_in_batches(op, _lower_mul_register_batches)


_ELEMENTWISE_LOWERERS = {
    "convert": _lower_convert_elementwise_ptx,
    "fill": _lower_fill_ptx,
    "mul": _lower_mul_elementwise,
    **{name: _lower_scalar_arithmetic for name in ("add", "sub", "max", "div", "square", "rsqrt", "neg", "exp")},
    "sigmoid": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_sigmoid_registers,
    ),
    "silu": partial(
        _lower_elementwise_in_batches,
        lower_registers=_lower_silu_registers,
    ),
}


def _leader_warp(prefix: str) -> str:
    """PTX setting ``${pred:<prefix>_other_warp}`` outside warp 0.

    Warp 0 holds the leader, thread 0. Each warp index is broadcast from the
    warp's first thread (CUTLASS's canonical_warp_idx_sync), so ptxas proves
    this predicate warp-uniform. It folds each broadcast of a special register
    away and merges the repeated reads, but keeps one of a combined value.
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
        setp.ne.u32 ${{pred:{prefix}_other_warp}}, ${{b32:{prefix}_warp}}, 0;
    """)


def _leader_thread(prefix: str, before_branch: str = "") -> str:
    """PTX that lets one elected thread of warp 0 do leader work.

    Every other warp branches to ``${label:<prefix>_done}``, without diverging
    inside a warp, and ``elect.sync`` sets ``${pred:<prefix>_leader}`` on one
    thread of warp 0, so ptxas can keep the work's operands in uniform
    registers. ``before_branch`` runs in every warp, so ptxas can keep its
    special-register reads out of loops.
    """
    return (
        _leader_warp(prefix)
        + before_branch
        + (
            f"@${{pred:{prefix}_other_warp}} bra ${{label:{prefix}_done}};\n"
            f"elect.sync _|${{pred:{prefix}_leader}}, 0xffffffff;\n"
        )
    )


# A cluster-to-leader release, in every warp: one elected thread arrives at
# its own CTA's mbarrier, for the CTA leader's next write, and a non-leader
# CTA's also at the cluster leader's, for the cluster leader's next write.
_CLUSTER_RELEASE_ARRIVAL = textwrap.dedent("""\
    bar.warp.sync 0xffffffff;
    elect.sync _|${pred:cl_arrive}, 0xffffffff;
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
_CLUSTER_RELEASE_LEADER_WAIT = _leader_warp("cl") + textwrap.dedent("""\
    @${pred:cl_other_warp} bra ${label:cl_done};
    mov.u32 ${b32:ticks}, 0x989680;
    ${label:cl_wait}:
    mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ${pred:ready}, [${b32:operand.sync.val}], ${b32:operand.sync.parity}, ${b32:ticks};
    @!${pred:ready} bra ${label:cl_wait};
    xor.b32 ${b32:operand.sync.parity}, ${b32:operand.sync.parity}, 1;
    ${label:cl_done}:
    """)


def _split_release(lock: Value) -> bool:
    """Whether a cluster mbarrier releases stages with split arrive/wait."""
    return any(
        isinstance(user, BarrierOp) and "phase" in user.attrs for user in lock.users()
    )


class LowerToPtxTemplate(LoweringPass):
    """Pass that rewrites high-level ops to ptx_template ops."""

    def __init__(self) -> None:
        super().__init__()
        self.cache_policy_values: dict[str, Value] = {}
        self.warp_ids: WarpIdsOp | None = None

    @lowers(TileSyncOp)
    def lower_tile_sync(self, op: TileSyncOp) -> tuple[list[Op], list[object]]:
        from ..tile_codegen import TILE_WAIT_PTX, TILE_WAIT_PAIR_PTX, TILE_PUBLISH_PTX

        names, template = {
            "wait": (["sync", "event", "expected"], TILE_WAIT_PTX),
            "wait_pair": (["sync", "first", "last"], TILE_WAIT_PAIR_PTX),
            "publish": (["sync", "offset", "length"], TILE_PUBLISH_PTX),
        }[cast(str, op.attrs["action"])]
        if op.attrs["action"] == "publish" and "static_events" in op.attrs:
            from ..analysis.cross_loop_scheduler import StaticU32Expression
            from ..tile_codegen import static_publication_ptx

            template = static_publication_ptx(
                cast(tuple[StaticU32Expression, ...], op.attrs["static_events"])
            )
        if (
            op.attrs["action"] == "wait"
            and isinstance(op.operands[1].definition, IndexConstantOp)
            and op.operands[1].definition.value == 0
        ):
            template = """
                setp.eq.u32 ${pred:no_fence}, ${b32:operand.expected.val}, 0;
                @${pred:no_fence} bra ${label:done};
                fence.proxy.async.global;
                ${label:done}:
            """
        if (
            op.attrs["action"] == "publish"
            and isinstance(op.operands[2].definition, IndexConstantOp)
            and op.operands[2].definition.value == 0
        ):
            template = "cp.async.bulk.wait_group 0;\nbar.sync 0;\nbar.sync 0;\n"
        # Ordinary scalar loads/stores use the generic proxy. They still need
        # the CTA join and release/acquire chain, but neither a TMA store drain
        # nor a generic-to-async proxy fence.
        if op.attrs.get("async_proxy") is False:
            template = template.replace("fence.proxy.async.global;", "")
            template = template.replace("cp.async.bulk.wait_group 0;", "")
        if op.attrs["action"] == "publish" and op.attrs.get("join_after") is False:
            assert template.rstrip().endswith("bar.sync 0;")
            template = template.rstrip()[: -len("bar.sync 0;")] + "\n"
        if "active_workers" in op.attrs:
            template = (
                "mov.u32 ${b32:publication_worker}, %ctaid.x;\n"
                f"setp.ge.u32 ${{pred:inactive_worker}}, ${{b32:publication_worker}}, {op.attrs['active_workers']};\n"
                "@${pred:inactive_worker} bra ${label:inactive_done};\n"
                + template + "\n${label:inactive_done}:\n"
            )
        if op.attrs.get("inactive_workers"):
            guard = "mov.u32 ${b32:publication_worker}, %ctaid.x;\n"
            for worker in cast(tuple[int, ...], op.attrs["inactive_workers"]):
                guard += (
                    f"setp.eq.u32 ${{pred:inactive_lane}}, ${{b32:publication_worker}}, {worker};\n"
                    "@${pred:inactive_lane} bra ${label:inactive_done};\n"
                )
            template = guard + template + "\n${label:inactive_done}:\n"
        return [PTXTemplateOp.build(
            [cast(Value, self.value_map[value]) for value in op.operands], [], template,
            operand_names=names, description=f"tile {op.attrs['action']}",
            attrs={"readiness_root": op.attrs["readiness_root"]} if "readiness_root" in op.attrs else None,
        )], []

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
        self.lower_block(op.body)
        return [op], []

    def _leader_warp(self, prefix: str) -> tuple[list[Value], list[str], str]:
        """Operands and PTX setting ``${pred:<prefix>_other_warp}`` outside warp 0.

        Warp 0 holds the leader, thread 0. The kernel's warp index is broadcast
        from each warp's first thread, so ptxas proves this predicate
        warp-uniform.
        """
        assert self.warp_ids is not None, "lower_kernel creates warp_ids"
        return (
            list(self.warp_ids.results),
            ["warp_x", "warp_y", "warp_z"],
            (
                f"or.b32 ${{b32:{prefix}_warp}}, ${{b32:operand.warp_x.val}}, "
                "${b32:operand.warp_y.val};\n"
                f"or.b32 ${{b32:{prefix}_warp}}, ${{b32:{prefix}_warp}}, "
                "${b32:operand.warp_z.val};\n"
                f"setp.ne.u32 ${{pred:{prefix}_other_warp}}, ${{b32:{prefix}_warp}}, 0;\n"
            ),
        )

    def _leader_thread(
        self,
        prefix: str,
        before_branch: str = "",
    ) -> tuple[list[Value], list[str], str]:
        """Operands and PTX that let one elected thread of warp 0 do leader work.

        Every other warp branches to ``${label:<prefix>_done}``, without
        diverging inside a warp, and ``elect.sync`` sets
        ``${pred:<prefix>_leader}`` on one thread of warp 0, so ptxas can keep
        the work's operands in uniform registers. ``before_branch`` runs in
        every warp, so ptxas can keep its special-register reads out of loops.
        """
        operands, names, leader_warp = self._leader_warp(prefix)
        return (
            operands,
            names,
            leader_warp
            + before_branch
            + (
                f"@${{pred:{prefix}_other_warp}} bra ${{label:{prefix}_done}};\n"
                f"elect.sync _|${{pred:{prefix}_leader}}, 0xffffffff;\n"
            ),
        )

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
            BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
            view_root_type.dtype,
        ).byte_size
        result_type = op.result().type
        assert isinstance(result_type, AddrMemTensorType)
        result_sizes = cast(tuple[int, ...], op.index_steps)
        layout = (
            physical_layout(view_root_type)
            if isinstance(view_root_type, SmemTensorType)
            else None
        )
        # Blocks split the contiguous dimension; the other one is the row.
        inner_dim = 0 if is_column_major(view_root_type) else 1
        outer_rows = view_root_type.sizes[1 - inner_dim]

        # Accumulate byte offset for each dimension.
        zero = IndexConstantOp.build(0)
        new_ops.append(zero)
        byte_offset = zero.result()

        for dim, idx in enumerate(op.indices):
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
                    block, inner = divmod(
                        idx * result_sizes[dim],
                        block_width,
                    )
                    byte_delta = (
                        block * cast(int, outer_rows) * block_width + inner
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

    @lowers(AllocOp)
    def lower_alloc(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        if isinstance(result_type, SmemTensorType):
            return self._lower_alloc_smem(op)
        if isinstance(result_type, (RmemTensorType, WrmemTensorType, WgrmemTensorType)):
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

    def _lower_alloc_lock(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        assert isinstance(lock_type, LockType)
        if lock_type.kind == "cluster_mbar":
            return self._lower_alloc_cluster_mbar(op)
        if lock_type.kind != "mbar":
            return [], [None]
        # InsertSyncs represents publication explicitly with BarrierOps.
        copies = _warpgroup_copies(op)
        count = op.attrs.get("array_count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("mbarrier array_count must be a positive integer")
        template = textwrap.dedent(f"""\
            .shared .align 8 .b64 ${{smem:bar}}[{copies * count}];
            mov.u32 ${{b32:result.dst.val}}, ${{smem:bar}};
            {_copy_offset("result.dst.val", 8 * count, copies)}
            mov.u32 ${{b32:result.dst.parity}}, 0;
            """) + textwrap.dedent("""\
            mov.u32 ${b32:tid_x}, %tid.x;
            mov.u32 ${b32:tid_y}, %tid.y;
            or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_y};
            mov.u32 ${b32:tid_z}, %tid.z;
            or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_z};
            setp.ne.s32 ${pred:not_t0}, ${b32:tid_x}, 0;
            @${pred:not_t0} bra ${label:after_init};
            INIT_ARRAY
            fence.mbarrier_init.release.cluster;
            ${label:after_init}:
        """)
        template = template.replace("INIT_ARRAY", "\n".join(
            f"mbarrier.init.shared::cta.b64 [${{b32:result.dst.val}}+{8 * i}], 1;"
            for i in range(count)
        ))
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[lock_type],
            template=template,
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_cluster_mbar(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
        if _split_release(op.result()) and not op.attrs.get("single_issuer"):
            # Striped multicast has issuers in multiple CTAs. Each issuer
            # must wait for every recipient's readers before writing its slice.
            assert kernel.num_threads is not None
            arrivals = kernel.num_threads // 32 * cluster_size
            template = textwrap.dedent(f"""\
                .shared .align 8 .b64 ${{smem:bar}}[1];
                mov.u32 ${{b32:result.dst.val}}, ${{smem:bar}};
                mov.u32 ${{b32:result.dst.parity}}, 0;
                mov.u32 ${{b32:tid_x}}, %tid.x;
                mov.u32 ${{b32:tid_y}}, %tid.y;
                or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_y}};
                mov.u32 ${{b32:tid_z}}, %tid.z;
                or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_z}};
                setp.ne.s32 ${{pred:not_t0}}, ${{b32:tid_x}}, 0;
                @${{pred:not_t0}} bra ${{label:after_init}};
                mbarrier.init.shared::cta.b64 [${{b32:result.dst.val}}], {arrivals};
                fence.mbarrier_init.release.cluster;
                ${{label:after_init}}:
            """)
            new_op = PTXTemplateOp.build(
                operands=[],
                result_types=[lock_type],
                template=template,
                operand_names=[],
                result_names=["dst"],
                description=str(op),
            )
            return [new_op], [new_op.result()]
        # InsertSyncs represents publication explicitly with BarrierOps.
        assert kernel.num_threads is not None
        warps = kernel.num_threads // 32
        # Every warp of the CTA arrives, and at the cluster leader every warp
        # of the cluster. InsertSyncs represents publication explicitly with
        # BarrierOps.
        template = textwrap.dedent(f"""\
            .shared .align 8 .b64 ${{smem:bar}}[1];
            mov.u32 ${{b32:result.dst.val}}, ${{smem:bar}};
            mov.u32 ${{b32:result.dst.parity}}, 0;
            mov.u32 ${{b32:tid_x}}, %tid.x;
            mov.u32 ${{b32:tid_y}}, %tid.y;
            or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_y}};
            mov.u32 ${{b32:tid_z}}, %tid.z;
            or.b32 ${{b32:tid_x}}, ${{b32:tid_x}}, ${{b32:tid_z}};
            setp.ne.s32 ${{pred:not_t0}}, ${{b32:tid_x}}, 0;
            @${{pred:not_t0}} bra ${{label:after_init}};
            mov.u32 ${{b32:cluster_rank}}, %cluster_ctarank;
            setp.eq.u32 ${{pred:rank0}}, ${{b32:cluster_rank}}, 0;
            mov.u32 ${{b32:arrivals}}, {warps};
            @${{pred:rank0}} mov.u32 ${{b32:arrivals}}, {warps * cluster_size};
            mbarrier.init.shared::cta.b64 [${{b32:result.dst.val}}], ${{b32:arrivals}};
            fence.mbarrier_init.release.cluster;
            ${{label:after_init}}:
        """)
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[lock_type],
            template=template,
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

    def _lower_alloc_registers(
        self,
        op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[op.result().type],
            template="",
            operand_names=[],
            result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

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
                BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
                smem_type.dtype,
            ).byte_size
        )
        # TMA swizzling and the corresponding software address swizzle operate
        # within 1 KiB regions and require the shared-memory base to match.
        alignment = (
            1024
            if smem_type.swizzle is not None and (
                physical_layout(smem_type) is not None
                or smem_type.attrs.get("scope") == "warp"
            )
            else 128
        )
        alignment = op.attrs.get("allocation_alignment", alignment)
        copies = 1 if op.attrs.get("warpgroup_shared") else _warpgroup_copies(op)
        stride = -(-n_bytes // alignment) * alignment
        template = textwrap.dedent(f"""\
            .shared .align {alignment} .b8 ${{smem:sym}}[{stride * (copies - 1) + n_bytes}];
            mov.u32 ${{b32:result.dst.ptr}}, ${{smem:sym}};
            {_copy_offset("result.dst.ptr", stride, copies)}
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
        sources = [op.src(i) for i in range(op.num_srcs)]
        dst_value = cast(Value, self.value_map[op.dst])
        source_values = [cast(Value, self.value_map[src]) for src in sources]

        for value in op.operands:
            if not isinstance(value.type, (RmemTensorType, WrmemTensorType, WgrmemTensorType)):
                raise NotImplementedError(
                    f"lower_elementwise: operands must be registers; got {value.type}"
                )
        lower = _ELEMENTWISE_LOWERERS.get(op.fn)
        if lower is None:
            raise NotImplementedError(f"lower_elementwise: unsupported fn {op.fn!r}")
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

    @lowers(LoadIndexOp)
    def lower_load_index(self, op: LoadIndexOp) -> tuple[list[Op], list[object]]:
        lowered = PTXTemplateOp.build(
            [cast(Value, self.value_map[op.src])], [op.result().type],
            template=("ld.global.s64 ${b64:raw}, [${b64:operand.src.ptr}];\n"
                      "setp.lt.s64 ${pred:negative}, ${b64:raw}, 0;\n"
                      f"@${{pred:negative}} add.s64 ${{b64:raw}}, ${{b64:raw}}, {op.attrs['extent']};\n"
                      "cvt.u32.u64 ${b32:result.index.val}, ${b64:raw};\n"),
            operand_names=["src"], result_names=["index"], description=str(op),
        )
        return [lowered], [lowered.result()]

    @lowers(RegisterViewOp)
    def lower_register_view(self, op: RegisterViewOp) -> tuple[list[Op], list[object]]:
        return [], [self.value_map[op.arg]]

    @lowers(ShuffleOp)
    def lower_shuffle(self, op: ShuffleOp) -> tuple[list[Op], list[object]]:
        [source] = _register_operand_regs(op.src, 'src')[1]
        lowered = PTXTemplateOp.build(
            [cast(Value, self.value_map[op.src])], [op.result().type],
            template=f"shfl.sync.{op.attrs['mode']}.b32 ${{f32:result.dst.r0}}, {source}, {op.attrs['offset']}, 31, 0xffffffff;\n",
            operand_names=['src'], result_names=['dst'], description=str(op))
        return [lowered], [lowered.result()]

    @lowers(IndexEqOp)
    def lower_index_eq(self, op: IndexEqOp) -> tuple[list[Op], list[object]]:
        lowered = PTXTemplateOp.build(
            [cast(Value, self.value_map[v]) for v in op.operands], [PredicateType()],
            template='setp.eq.u32 ${pred:result.eq.val}, ${b32:operand.lhs.val}, ${b32:operand.rhs.val};\n',
            operand_names=['lhs', 'rhs'], result_names=['eq'], description=str(op))
        return [lowered], [lowered.result()]

    @lowers(ReduceOp)
    def lower_reduce(self, op: ReduceOp) -> tuple[list[Op], list[object]]:
        raise NotImplementedError('decompose reductions with LowerToScalar before PTX lowering')

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
            raise NotImplementedError(f"lower_async: tma_move with lock kind {kind!r}")
        mms = [body_op for body_op in body_ops if isinstance(body_op, MatmulOp)]
        if body_ops and len(mms) == len(body_ops):
            return self._lower_async_matmul_wgmma(op, mms)
        raise NotImplementedError(
            f"lower_async: unsupported body ops {[type(o).__name__ for o in body_ops]}"
        )

    def _lower_async_matmul_wgmma(
        self,
        op: AsyncOp,
        mms: list[MatmulOp],
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
                or tile.index_steps != base_tile.index_steps
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
                indices,
                base_indices,
                tile.index_steps,
                typ.strides,
            ):
                assert index is not None and base_index is not None
                assert size is not None and stride is not None
                delta += (index - base_index) * size * stride
            byte_delta = (
                delta
                * cast(
                    BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
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

        def _desc(name, base, delta, value, hint, const, offset=None):
            if delta is not None:
                return f"add.u64 ${{b64:{name}}}, ${{b64:{base}}}, {delta // 16};\n"
            operand = _operand(cast(Value, self.value_map[value]), hint)
            return _build_desc(
                name,
                f"operand.{operand}.ptr",
                const,
                offset,
            )

        streamed_template = ""
        for j, mm in enumerate(mms):
            desc_template = ""
            mma_template = ""
            lhs_type = mm.lhs.type
            rhs_type = mm.rhs.type
            dst_type = mm.dst.type
            assert (
                isinstance(lhs_type, SmemTensorType)
                and isinstance(rhs_type, SmemTensorType)
                and isinstance(dst_type, WgrmemTensorType)
                and dst_type.warps == 4
            ), (
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
            fp8 = lhs_type.dtype.name in ("e4m3", "e5m2")
            mma_k = 32 if fp8 else 16
            assert k_dim == mma_k, f"_lower_async_matmul_wgmma: k must be {mma_k}; got {k_dim}"
            if dst_type.dtype != F32Type():
                raise NotImplementedError("WGMMA accumulation requires f32")
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
                    BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
                    lhs_type.dtype,
                ).byte_size
            )
            rhs_row_stride_bytes = (
                cast(int, rhs_row_stride)
                * cast(
                    BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
                    rhs_type.dtype,
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
                lead_dim, lhs_row_stride_bytes * 8, _swizzle_to_bits[lhs_type.swizzle]
            )
            desc_const_rhs = _build_static_desc(
                lead_dim, rhs_row_stride_bytes * 8, _swizzle_to_bits[rhs_type.swizzle]
            )

            lhs_delta = _tile_delta(mm.lhs, mms[0].lhs, j)
            # A commit group may contain independent projections. Reuse each
            # RHS's own first descriptor, including after the operand changes.
            rhs_base = next(
                (i for i in range(j) if _tile_delta(mm.rhs, mms[i].rhs, j - i) is not None),
                j,
            )
            rhs_delta = _tile_delta(mm.rhs, mms[rhs_base].rhs, j - rhs_base)
            dst_name = _operand(cast(Value, self.value_map[mm.dst]), "dst")

            desc_rhs = f"desc_rhs{j}"
            desc_template += _desc(
                desc_rhs,
                f"desc_rhs{rhs_base}",
                rhs_delta,
                mm.rhs,
                "rhs",
                desc_const_rhs,
            )
            for m_it in range(m_iters):
                lhs_offset = m_it * 64 * lhs_row_stride_bytes
                desc_lhs = f"desc_lhs{j}_{m_it}"
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
            rhs_dtype = rhs_type.dtype.name
            if not ((fp8 and rhs_dtype in ("e4m3", "e5m2"))
                    or (not fp8 and input_dtype in ("bf16", "f16") and rhs_dtype == input_dtype)):
                raise NotImplementedError(
                    "wgmma inputs must have the same dtype; got "
                    f"{lhs_type.dtype} and {rhs_type.dtype}"
                )
            # Static destination views select storage in the root fragment.
            dst_registers = _register_operand_regs(mm.dst, dst_name)[1]
            assert len(dst_registers) == m_iters * num_regs_per_wgmma
            transpose_args = "" if fp8 else ", 0, 0"
            reg_offset = 0
            for m_it in range(m_iters):
                regs = ", ".join(
                    dst_registers[reg_offset + i]
                    for i in range(num_regs_per_wgmma)
                )
                mma_template += textwrap.dedent(f"""\
                    wgmma.mma_async.sync.aligned.m64n{n_dim}k{mma_k}.f32.{input_dtype}.{rhs_dtype}
                      {{{regs}}},
                      ${{b64:desc_lhs{j}_{m_it}}},
                      ${{b64:{desc_rhs}}},
                      {int(mm.accumulate)}, 1, 1{transpose_args};
                """)
                reg_offset += num_regs_per_wgmma
            streamed_template += desc_template + mma_template

        # One commit group for the whole batch: single fence, all mma_async, one
        # commit. Hardware serializes same-accumulator mmas, so the await's lone
        # wait_group covers them all. The await also emits an asymmetric named
        # barrier that closes the cross-warp WAR (thread-0 TMA reload vs
        # warpgroup-wide wgmma reads).
        template = (
            "wgmma.fence.sync.aligned;\n"
            + streamed_template
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
        self,
        op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower synchronous MoveOp to PTX."""
        dst_type = op.dst.type
        src_type = op.src.type
        if isinstance(dst_type, RegMemTensorType) and isinstance(src_type, RegMemTensorType):
            dsuffix, dregs = _register_operand_regs(op.dst, "dst")
            ssuffix, sregs = _register_operand_regs(op.src, "src")
            if dst_type != src_type or dsuffix != ssuffix or len(dregs) != len(sregs):
                raise NotImplementedError("register moves require identical ownership and dtype")
            template = "".join(f"mov.{dsuffix} {d}, {s};\n" for d, s in zip(dregs, sregs))
            return [PTXTemplateOp.build(
                [cast(Value, self.value_map[op.dst]), cast(Value, self.value_map[op.src])],
                [], template=template, operand_names=["dst", "src"], description=str(op),
            )], []
        if any(isinstance(t, WrmemTensorType) or (isinstance(t, WgrmemTensorType) and t.layout == "wgmma_row") for t in (dst_type, src_type)):
            raise NotImplementedError('decompose distributed moves with LowerToScalar before PTX lowering')

        if {type(dst_type), type(src_type)} == {RmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_rmem(op)
        if {type(dst_type), type(src_type)} == {RmemTensorType, SmemTensorType}:
            return self._lower_move_smem_rmem(op)
        if {type(dst_type), type(src_type)} == {SmemTensorType, GmemTensorType}:
            return self._lower_move_gmem_smem(op)
        if (
            isinstance(dst_type, WgrmemTensorType)
            and isinstance(src_type, AddrMemTensorType)
            and dst_type.layout in ("wgmma_layout_0", "wgmma_warp")
        ):
            return self._lower_move_wgrmem_wgmma(op)
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
            and src_type.layout in ("wgmma_layout_0", "wgmma_warp")
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
        if kind == "mbar":
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
        if scope == "warp":
            template = "bar.warp.sync 0xffffffff;\n"
        elif scope == "block":
            drain = "cp.async.bulk.wait_group.read 0;\n" if op.attrs.get("drain_bulk") else ""
            template = drain + "bar.sync 0;\n"
        elif scope == "cluster_to_leader" and op.attrs.get("phase") == "arrive":
            # Signal each possible remote writer, then continue. A single
            # multicast issuer only needs arrivals at rank 0 and the local
            # CTA; striped multicast needs them at all issuing CTAs.
            assert op.sync is not None
            if (
                isinstance(op.sync.definition, AllocOp)
                and op.sync.definition.attrs.get("single_issuer") is True
            ):
                template = _CLUSTER_RELEASE_ARRIVAL
                return [
                    PTXTemplateOp.build(
                        [cast(Value, self.value_map[op.sync])], [],
                        template=template, operand_names=["sync"], description=str(op),
                    )
                ], []
            kernel = find_ancestor_of_type(op, KernelOp)
            assert kernel is not None
            cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
            template = (
                "bar.warp.sync 0xffffffff;\n"
                "elect.sync _|${pred:release}, 0xffffffff;\n"
                + "".join(
                    textwrap.dedent(f"""\
                        mov.u32 ${{b32:rank}}, {rank};
                        mapa.shared::cluster.u32 ${{b32:remote}}, ${{b32:operand.sync.val}}, ${{b32:rank}};
                        @${{pred:release}} mbarrier.arrive.shared::cluster.b64 _, [${{b32:remote}}];
                        """)
                    for rank in range(cluster_size)
                )
            )
            operands = [cast(Value, self.value_map[op.sync])]
            operand_names = ["sync"]
        elif scope == "cluster_to_leader" and op.attrs.get("phase") == "wait":
            # Only the leader's warp, which issues the next write, waits for
            # every warp of the cluster.
            assert op.sync is not None
            template = _leader_warp("cl") + textwrap.dedent("""\
                @${pred:cl_other_warp} bra ${label:cl_done};
                mov.u32 ${b32:ticks}, 0x989680;
                ${label:cl_wait}:
                mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ${pred:ready}, [${b32:operand.sync.val}], ${b32:operand.sync.parity}, ${b32:ticks};
                @!${pred:ready} bra ${label:cl_wait};
                xor.b32 ${b32:operand.sync.parity}, ${b32:operand.sync.parity}, 1;
                ${label:cl_done}:
                """)
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
            arrive = (
                "relaxed" if op.attrs.get("initialization_publication") else "release"
            )
            template = textwrap.dedent(f"""\
                barrier.cluster.arrive.{arrive};
                barrier.cluster.wait.acquire;
                """)
        elif scope == "block_leader":
            # Warp 0, which holds the leader t0, waits; the other warps arrive
            # and continue, on the named barrier id chosen by insert_syncs
            # (unique among concurrently-live block-leader barriers). Warps
            # are told apart by their warp-uniform index, and arrive first so
            # an arrival never waits behind a skipped bar.sync.
            bar_id = op.attrs.get("id", 1)
            num_threads = op.attrs["num_threads"]
            template = _leader_warp("bl") + textwrap.dedent(f"""\
                @${{pred:bl_other_warp}} bar.arrive {bar_id}, {num_threads};
                @!${{pred:bl_other_warp}} bar.sync {bar_id}, {num_threads};
                """)
        elif scope == "grid":
            assert op.sync is not None
            ancestor = find_ancestor_of_type(op, KernelOp)
            assert ancestor is not None
            grid_size = math.prod(
                dim if dim is not None else 1 for dim in ancestor.grid_dim
            )
            sync = cast(Value, self.value_map[op.sync])
            drain = "cp.async.bulk.wait_group 0;\n" if op.attrs.get("drain_bulk") else ""
            template = drain + textwrap.dedent(f"""\
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
        self,
        op: AwaitOp,
        lock: Value,
    ) -> tuple[list[Op], list[object]]:
        # Materializing the optional hint in a virtual register lets ptxas
        # select the compact fused wait; an immediate or test_wait expands
        # into substantially more phase-check, sleep, or yield instructions.
        multicast_async = lock_has_tma_multicast(op.lock)
        acquire_scope = "cluster" if multicast_async else "cta"
        template = textwrap.dedent(rf"""
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
        self,
        op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        # wait_group drains the commit group so the warpgroup's wgmma reads of
        # the smem operands have retired. The buffer-reuse WAR (thread-0's TMA
        # reload vs those reads) is handled by insert_syncs, which emits a barrier
        # before the reload — so no barrier is needed here.
        pending = cast(int, op.attrs.get("pending", 0))
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
        pending = cast(int, op.attrs.get("pending", 0))
        template = f"\ncp.async.bulk.wait_group.read {pending};\n"
        if op.lock.type.attrs.get("scope") == "warp":
            # Bulk groups belong to the issuing lane. Rendezvous after every
            # lane's wait before any lane can overwrite the shared source.
            template += "bar.warp.sync 0xffffffff;\n"
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
        register = op.dst if is_load else op.src
        rmem_type = register.type
        assert isinstance(gmem_ir_type, GmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        gmem_label = "src" if is_load else "dst"
        dtype_bytes = cast(
            BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
            gmem_ir_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        registers = _register_operand_regs(register, "dst" if is_load else "src")[1]
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
                    f"{registers[flat_idx]}, "
                    f"[{base_reg}+{offset}];\n"
                )
            else:
                instruction = 'red.relaxed.gpu.global.add' if op.attrs.get('atomic_add') else 'st.global'
                template += (
                    f"{instruction}.{suffix} "
                    f"[{base_reg}+{offset}], "
                    f"{registers[flat_idx]};\n"
                )

        vector_width = op.attrs.get("vector_width", 1)
        if (vector_width in (2, 4) and vector_width * dtype_bytes <= 16
                and len(registers) == vector_width and gmem_ir_type.strides[-1] == 1
                and not op.attrs.get("atomic_add")):
            packed = "{" + ", ".join(registers) + "}"
            instruction = f"ld.global{'.nc' if op.attrs.get('nc') else ''}" if is_load else "st.global"
            args = f"{packed}, [{base_reg}]" if is_load else f"[{base_reg}], {packed}"
            predicate = "${pred:vector_aligned}"
            alignment = vector_width * dtype_bytes
            vector_template = f"{instruction}.v{vector_width}.{suffix} {args};\n"
            template = (
                f"and.b64 ${{b64:vector_alignment}}, {base_reg}, {alignment - 1};\n"
                f"setp.eq.u64 {predicate}, ${{b64:vector_alignment}}, 0;\n"
                f"@!{predicate} bra ${{label:vector_scalar}};\n"
                f"{instruction}.v{vector_width}.{suffix} {args};\n"
                "bra ${label:vector_done};\n${label:vector_scalar}:\n"
                + template + "${label:vector_done}:\n"
            )
            if cast(int, op.attrs.get("alignment", 0)) >= alignment:
                template = vector_template
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
            BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
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

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

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
        assert wgrmem_type.layout in ("wgmma_layout_0", "wgmma_warp")
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
        assert m_dim % (8 if wgrmem_type.warps == 1 else 64) == 0, f"unsupported fragment rows: {m_dim}"
        assert root_n % 8 == 0 and 8 <= root_n <= 256, (
            f"wgmma store: n must be a multiple of 8 in [8, 256]; got {root_n}"
        )
        assert isinstance(memory_type.dtype, (BF16Type, F16Type, F32Type, F64Type))
        dtype_bytes = memory_type.dtype.byte_size
        row_stride_bytes = cast(int, memory_type.strides[0]) * dtype_bytes
        column_stride_bytes = cast(int, memory_type.strides[1]) * dtype_bytes
        column_broadcast = is_load and column_stride_bytes == 0
        assert column_broadcast or (
            column_stride_bytes == dtype_bytes
            and row_stride_bytes % (2 * dtype_bytes) == 0
        ), "wgmma element pairs must be contiguous and aligned"
        assert not (
            isinstance(memory_type, SmemTensorType)
            and physical_layout(memory_type) is not None
        ), f"wgmma move: strided addressing cannot store {memory_type}"

        new_ops, byte_offset = _build_register_layout_dynamic_offset(
            wgrmem_type.register_layout(fragment),
            (row_stride_bytes, column_stride_bytes),
        )
        column_bounds: list[tuple[Value, int]] = []
        if is_load and isinstance(memory_type, GmemTensorType) and not column_broadcast:
            # The fragment can cover the last, incomplete tile of an input
            # view. TMA masks the GEMM operands, but these epilogue loads use
            # ordinary global instructions and need their own column guard.
            current = memory
            tiles: list[TileOp] = []
            while isinstance(current.definition, TileOp):
                tile = current.definition
                parent_type = cast(MemTensorType, tile.arg.type)
                current_type = cast(MemTensorType, current.type)
                if parent_type.rank != 2 or current_type.rank != 2:
                    break
                tiles.append(tile)
                extent = parent_type.static_shape[1]
                if extent % current_type.static_shape[1]:
                    if extent % 2:
                        raise NotImplementedError("partial fragment loads require complete element pairs")
                    constant = sum(
                        index * cast(MemTensorType, view.result().type).static_shape[1]
                        for view in tiles
                        if isinstance(index := view.indices[1], int)
                    )
                    start = IndexConstantOp.build(constant)
                    new_ops.append(start)
                    base = start.result()
                    for view in tiles:
                        index = view.indices[1]
                        if isinstance(index, Value):
                            size = IndexConstantOp.build(cast(MemTensorType, view.result().type).static_shape[1])
                            scaled = IndexMulOp.build(cast(Value, self.value_map.get(index, index)), size.result())
                            added = IndexAddOp.build(base, scaled.result())
                            new_ops.extend([size, scaled, added])
                            base = added.result()
                    column_bounds.append((base, extent))
                current = tile.arg
        column_offset = None
        if column_bounds:
            column_ops, column_offset = _build_register_layout_dynamic_offset(
                wgrmem_type.register_layout(fragment), (0, 1),
            )
            new_ops.extend(column_ops)
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
            template += textwrap.dedent("""\
                mov.u32 ${b32:store_tid}, %tid.x;
                shr.u32 ${b32:store_wg}, ${b32:store_tid}, 7;
            """)
            if "writer_wg_divisor" in op.attrs:
                template += f"div.u32 ${{b32:store_wg}}, ${{b32:store_wg}}, {op.attrs['writer_wg_divisor']};\n"
            if "writer_wg_modulus" in op.attrs:
                template += f"rem.u32 ${{b32:store_wg}}, ${{b32:store_wg}}, {op.attrs['writer_wg_modulus']};\n"
            template += f"setp.eq.u32 ${{pred:store_writer}}, ${{b32:store_wg}}, {writer_wg};\n"
            predicate = "@${pred:store_writer} "
        # A pair is one packed BF16/F16 register or two F32 registers.
        f32 = isinstance(wgrmem_type.dtype, F32Type)
        instruction = f"{'ld' if is_load else 'st'}.{space}{'.v2' if f32 else ''}.b32"

        # Broadcast columns load one scalar per distinct row and duplicate its
        # bits into the fragment pair. Other columns reuse that register value.
        # This reads the original strided view; it does not materialize a bias.
        loaded_rows: dict[int, str] = {}
        column_predicates: dict[int, str] = {}
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
            registers = (
                f"{{${{f32:operand.fragment.r{2 * pair}}}, "
                f"${{f32:operand.fragment.r{2 * pair + 1}}}}}"
                if f32
                else f"${{b32:operand.fragment.r{pair}}}"
            )
            location = f"[{address}+{offset}]"
            load_predicate = ""
            if column_bounds:
                if dj not in column_predicates:
                    valid = f"${{pred:column_valid{dj}}}"
                    template += f"add.u32 ${{b32:column{dj}}}, ${{b32:operand.column.val}}, {dj + 1};\n"
                    for bound, (_, extent) in enumerate(column_bounds):
                        test = valid if bound == 0 else f"${{pred:column_bound{dj}_{bound}}}"
                        template += (
                            f"add.u32 ${{b32:column_at{dj}_{bound}}}, ${{b32:column{dj}}}, ${{b32:operand.bound{bound}.val}};\n"
                            f"setp.lt.u32 {test}, ${{b32:column_at{dj}_{bound}}}, {extent};\n"
                            + ("" if bound == 0 else f"and.pred {valid}, {valid}, {test};\n")
                        )
                    column_predicates[dj] = valid
                load_predicate = f"@{column_predicates[dj]} "
                if f32:
                    for element in (0, 1):
                        template += f"mov.b32 ${{f32:operand.fragment.r{2 * pair + element}}}, 0;\n"
                else:
                    template += f"mov.b32 ${{b32:operand.fragment.r{pair}}}, 0;\n"
            if op.attrs.get("atomic_add") is True:
                if is_load or not isinstance(memory_type, GmemTensorType) or not f32:
                    raise ValueError("atomic writeback requires an F32 register-to-global move")
                for element in (0, 1):
                    template += (
                        f"{predicate}red.relaxed.gpu.global.add.f32 "
                        f"[{address}+{offset + element * dtype_bytes}], "
                        f"${{f32:operand.fragment.r{2 * pair + element}}};\n"
                    )
                continue
            template += (
                f"{load_predicate}{instruction} {registers}, {location};\n"
                if is_load
                else f"{predicate}{instruction} {location}, {registers};\n"
            )

        new_op = PTXTemplateOp.build(
            operands=[
                cast(Value, self.value_map[memory]),
                cast(Value, self.value_map[fragment]),
                byte_offset,
                *([] if column_offset is None else [column_offset]),
                *(base for base, _ in column_bounds),
            ],
            result_types=[],
            template=template,
            operand_names=["memory", "fragment", "offset", *([] if column_offset is None else ["column"]),
                           *(f"bound{index}" for index in range(len(column_bounds)))],
            result_names=[],
            description=str(op),
        )
        new_ops.append(new_op)
        return new_ops, []

    def _lower_move_wgrmem_wgmma_to_stmatrix(
        self,
        op: MoveOp,
        register_starts: list[int],
    ) -> tuple[list[Op], list[object]]:
        """Use STMatrix/LDMatrix with the same swizzle and fragment mapping."""
        is_load = isinstance(op.dst.type, WgrmemTensorType)
        fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
        dst = cast(Value, self.value_map[memory])
        src = cast(Value, self.value_map[fragment])
        dst_type = memory.type
        src_type = fragment.type
        assert isinstance(dst_type, SmemTensorType)
        assert isinstance(src_type, WgrmemTensorType)
        col_start = (
            _static_view_origin(fragment)[1]
            if isinstance(fragment.definition, TileOp)
            else op.attrs.get("wgmma_col_start", 0)
        )
        assert isinstance(col_start, int)
        src_root = root_buffer(fragment)
        dst_root = _stmatrix_layout_root(memory)
        assert src_root is not None and dst_root is not None
        src_root_type = src_root.type
        assert isinstance(src_root_type, WgrmemTensorType)
        n_dim = cast(int, src_root_type.sizes[1])
        root_type = dst_root.type
        assert isinstance(root_type, SmemTensorType)
        root_sizes = cast(tuple[int, ...], root_type.sizes)
        transposed = is_column_major(root_type)
        layout = physical_layout(dst_type)
        # Dense row-major staging needs no physical blocking. Its full row
        # width gives the same affine STMatrix addresses and a 2D TMA store.
        block_width = layout[0] if layout is not None else root_sizes[1]
        if src_type.layout == "wgmma_warp" and dst_type.swizzle:
            block_width = cast(int, root_type.strides[-2])
        # TODO: Model writer ownership as a warpgroup-uniform IR condition so
        # loop peeling/unrolling can specialize each slice and lower the whole
        # gather to one branch instead of predicating every stmatrix.
        writer_wg = op.attrs.get("writer_wg") if not is_load else None
        predicate = ""
        writer_setup = ""
        if writer_wg is not None:
            predicate = "@${pred:st_writer} "
            writer_setup = textwrap.dedent("""\
                mov.u32 ${b32:st_full_tid}, %tid.x;
                shr.u32 ${b32:st_wg}, ${b32:st_full_tid}, 7;
            """)
            if "writer_wg_divisor" in op.attrs:
                writer_setup += f"div.u32 ${{b32:st_wg}}, ${{b32:st_wg}}, {op.attrs['writer_wg_divisor']};\n"
            if "writer_wg_modulus" in op.attrs:
                writer_setup += f"rem.u32 ${{b32:st_wg}}, ${{b32:st_wg}}, {op.attrs['writer_wg_modulus']};\n"
            writer_setup += f"setp.eq.u32 ${{pred:st_writer}}, ${{b32:st_wg}}, {writer_wg};\n"
        # Lane l addresses row l % 8 of matrix l / 8. Untransposed, the x4
        # matrices sit side by side along the fragment row. Transposed, they
        # are the 2x2 matrices of a 16x16 fragment block, and a stored row is
        # a fragment column: lane l writes physical row
        # (l & 7) + 8 * (l >> 4) at column 16 * warp + 8 * ((l >> 3) & 1).
        lane_coordinates = (
            textwrap.dedent("""\
            shr.u32 ${b32:st_hi}, ${b32:st_lane}, 4;
            mad.lo.u32 ${b32:st_row}, ${b32:st_hi}, 8, ${b32:st_row};
            shr.u32 ${b32:st_col}, ${b32:st_lane}, 3;
            and.b32 ${b32:st_col}, ${b32:st_col}, 1;
            shl.b32 ${b32:st_col}, ${b32:st_col}, 3;
            mad.lo.u32 ${b32:st_col}, ${b32:st_warp}, 16, ${b32:st_col};
        """)
            if transposed
            else textwrap.dedent("""\
            mad.lo.u32 ${b32:st_row}, ${b32:st_warp}, 16, ${b32:st_row};
            shr.u32 ${b32:st_col}, ${b32:st_lane}, 3;
            shl.b32 ${b32:st_col}, ${b32:st_col}, 3;
        """)
        )
        if src_type.layout == "wgmma_warp":
            lane_coordinates = lane_coordinates.replace("mad.lo.u32 ${b32:st_row}, ${b32:st_warp}, 16, ${b32:st_row};", "")
        template = (
            writer_setup
            + textwrap.dedent("""\
            mov.u32 ${b32:st_lane}, %laneid;
            mov.u32 ${b32:st_tid}, %tid.x;
            and.b32 ${b32:st_tid}, ${b32:st_tid}, 127;
            shr.u32 ${b32:st_warp}, ${b32:st_tid}, 5;
            and.b32 ${b32:st_row}, ${b32:st_lane}, 7;
        """)
            + lane_coordinates
        )
        if layout is not None and block_width < 32:
            # A lane's matrix starts up to 24 columns into the 32 an x4
            # covers, so narrow blocks put it in a later block.
            template += textwrap.dedent(f"""\
                shr.u32 ${{b32:st_block}}, ${{b32:st_col}}, {block_width.bit_length() - 1};
                and.b32 ${{b32:st_col}}, ${{b32:st_col}}, {block_width - 1};
                mad.lo.u32 ${{b32:st_row}}, ${{b32:st_block}}, {root_sizes[0]}, ${{b32:st_row}};
            """)
        template += textwrap.dedent(f"""\
            mad.lo.u32 ${{b32:st_off}}, ${{b32:st_row}}, {block_width}, ${{b32:st_col}};
            shl.b32 ${{b32:st_off}}, ${{b32:st_off}}, 1;
        """)
        if src_type.layout == "wgmma_warp":
            pairs = _stmatrix_pairs(op)
            groups = []
            start = 0
            while start < len(pairs):
                count = 4 if len(pairs) - start >= 4 else 2 if len(pairs) - start >= 2 else 1
                groups.append(([p for p, _, _ in pairs[start:start + count]], pairs[start][2] * 2))
                start += count
        else:
            groups = (
                _stmatrix_transposed_groups(
                    register_starts, n_dim, col_start, root_sizes[1]
                )
                if transposed
                else _stmatrix_groups(
                    register_starts,
                    {pair for pair, _, _ in _stmatrix_pairs(op)},
                    n_dim,
                    col_start,
                    root_sizes[0],
                    block_width,
                )
            )
        # Swizzle<log2(w / 8), 4, 3>: the 16-byte chunk within a 2w-byte row
        # is XORed with the row bits above it.
        swizzle_mask = (block_width // 8 - 1) << 7 if dst_type.swizzle else 0
        for registers, physical_offset in groups:
            regs = ", ".join(f"${{b32:operand.src.r{i}}}" for i in registers)
            template += (
                textwrap.dedent(f"""\
                add.u32 ${{b32:st_addr_off}}, ${{b32:st_off}}, {physical_offset};
                and.b32 ${{b32:st_swizzle_mask}}, ${{b32:st_addr_off}}, {swizzle_mask};
                shr.u32 ${{b32:st_swizzle_mask}}, ${{b32:st_swizzle_mask}}, 3;
                xor.b32 ${{b32:st_addr_off}}, ${{b32:st_addr_off}},
                    ${{b32:st_swizzle_mask}};
                add.u32 ${{b32:st_addr}}, ${{b32:operand.dst.ptr}},
                    ${{b32:st_addr_off}};
            """)
                if swizzle_mask
                else textwrap.dedent(f"""\
                add.u32 ${{b32:st_addr}}, ${{b32:operand.dst.ptr}},
                    ${{b32:st_off}};
                add.u32 ${{b32:st_addr}}, ${{b32:st_addr}}, {physical_offset};
            """)
            )
            shape = f"x{len(registers)}" + (".trans" if transposed else "")
            if is_load:
                template += (
                    f"ldmatrix.sync.aligned.m8n8.{shape}.shared::cta.b16 "
                    f"{{{regs}}}, [${{b32:st_addr}}];\n"
                )
            else:
                template += (
                    f"{predicate}stmatrix.sync.aligned.m8n8.{shape}.shared::cta.b16 "
                    f"[${{b32:st_addr}}], {{{regs}}};\n"
                )
        lowered = PTXTemplateOp.build(
            operands=[dst, src],
            result_types=[],
            template=template,
            operand_names=["dst", "src"],
            result_names=[],
            description=str(op),
        )
        return [lowered], []

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
        register = op.dst if is_load else op.src
        rmem_type = register.type
        assert isinstance(smem_ir_type, SmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        smem_label = "src" if is_load else "dst"
        dtype_bytes = cast(
            BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
            smem_ir_type.dtype,
        ).byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        registers = _register_operand_regs(register, "dst" if is_load else "src")[1]
        template = ""
        base_reg = f"${{b32:operand.{smem_label}.ptr}}"

        for flat_idx, idx in enumerate(
            itertools.product(
                *[range(s) for s in cast(tuple[int, ...], rmem_type.sizes)]
            )
        ):
            strides = cast(tuple[int, ...], smem_ir_type.strides)
            offset = sum(c * s for c, s in zip(idx, strides)) * dtype_bytes
            reg = registers[flat_idx]
            if is_load:
                template += f"ld.shared.{suffix} {reg}, [{base_reg}+{offset}];\n"
            else:
                template += f"st.shared.{suffix} [{base_reg}+{offset}], {reg};\n"

        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[],
            template=template,
            operand_names=operand_names,
            result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_async_tma_store(
        self,
        op: TmaMoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower a CTA-wide or warp-private smem → gmem TMA store.

        One elected thread of warp 0 issues cp.async.bulk.tensor.2d and commits
        the group; every thread's bulk await waits on its own groups, so it
        covers whichever thread committed. The barrier inserted before the async
        orders all threads' shared-memory writes before the store starts.
        Warp-private stores publish every lane through the async-proxy fence
        and converge with elect.sync before one lane of each warp issues TMA.
        """
        dst_value = cast(Value, self.value_map[op.dst])
        src_value = cast(Value, self.value_map[op.src])
        idx_values = [cast(Value, self.value_map[v]) for v in op.indices]
        desc_type = dst_value.type
        assert isinstance(desc_type, TmaDescType)

        warp_private = op.attrs.get("tma_store_scope") == "warp"
        leader_operands, leader_names, leader_thread = (
            ([], [], "elect.sync _|${pred:st_leader}, 0xffffffff;\n")
            if warp_private
            else self._leader_thread("st")
        )
        operands = [dst_value, src_value, *idx_values, *leader_operands]
        operand_names = [
            "dst",
            "src",
            *[f"idx{i}" for i in range(len(idx_values))],
            *leader_names,
        ]
        coordinate_lines, coordinates = _tma_coordinates(
            desc_type,
            [f"idx{i}" for i in range(len(idx_values))],
            "store",
        )
        tensor_rank = len(coordinates)
        indices = ", ".join(coordinates)
        tma_instruction = self._tma_instruction(
            op,
            f"cp.async.bulk.tensor.{tensor_rank}d.global.shared::cta.bulk_group",
            f"[${{b64:operand.dst.ptr}}, {{{indices}}}], [${{b32:operand.src.ptr}}]",
            operands,
            operand_names,
            "cache_policy",
        )

        # The RAW (all threads finish writing the smem tile before the leader
        # stores it) is handled by insert_syncs, which emits a barrier before
        # this async.
        # Warp-private groups commit on every lane (nonissuers commit empty
        # groups). Predicating this commit introduces reconvergence that can
        # drain unrelated pending WGMMA; the warp rendezvous after source-read
        # completion still protects scratch reuse.
        lines = [
            "fence.proxy.async.shared::cta;",
            *coordinate_lines,
            f"@${{pred:st_leader}} {tma_instruction}",
            (
                "cp.async.bulk.commit_group;"
                if warp_private
                else "@${pred:st_leader} cp.async.bulk.commit_group;"
            ),
            "${label:st_done}:",
        ]
        if warp_private:
            # All lane-local STMatrix writes cross the async proxy fence
            # before the full-warp election permits the issuer to proceed.
            template = (
                "fence.proxy.async.shared::cta;\n"
                + leader_thread
                + "\n".join(lines[1:])
                + "\n"
            )
        else:
            template = _leader_thread("st") + "\n".join(lines) + "\n"

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

        leader_thread = _leader_thread(
            "ld",
            before_branch=(
                "mov.u32 ${b32:cluster_rank}, %cluster_ctarank;\n"
                "setp.eq.u32 ${pred:cluster_leader}, ${b32:cluster_rank}, 0;\n"
            ),
        )
        operands: list[Value] = [lock_value]
        operand_names: list[str] = ["lock"]
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
                    BF16Type | F16Type | E4M3Type | E5M2Type | F32Type | F64Type,
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
                f"cp.async.bulk.tensor.{tensor_rank}d.shared::cluster.global."
                "mbarrier::complete_tx::bytes"
            )
            arguments = (
                f"[${{b32:operand.{dn}.ptr}}], "
                f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                f"[${{b32:operand.lock.val}}]"
            )
            issue = "ld_leader"
            if tma.multicast is True:
                assert multicast_mask is not None
                opcode = f"{opcode}.multicast::cluster"
                arguments = f"{arguments}, {multicast_mask}"
                issue = "multicast_issue"
                if multicast_count > 1:
                    # Cluster CTAs share a group's multicast slices: rank r
                    # issues every slice j with j % cluster_size == r.
                    issue = f"multicast_issue{i}"
                    load_lines += [
                        f"setp.eq.u32 ${{pred:issuer{i}}}, "
                        f"${{b32:cluster_rank}}, {multicasts % cluster_size};",
                        f"and.pred ${{pred:{issue}}}, ${{pred:ld_leader}}, "
                        f"${{pred:issuer{i}}};",
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
            "@${pred:ld_leader} "
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
            f"[${{b32:operand.lock.val}}], {total_bytes};",
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


def _build_register_layout_dynamic_offset(
    layout: RegisterLayout,
    strides: tuple[int, ...],
) -> tuple[list[Op], Value]:
    """Materialize IR ownership expressions as an affine byte offset."""
    ops: list[Op] = []
    constants: dict[int, Value] = {}
    sources: dict[str, Value] = {}

    def constant(value: int) -> Value:
        if value not in constants:
            op = IndexConstantOp.build(value)
            ops.append(op)
            constants[value] = op.result()
        return constants[value]

    def source(name: str) -> Value:
        if name not in sources:
            if name == 'thread_x':
                op = ThreadIdOp.build(CollectiveAxis.X)
                ops.append(op)
                sources[name] = op.value
            else:
                op = IndexModOp.build(source('thread_x'), constant(32))
                ops.append(op)
                sources[name] = op.result()
        return sources[name]

    offset = None
    bases = layout.thread_bases or ((),) * len(layout.shape)
    for base, stride in zip(bases, strides, strict=True):
        if stride == 0:
            continue
        for term in base:
            value = source(term.source)
            for operation, factor in ((IndexDivOp, term.divisor),
                                      (IndexModOp, term.modulus),
                                      (IndexMulOp, term.scale * stride)):
                if factor is None or (operation != IndexModOp and factor == 1):
                    continue
                op = operation.build(value, constant(factor))
                ops.append(op)
                value = op.result()
            if offset is not None:
                op = IndexAddOp.build(offset, value)
                ops.append(op)
                value = op.result()
            offset = value
    return ops, offset if offset is not None else constant(0)


def _stmatrix_transposed_groups(
    register_starts: list[int],
    n_dim: int,
    col_start: int,
    dst_rows: int,
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
    transposed: bool = False,
) -> int:
    """Byte offset for a fragment pair in the destination's blocked layout."""
    di, dj = _wgmma_pair_static_offset(pair_index, n_dim)
    row, col = di, dj - col_start
    if transposed:
        row, col = col, row
    assert row >= 0 and col >= 0 and col % 8 == 0
    return (
        col // block_width * dst_rows * block_width
        + row * block_width
        + col % block_width
    ) * 2


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
    lines = (
        [
            f"shr.u32 {block}, {col}, {block_width.bit_length() - 1};",
            f"and.b32 {inner}, {col}, {block_width - 1};",
        ]
        if block_width & (block_width - 1) == 0
        else [
            f"div.u32 {block}, {col}, {block_width};",
            f"rem.u32 {inner}, {col}, {block_width};",
        ]
    )
    if order == "row_major":
        coordinates = [inner, block, row, *outer]
    else:
        coordinates = [inner, row, block, *outer]
    return lines, coordinates


def _stmatrix_layout_root(value: Value) -> Value:
    """Keep the typed layout boundary when shared storage is reused."""
    while isinstance(value.definition, (TileOp, ReshapeOp)):
        value = value.definition.arg
    return value


def _stmatrix_register_starts(op: MoveOp) -> list[int] | None:
    is_load = isinstance(op.dst.type, WgrmemTensorType)
    fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
    dst_type = memory.type
    src_type = fragment.type
    dst_root = _stmatrix_layout_root(memory)
    layout = physical_layout(dst_type) if isinstance(dst_type, SmemTensorType) else None
    root_type = dst_root.type
    plain = (
        isinstance(root_type, SmemTensorType)
        and root_type.rank == 2
        and root_type.is_canonical_strides
        and cast(int, root_type.sizes[1]) % 8 == 0
        and root_type.swizzle is None
        and isinstance(dst_type, SmemTensorType)
        and dst_type.swizzle is None
        and physical_layout(root_type) is None
        and layout is None
    )
    swizzled_warp = (
        isinstance(src_type, WgrmemTensorType)
        and src_type.layout == "wgmma_warp"
        and isinstance(root_type, SmemTensorType) and root_type.rank == 2
        and isinstance(dst_type, SmemTensorType)
        and root_type.swizzle == dst_type.swizzle == "128B"
        and list(root_type.strides) == list(dst_type.strides) == [64, 1]
        and 0 < cast(int, dst_type.sizes[1]) <= 64
        and cast(int, dst_type.sizes[1]) % 8 == 0
        and physical_layout(root_type) is None and layout is None
    )
    if not (
        isinstance(dst_type, SmemTensorType)
        and isinstance(src_type, WgrmemTensorType)
        and src_type.layout in ("wgmma_layout_0", "wgmma_warp")
        and isinstance(src_type.dtype, (BF16Type, F16Type))
        and dst_type.dtype == src_type.dtype
        and dst_type.rank == 2
        and (dst_type.strides[-1] == 1 or is_column_major(dst_type))
        and (plain or swizzled_warp or (
            layout is not None
            and layout[1] == "block_major"
            and dst_type.swizzle == _block_swizzle(layout[0])
        ))
    ):
        return None
    if src_type.layout == "wgmma_warp":
        if (not (plain or swizzled_warp) or src_type.sizes[0] != 8 or src_type.warps != 1
                or list(dst_type.sizes) != list(src_type.sizes)):
            return None
        return [pair for pair, _, _ in _stmatrix_pairs(op)]
    block_width = cast(int, root_type.sizes[1]) if plain else layout[0]

    dst_root = _stmatrix_layout_root(memory)
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
        (not plain and (
            root_type.swizzle != _block_swizzle(block_width)
            or physical_layout(root_type) != (block_width, "block_major")
        ))
        or cast(int, root_type.sizes[0]) % 64
        or not dense
    ):
        return None

    m_dim, n_dim = cast(tuple[int, int], src_type.sizes)
    src_root = root_buffer(fragment)
    assert src_root is not None
    src_root_type = src_root.type
    assert isinstance(src_root_type, WgrmemTensorType)
    root_n = cast(int, src_root_type.sizes[1])
    is_view = isinstance(fragment.definition, TileOp)
    row_start, col_start = (
        _static_view_origin(fragment)
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
        or m_dim % 64
        or dst_n % (8 if transposed or plain else block_width)
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


def _stmatrix_pairs(op: MoveOp) -> list[tuple[int, int, int]]:
    """WGMMA pairs transferred in either direction, relative to shared memory."""
    is_load = isinstance(op.dst.type, WgrmemTensorType)
    fragment, memory = (op.dst, op.src) if is_load else (op.src, op.dst)
    if isinstance(fragment.definition, TileOp):
        return _wgmma_view_pairs(fragment)
    col_start = op.attrs.get("wgmma_col_start", 0)
    assert isinstance(col_start, int)
    dst_n = cast(int, cast(MemTensorType, memory.type).sizes[1])
    return [
        (pair, row, col - col_start)
        for pair, row, col in _wgmma_view_pairs(fragment)
        if col_start <= col < col_start + dst_n
    ]


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
