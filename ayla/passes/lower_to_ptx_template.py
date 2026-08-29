"""Lower high-level GPU-dialect ops to ptx_template ops.

Implements `LowerToPtxTemplate`, a `LoweringPass` subclass that
dispatches to per-op rewriter methods. Today only `tile` is lowered;
other ops (alloc, load, store, matmul) will be added incrementally.
"""
from __future__ import annotations

import itertools
import math
import struct
import textwrap

from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    AsyncCommitOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BF16Type,
    ConvertOp,
    DeallocOp,
    F16Type,
    F32Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    PartialSplitBlockIdsOp,
    SplitNBlockIdOp,
    PTXTemplateOp,
    PtrAddOp,
    PtrType,
    RmemTensorType,
    SmemTensorType,
    ThreadIdsOp,
    TmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    ToPtrOp,
    WrmemTensorType,
    IndexType,
    physical_layout,
    root_buffer,
)
from ..base.ir import Op, Value, find_ops
from ..base.lowering_pass import LoweringPass, lowers
from ..targets import PtxTarget


class LowerToPtxTemplate(LoweringPass):
    """Pass that rewrites high-level ops to ptx_template ops."""

    def __init__(self, target: PtxTarget) -> None:
        super().__init__()
        self.target = target
        self.cache_policy_values: dict[str, Value] = {}

    @lowers(PartialSplitBlockIdsOp)
    def lower_partial_split_block_ids(
        self, op: PartialSplitBlockIdsOp,
    ) -> tuple[list[Op], list[object]]:
        split_stride = op.n_tiles + 1
        split_extent = op.split_rows * split_stride
        # Both regions are evaluated and selected uniformly by CTA. The tail
        # subtraction may wrap for prefix CTAs, but those temporary values are
        # discarded by selp; every divisor is a positive compile-time value.
        template = textwrap.dedent(f"""\
            mov.u32 ${{b32:bid}}, %ctaid.x;
            setp.lt.u32 ${{pred:split_prefix}}, ${{b32:bid}}, {split_extent};
            div.u32 ${{b32:prefix_m}}, ${{b32:bid}}, {split_stride};
            rem.u32 ${{b32:prefix_local_n}}, ${{b32:bid}}, {split_stride};
            min.u32 ${{b32:prefix_n}}, ${{b32:prefix_local_n}}, {op.n_tiles - 1};
            setp.ge.u32 ${{pred:local_narrow}}, ${{b32:prefix_local_n}}, {op.n_tiles - 1};
            and.pred ${{pred:narrow}}, ${{pred:split_prefix}}, ${{pred:local_narrow}};
            selp.u32 ${{b32:result.narrow.val}}, 1, 0, ${{pred:narrow}};
            mov.u32 ${{b32:result.half.val}}, 0;
            @${{pred:narrow}} sub.u32 ${{b32:result.half.val}}, ${{b32:prefix_local_n}}, {op.n_tiles - 1};
            sub.u32 ${{b32:tail_bid}}, ${{b32:bid}}, {split_extent};
            div.u32 ${{b32:tail_m}}, ${{b32:tail_bid}}, {op.n_tiles};
            add.u32 ${{b32:tail_m}}, ${{b32:tail_m}}, {op.split_rows};
            rem.u32 ${{b32:tail_n}}, ${{b32:tail_bid}}, {op.n_tiles};
            selp.u32 ${{b32:result.m.val}}, ${{b32:prefix_m}}, ${{b32:tail_m}}, ${{pred:split_prefix}};
            selp.u32 ${{b32:result.n.val}}, ${{b32:prefix_n}}, ${{b32:tail_n}}, ${{pred:split_prefix}};
        """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[
                IndexType(), IndexType(), IndexType(), IndexType(),
            ],
            template=template, operand_names=[],
            result_names=["m", "n", "narrow", "half"],
            description=str(op),
        )
        return [new_op], list(new_op.results)

    @lowers(SplitNBlockIdOp)
    def lower_split_n_block_id(
        self, op: SplitNBlockIdOp,
    ) -> tuple[list[Op], list[object]]:
        template = textwrap.dedent(f"""\
            mov.u32 ${{b32:bid}}, %ctaid.x;
            setp.gt.u32 ${{pred:after_split}}, ${{b32:bid}}, {op.split_tile};
            selp.u32 ${{b32:skip}}, 1, 0, ${{pred:after_split}};
            sub.u32 ${{b32:result.n.val}}, ${{b32:bid}}, ${{b32:skip}};
            setp.eq.u32 ${{pred:narrow}}, ${{b32:result.n.val}}, {op.split_tile};
            selp.u32 ${{b32:result.narrow.val}}, 1, 0, ${{pred:narrow}};
            sub.u32 ${{b32:result.half.val}}, ${{b32:bid}}, {op.split_tile};
            @!${{pred:narrow}} mov.u32 ${{b32:result.half.val}}, 0;
        """)
        index = IndexType()
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[index, index, index], template=template,
            operand_names=[], result_names=["n", "narrow", "half"],
            description=str(op),
        )
        return [new_op], list(new_op.results)

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        # TODO: Materialize cache policies in an earlier pass, potentially
        # alongside buffer promotion, so this pass only renders their PTX.
        cache_hints = sorted({
            hint
            for tma in find_ops(op, lambda candidate: isinstance(
                candidate, TmaMoveOp,
            ))
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
        """Lower a tile op to ToPtrOp + index arithmetic + PtrAddOp."""
        arg = self.value_map[op.arg]
        new_ops: list[Op] = []

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

        root_type = root_buffer(op.arg).type
        assert isinstance(root_type, AddrMemTensorType)
        dtype_bytes = root_type.dtype.byte_size
        result_sizes = list(op.result().type.sizes)
        element_offset_dims = set(op.attrs.get("element_offset_dims", []))
        layout = (
            physical_layout(root_type)
            if isinstance(root_type, SmemTensorType) else None
        )

        # Accumulate byte offset for each dimension.
        zero = IndexConstantOp.build(0)
        new_ops.append(zero)
        byte_offset = zero.result()

        for dim, idx in enumerate(op.indices):
            if isinstance(idx, int):
                const_op = IndexConstantOp.build(idx)
                new_ops.append(const_op)
                idx = const_op.result()
            else:
                idx = self.value_map[idx]

            tile_size_op = IndexConstantOp.build(
                1 if dim in element_offset_dims else result_sizes[dim]
            )
            new_ops.append(tile_size_op)
            elem_off = IndexMulOp.build(idx, tile_size_op.result())
            new_ops.append(elem_off)

            stride = root_type.strides[dim]
            if (
                layout is not None
                and layout[1] == "block_major"
            ):
                block_width, _ = layout
                assert root_type.rank == 2
                if dim == 0:
                    stride = block_width
                elif dim == 1:
                    assert result_sizes[1] % block_width == 0
                    stride = root_type.sizes[0]

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
        if isinstance(result_type, WrmemTensorType):
            return self._lower_alloc_wrmem(op)
        if isinstance(result_type, TmemTensorType):
            return self._lower_alloc_tmem(op)
        if isinstance(result_type, LockType):
            return self._lower_alloc_lock(op)
        raise NotImplementedError(
            f"lower_alloc: unsupported result type {type(result_type).__name__}"
        )

    def _lower_alloc_tmem(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        assert isinstance(result_type, TmemTensorType)
        if self.target.tensor_core_kind != "tcgen05":
            raise ValueError(
                "tensor-memory allocation requires a Blackwell target; "
                f"got {self.target.architecture!r}"
            )
        if op.init not in (None, 0, 0.0):
            raise ValueError("tensor-memory allocation only supports zero init")
        cta_group = int(result_type.attrs.get("cta_group", 1))
        if cta_group not in (1, 2):
            raise ValueError("tensor-memory cta_group must be 1 or 2")
        # tcgen05.alloc is warp-collective. Warp 0 briefly performs the required
        # issue sequence in each participating CTA; it is not assigned a
        # persistent pipeline role.
        template = textwrap.dedent(f"""\
            .shared .align 4 .b32 ${{smem:tmem_addr}}[1];
            mov.u32 ${{b32:tmem_tid}}, %tid.x;
            setp.lt.u32 ${{pred:tmem_warp0}}, ${{b32:tmem_tid}}, 32;
            @${{pred:tmem_warp0}} tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32 [${{smem:tmem_addr}}], {result_type.columns};
            bar.sync 0;
            ld.shared.b32 ${{b32:result.dst.ptr}}, [${{smem:tmem_addr}}];
            bar.sync 0;
            @${{pred:tmem_warp0}} tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned;
        """)
        if op.init is not None:
            if not (
                result_type.sizes[0] in (64, 128)
                and result_type.sizes[1] % 8 == 0
                and 8 <= result_type.sizes[1] <= result_type.columns
                and result_type.dtype.name == "f32"
            ):
                raise NotImplementedError(
                    "tensor-memory zero init currently supports "
                    "tmem<64|128x8..512xf32> with 8-column granularity"
                )
            template += textwrap.dedent(f"""\
                mov.u32 ${{b32:tmem_warp}}, ${{b32:tmem_tid}};
                shr.u32 ${{b32:tmem_warp}}, ${{b32:tmem_warp}}, 5;
                shl.b32 ${{b32:tmem_lane_base}}, ${{b32:tmem_warp}}, 21;
                add.u32 ${{b32:tmem_warp_addr}}, ${{b32:result.dst.ptr}}, ${{b32:tmem_lane_base}};
                mov.b32 ${{b32:tmem_zero}}, 0;
            """)
            for col in range(0, result_type.columns, 128):
                width = min(128, result_type.columns - col)
                zeros = ", ".join(
                    "${b32:tmem_zero}" for _ in range(width)
                )
                template += textwrap.dedent(f"""\
                    add.u32 ${{b32:tmem_chunk_addr}}, ${{b32:tmem_warp_addr}}, {col};
                    tcgen05.st.sync.aligned.32x32b.x{width}.b32 [${{b32:tmem_chunk_addr}}], {{{zeros}}};
                    tcgen05.wait::st.sync.aligned;
                """)
            template += textwrap.dedent("""\
                bar.sync 0;
                fence.proxy.async.shared::cta;
            """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[result_type], template=template,
            operand_names=[], result_names=["dst"], description=str(op),
        )
        return [new_op], [new_op.result()]

    @lowers(DeallocOp)
    def lower_dealloc(
        self, op: DeallocOp,
    ) -> tuple[list[Op], list[object]]:
        value = self.value_map[op.value]
        value_type = op.value.type
        assert isinstance(value_type, TmemTensorType)
        if self.target.tensor_core_kind != "tcgen05":
            raise ValueError(
                "tensor-memory deallocation requires a Blackwell target; "
                f"got {self.target.architecture!r}"
            )
        cta_group = int(value_type.attrs.get("cta_group", 1))
        cluster_sync = (
            "barrier.cluster.arrive.aligned;\n"
            "barrier.cluster.wait.aligned;\n"
            if cta_group == 2 else ""
        )
        template = textwrap.dedent(f"""\
            bar.sync 0;
            {cluster_sync}
            mov.u32 ${{b32:tmem_tid}}, %tid.x;
            setp.lt.u32 ${{pred:tmem_warp0}}, ${{b32:tmem_tid}}, 32;
            @${{pred:tmem_warp0}} tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32 ${{b32:operand.src.ptr}}, {value_type.columns};
        """)
        new_op = PTXTemplateOp.build(
            operands=[value], result_types=[], template=template,
            operand_names=["src"], result_names=[], description=str(op),
        )
        return [new_op], []

    def _lower_alloc_lock(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        assert isinstance(lock_type, LockType)
        if lock_type.kind not in ("mbar", "tcgen05"):
            return [], [None]
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
            bar.sync 0;
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
        n = result_type.num_elements
        suffix = _ptx_suffix(result_type.dtype)

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

    def _lower_alloc_wrmem(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        result_type = op.result().type
        assert isinstance(result_type, WrmemTensorType)
        n = result_type.num_elements_per_thread
        suffix = result_type.dtype.name
        if result_type.dtype.byte_size == 2:  # bf16/f16 packed in pairs into b32
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
        n_bytes = smem_type.num_elements * smem_type.dtype.byte_size
        # TMA swizzling and the corresponding software address swizzle operate
        # within 1 KiB regions and require the shared-memory base to match.
        alignment = (
            1024
            if smem_type.attrs.get("tcgen05_layout") is not None
            or smem_type.attrs.get("tcgen05_matrix")
            or (
                smem_type.swizzle is not None
                and physical_layout(smem_type) is not None
            )
            else 128
        )
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

    @lowers(ConvertOp)
    def lower_convert(
        self, op: ConvertOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower element-wise dtype conversion on wrmem tensors.

        For f32 → bf16/f16, converts consecutive pairs directly into packed
        b32 registers with the PTX ``*x2`` conversion form.
        """
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]
        dst_type = dst_value.type
        src_type = src_value.type

        if not (isinstance(src_type, WrmemTensorType)
                and isinstance(dst_type, WrmemTensorType)):
            raise NotImplementedError(
                f"lower_convert: only wrmem supported; got {src_type} -> {dst_type}"
            )
        assert tuple(src_type.sizes) == tuple(dst_type.sizes)
        assert src_type.warps == dst_type.warps

        src_n = src_type.num_elements_per_thread
        if not (isinstance(src_type.dtype, F32Type)
                and isinstance(dst_type.dtype, (BF16Type, F16Type))):
            raise NotImplementedError(
                f"lower_convert: unsupported {src_type.dtype} -> {dst_type.dtype}"
            )

        template = ""
        for i in range(0, src_n, 2):
            dst_reg = i // 2
            template += (
                f"cvt.rn.{dst_type.dtype.name}x2.f32 "
                f"${{b32:operand.dst.r{dst_reg}}}, "
                f"${{f32:operand.src.r{i + 1}}}, "
                f"${{f32:operand.src.r{i}}};\n"
            )

        new_op = PTXTemplateOp.build(
            operands=[dst_value, src_value],
            result_types=[],
            template=template,
            operand_names=["dst", "src"],
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
        if (isinstance(lhs_type, WrmemTensorType)
                and isinstance(rhs_type, WrmemTensorType)
                and isinstance(dst_type, WrmemTensorType)):
            return self._lower_matmul_mma_sync(op)
        raise NotImplementedError(
            f"lower_matmul: unsupported operand types; got "
            f"{lhs_type} @ {rhs_type} -> {dst_type}"
        )

    def _lower_matmul_fma(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]]:
        lhs = self.value_map[op.lhs]
        rhs = self.value_map[op.rhs]
        dst = self.value_map[op.dst]
        if not isinstance(lhs.type.dtype, F32Type):
            raise NotImplementedError(
                f"lower_async_matmul fma: only f32 supported; got {lhs.type.dtype}"
            )
        if not (lhs.type.dtype == rhs.type.dtype == dst.type.dtype):
            raise NotImplementedError(
                f"lower_async_matmul fma: dtype mismatch; got "
                f"{lhs.type.dtype}, {rhs.type.dtype}, {dst.type.dtype}"
            )
        m_dim, k_dim = lhs.type.sizes
        n_dim, k_dim2 = rhs.type.sizes
        if k_dim != k_dim2:
            raise NotImplementedError(
                f"lower_async_matmul fma: k-dim mismatch; got "
                f"{lhs.type} @ {rhs.type}"
            )
        if tuple(dst.type.sizes) != (m_dim, n_dim):
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
        lhs = self.value_map[op.lhs]
        rhs = self.value_map[op.rhs]
        dst = self.value_map[op.dst]
        assert dst.type.layout == "mma", (
            f"_lower_async_matmul_mma_sync: dst layout must be 'mma'; got {dst.type.layout!r}"
        )
        if not (isinstance(lhs.type.dtype, (BF16Type, F16Type))
                and isinstance(rhs.type.dtype, (BF16Type, F16Type))
                and isinstance(dst.type.dtype, F32Type)):
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: expected bf16/f16 inputs, f32 output; "
                f"got {lhs.type.dtype}, {rhs.type.dtype} -> {dst.type.dtype}"
            )
        m_dim, k_dim = lhs.type.sizes
        n_dim, k_dim2 = rhs.type.sizes
        if k_dim != k_dim2:
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: k-dim mismatch; got "
                f"{lhs.type} @ {rhs.type}"
            )
        if tuple(dst.type.sizes) != (m_dim, n_dim):
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: dst shape must be {m_dim}x{n_dim}; "
                f"got {dst.type}"
            )
        if m_dim != 16 or k_dim % 16 != 0 or n_dim % 8 != 0:
            raise NotImplementedError(
                f"lower_async_matmul mma.sync: need m=16, k%%16=0, n%%8=0; "
                f"got {m_dim}x{k_dim} @ {k_dim}x{n_dim}"
            )
        in_dtype = lhs.type.dtype.name
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
        if body_ops and all(isinstance(o, TmaMoveOp) for o in body_ops):
            kind = op.lock.type.kind
            if kind == "mbar":  # one or more gmem→smem loads sharing the mbar
                lock_value = self.value_map[op.lock]
                return self._lower_async_tma_loads(op, body_ops, lock_value)
            if kind == "bulk":  # smem→gmem store (single)
                assert len(body_ops) == 1, (
                    f"lower_async: bulk TMA store group must be a single move; "
                    f"got {len(body_ops)}"
                )
                return self._lower_async_tma_store(body_ops[0])
            raise NotImplementedError(
                f"lower_async: tma_move with lock kind {kind!r}"
            )
        if body_ops and all(
            isinstance(o, (MatmulOp, AwaitOp, AsyncCommitOp)) for o in body_ops
        ):
            if op.lock.type.kind == "tcgen05":
                return self._lower_async_matmul_tcgen05(op, body_ops)
            if any(isinstance(o, AsyncCommitOp) for o in body_ops):
                raise NotImplementedError(
                    "in-group partial commits are currently supported only "
                    "for tcgen05"
                )
            if any(isinstance(o, AwaitOp) for o in body_ops):
                raise NotImplementedError(
                    "in-group input waits are currently supported only for tcgen05"
                )
            return self._lower_async_matmul_wgmma(op, body_ops)
        raise NotImplementedError(
            f"lower_async: unsupported body ops "
            f"{[type(o).__name__ for o in body_ops]}"
        )

    def _lower_async_matmul_wgmma(
        self, op: AsyncOp, mms: list[MatmulOp],
    ) -> tuple[list[Op], list[object]]:
        """Lower a group of smem × smem → wrmem matmuls to wgmma.mma_async.

        The whole group is one wgmma commit group: a single ``wgmma.fence``, all
        the ``mma_async`` (accumulating into the shared dst), and a single
        ``wgmma.commit_group``. The hardware serializes same-accumulator mmas, so
        the await's lone ``wgmma.wait_group`` covers them all (this is the batched
        form nvcc emits, cf. ``fastcu_5_1``). The group's await op (lock kind
        ``wgmma``) emits the wait plus a ``bar.sync`` that closes the TMA-reload
        vs warpgroup-read WAR (see ``_lower_await_wgmma``).
        """
        if self.target.tensor_core_kind != "wgmma":
            raise ValueError(
                "wgmma lowering requires a Hopper target; "
                f"module target {self.target.architecture!r} uses "
                f"{self.target.tensor_core_kind!r} tensor cores"
            )
        assert op.lock.type.kind == "wgmma", (
            f"_lower_async_matmul_wgmma: expected lock kind 'wgmma'; got {op.lock.type.kind!r}"
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
            # TODO: Support arbitrary-length TileOp chains. Currently descriptor
            # reuse is limited to siblings in one WGMMA commit group that share
            # the same direct parent/base.
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
            # Non-affine layouts need a root-aware address mapping. Rebuild
            # their descriptors until that support is introduced with K=128.
            if typ.attrs.get("physical_layout") is not None:
                return None
            delta = sum(
                (int(index) - int(base_index)) * size * stride
                for index, base_index, size, stride in zip(
                    indices, base_indices, typ.sizes, typ.strides,
                )
            )
            byte_delta = delta * typ.dtype.byte_size
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
            operand = _operand(self.value_map[value], hint)
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
                    and isinstance(dst_type, WrmemTensorType)
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
            m_dim, k_dim = lhs_type.sizes
            n_dim, k_dim2 = rhs_type.sizes
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
            lhs_row_stride_bytes = lhs_type.strides[0] * lhs_type.dtype.byte_size
            rhs_row_stride_bytes = rhs_type.strides[0] * rhs_type.dtype.byte_size

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
            dst_name = _operand(self.value_map[mm.dst], "dst")

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

            reg_offset = 0
            for m_it in range(m_iters):
                regs = ", ".join(
                    f"${{f32:operand.{dst_name}.r{reg_offset + i}}}"
                    for i in range(num_regs_per_wgmma)
                )
                mma_template += textwrap.dedent(f"""\
                    wgmma.mma_async.sync.aligned.m64n{n_dim}k16.f32.bf16.bf16
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

    def _lower_async_matmul_tcgen05(
        self, op: AsyncOp,
        body_ops: list[MatmulOp | AwaitOp | AsyncCommitOp],
    ) -> tuple[list[Op], list[object]]:
        """Lower the first supported SM100 BF16 tile to ``tcgen05.mma``.

        ``tcgen05.mma`` has single-thread issue semantics. One elected lane of
        warp 0 issues the MMA and its completion commit, while all four warps
        execute the same surrounding load/compute/store pipeline. This is the
        instruction's required issue sequence, not producer/consumer warp
        specialization.
        """
        if self.target.tensor_core_kind != "tcgen05":
            raise ValueError(
                "tcgen05 lowering requires a Blackwell target; "
                f"got {self.target.architecture!r}"
            )
        mms = [body_op for body_op in body_ops if isinstance(body_op, MatmulOp)]
        input_waits = [
            (index, body_op)
            for index, body_op in enumerate(body_ops)
            if isinstance(body_op, AwaitOp)
        ]
        partial_commits = [
            (index, body_op)
            for index, body_op in enumerate(body_ops)
            if isinstance(body_op, AsyncCommitOp)
        ]
        if not mms:
            raise ValueError("tcgen05 async group requires at least one matmul")
        if input_waits:
            wait_index, input_wait = input_waits[0]
            matmuls_before_wait = sum(
                isinstance(body_op, MatmulOp)
                for body_op in body_ops[:wait_index]
            )
            if (
                len(input_waits) != 1
                or input_wait.lock.type.kind != "mbar"
                or matmuls_before_wait == 0
                or matmuls_before_wait == len(mms)
            ):
                raise ValueError(
                    "tcgen05 async input wait requires one mbar await between matmuls"
                )
        else:
            input_wait = None
            matmuls_before_wait = None
        previous_commit_position = 0
        commit_locks = {op.lock}
        for commit_index, commit in partial_commits:
            commit_position = sum(
                isinstance(body_op, MatmulOp)
                for body_op in body_ops[:commit_index]
            )
            if (
                commit.lock.type.kind != "tcgen05"
                or commit.lock in commit_locks
                or commit_position <= previous_commit_position
                or commit_position == len(mms)
            ):
                raise ValueError(
                    "tcgen05 partial commits require distinct tcgen05 locks "
                    "at strictly increasing positions between matmuls"
                )
            previous_commit_position = commit_position
            commit_locks.add(commit.lock)
        for mm in mms:
            lhs_type, rhs_type, dst_type = mm.lhs.type, mm.rhs.type, mm.dst.type
            mm_cta_group = int(dst_type.attrs.get("cta_group", 1))
            logical_n = rhs_type.sizes[0]
            if mm_cta_group == 2 and (
                rhs_type.attrs.get("tcgen05_layout") != "rhs_32x16_cta2"
            ):
                logical_n *= 2
            supported = (
                isinstance(lhs_type, SmemTensorType)
                and isinstance(rhs_type, SmemTensorType)
                and isinstance(dst_type, TmemTensorType)
                and lhs_type.sizes[0] in (64, 128)
                and lhs_type.sizes[1] == 16
                and rhs_type.sizes[0] % 8 == 0
                and 8 <= rhs_type.sizes[0] <= 256
                and rhs_type.sizes[1] == 16
                and tuple(dst_type.sizes) == (lhs_type.sizes[0], logical_n)
                and lhs_type.dtype.name == rhs_type.dtype.name == "bf16"
                and dst_type.dtype.name == "f32"
                and lhs_type.swizzle in ("32B", "128B")
                and rhs_type.swizzle == "128B"
            )
            if not supported:
                raise NotImplementedError(
                    "tcgen05 lowering currently supports "
                    "smem<64|128x16xbf16 swizzle=32B|128B> @ "
                    "smem<8..256x16xbf16 swizzle=128B> -> "
                    "tmem<64|128x8..256xf32>"
                )

        dst = self.value_map[mms[0].dst]
        if any(self.value_map[mm.dst] is not dst for mm in mms[1:]):
            raise NotImplementedError(
                "tcgen05 async groups must share one accumulator"
            )
        lock = self.value_map[op.lock]
        cta_group = int(mms[0].dst.type.attrs.get("cta_group", 1))
        if cta_group not in (1, 2) or any(
            int(mm.dst.type.attrs.get("cta_group", 1)) != cta_group
            for mm in mms
        ):
            raise ValueError("tcgen05 async group requires one cta_group of 1 or 2")
        if partial_commits and cta_group != 1:
            raise NotImplementedError(
                "tcgen05 partial commits currently support only cta_group 1"
            )
        direct_tma_layout = bool(
            mms[0].rhs.type.attrs.get("tcgen05_tma_layout")
        )
        if any(
            bool(mm.rhs.type.attrs.get("tcgen05_tma_layout"))
            != direct_tma_layout
            for mm in mms
        ):
            raise NotImplementedError(
                "tcgen05 async groups cannot mix packed and direct-TMA RHS layouts"
            )
        hybrid_n_tiles = mms[0].attrs.get("hybrid_n_tiles")
        hybrid_n_width = mms[0].attrs.get("hybrid_n_width")
        hybrid_split_rows = mms[0].attrs.get("hybrid_split_rows")
        hybrid_split_tile = mms[0].attrs.get("hybrid_split_tile")
        hybrid_element_offsets = bool(
            mms[0].attrs.get("hybrid_element_offsets")
        )
        if any(
            (
                mm.attrs.get("hybrid_n_tiles") != hybrid_n_tiles
                or mm.attrs.get("hybrid_n_width") != hybrid_n_width
                or mm.attrs.get("hybrid_split_rows") != hybrid_split_rows
                or mm.attrs.get("hybrid_split_tile") != hybrid_split_tile
                or bool(mm.attrs.get("hybrid_element_offsets"))
                != hybrid_element_offsets
                or mm.controls != mms[0].controls
            )
            for mm in mms[1:]
        ):
            raise ValueError("tcgen05 async groups must share hybrid-N controls")
        n = mms[0].rhs.type.sizes[0]
        if cta_group == 2 and (
            mms[0].rhs.type.attrs.get("tcgen05_layout")
            != "rhs_32x16_cta2"
        ):
            n *= 2
        m = 256 if cta_group == 2 else mms[0].lhs.type.sizes[0]
        idesc_value = ((m // 16) << 24) | ((n // 8) << 17) | 0x490
        if not direct_tma_layout:
            idesc_value |= 1 << 16
        idesc = f"0x{idesc_value:08x}"
        # Descriptor constants encode the compact SM100 layouts used by the
        # cooperative staging path. Direct TMA staging retains the source's
        # [N, K] physical orientation, so its B-major bit differs from the
        # packed [K, N] cooperative layout.
        operands = [dst, lock]
        operand_names = ["dst", "lock"]
        if input_wait is not None:
            operands.append(self.value_map[input_wait.lock])
            operand_names.append("input_lock")
        commit_names: dict[AsyncCommitOp, str] = {}
        for index, (_, commit) in enumerate(partial_commits):
            name = f"partial_lock{index}"
            operands.append(self.value_map[commit.lock])
            operand_names.append(name)
            commit_names[commit] = name
        if hybrid_n_tiles is not None:
            if (
                not isinstance(hybrid_n_tiles, int)
                or hybrid_n_tiles < 2
                or cta_group != 1
                or n not in (232, 240, 248, 256)
                or not isinstance(hybrid_n_width, int)
                or hybrid_n_width % 8
                or not 8 <= hybrid_n_width < n
                or not direct_tma_layout
            ):
                raise ValueError(
                    "hybrid-N tcgen05 requires a direct-TMA group-1 "
                    "N232/N240/N248/N256 tile and a valid narrow width"
                )
            if (
                hybrid_split_rows is not None
                or hybrid_split_tile is not None
                or hybrid_element_offsets
            ):
                if (
                    len(mms[0].controls) != 2
                    or (
                        hybrid_split_rows is not None
                        and (
                            type(hybrid_split_rows) is not int
                            or hybrid_split_rows < 1
                        )
                    )
                    or (
                        hybrid_split_tile is not None
                        and (
                            type(hybrid_split_tile) is not int
                            or not 0 <= hybrid_split_tile < hybrid_n_tiles
                        )
                    )
                ):
                    raise ValueError(
                        "controlled hybrid-N tcgen05 requires valid split "
                        "attributes and narrow/half controls"
                    )
                narrow, half = (
                    self.value_map[value] for value in mms[0].controls
                )
                operands.extend([narrow, half])
                operand_names.extend(["hybrid_narrow", "hybrid_half"])
            elif mms[0].controls:
                raise ValueError(
                    "full-row hybrid-N tcgen05 does not accept controls"
                )
            # The final physical N tile is represented by two CTAs. Both load
            # the same N256 B tile, then a uniform CTA predicate selects N128
            # MMA width and the corresponding 16 KiB shared-memory half. This
            # is launch-wave balancing, not persistent warp specialization.
        cluster_issue = ""
        if cta_group == 2:
            cluster_issue = textwrap.dedent("""\
                barrier.cluster.arrive.aligned;
                barrier.cluster.wait.aligned;
                mov.u32 ${b32:cluster_rank}, %cluster_ctarank;
                and.b32 ${b32:cluster_parity}, ${b32:cluster_rank}, 1;
                setp.eq.u32 ${pred:cluster_even}, ${b32:cluster_parity}, 0;
            """)
        template = textwrap.dedent("""\
            fence.proxy.async.shared::cta;
        """) + cluster_issue + textwrap.dedent("""\
            mov.u32 ${b32:tid}, %tid.x;
            shr.u32 ${b32:warp}, ${b32:tid}, 5;
            setp.eq.u32 ${pred:warp0}, ${b32:warp}, 0;
            elect.sync ${b32:elected_lane}|${pred:elected}, -1;
            and.pred ${pred:issue}, ${pred:warp0}, ${pred:elected};
        """)
        if cta_group == 2:
            template += "and.pred ${pred:issue}, ${pred:issue}, ${pred:cluster_even};\n"
        if op.attrs.get("branch_elected_issue"):
            # Descriptor construction belongs to the instruction-mandated
            # elected issue sequence. Skipping it in non-issuing lanes is not
            # a persistent producer/consumer role: every warp still follows
            # the same surrounding load, compute-wait, and store pipeline.
            branch_target = (
                "before_tcgen05_input_wait"
                if input_wait is not None else "after_tcgen05_issue"
            )
            template += f"@!${{pred:issue}} bra ${{label:{branch_target}}};\n"
        if hybrid_n_tiles is not None:
            assert isinstance(hybrid_n_width, int)
            narrow_idesc_value = (
                ((m // 16) << 24)
                | ((hybrid_n_width // 8) << 17)
                | 0x490
            )
            if (
                hybrid_split_rows is not None
                or hybrid_split_tile is not None
                or hybrid_element_offsets
            ):
                template += textwrap.dedent(f"""\
                    setp.ne.u32 ${{pred:narrow_n}}, ${{b32:operand.hybrid_narrow.val}}, 0;
                    mov.u32 ${{b32:rhs_half}}, ${{b32:operand.hybrid_half.val}};
                    @!${{pred:narrow_n}} mov.b32 ${{b32:idesc}}, {idesc};
                    @${{pred:narrow_n}} mov.b32 ${{b32:idesc}}, 0x{narrow_idesc_value:08x};
                    shl.b32 ${{b32:rhs_half_offset}}, ${{b32:rhs_half}}, 14;
                """)
            else:
                half_instruction = (
                    f"shr.b32 ${{b32:rhs_half}}, ${{b32:hybrid_bid}}, "
                    f"{hybrid_n_tiles.bit_length() - 1};"
                    if hybrid_n_tiles & (hybrid_n_tiles - 1) == 0
                    else (
                        f"div.u32 ${{b32:rhs_half}}, ${{b32:hybrid_bid}}, "
                        f"{hybrid_n_tiles};"
                    )
                )
                template += textwrap.dedent(f"""\
                    mov.u32 ${{b32:hybrid_bid}}, %ctaid.x;
                    setp.ge.u32 ${{pred:narrow_n}}, ${{b32:hybrid_bid}}, {hybrid_n_tiles - 1};
                    @!${{pred:narrow_n}} mov.b32 ${{b32:idesc}}, {idesc};
                    @${{pred:narrow_n}} mov.b32 ${{b32:idesc}}, 0x{narrow_idesc_value:08x};
                    __HALF_INSTRUCTION__
                    shl.b32 ${{b32:rhs_half_offset}}, ${{b32:rhs_half}}, 14;
                """).replace("__HALF_INSTRUCTION__", half_instruction)
        else:
            template += f"mov.b32 ${{b32:idesc}}, {idesc};\n"
        template += textwrap.dedent("""\
            mov.b32 ${b32:disable}, 0;
            mov.pred ${pred:use_d}, -1;
        """)
        disable_vector = ", ".join(
            "${b32:disable}" for _ in range(4 * cta_group)
        )
        mma_index = 0
        for body_op in body_ops:
            if isinstance(body_op, AsyncCommitOp):
                template += textwrap.dedent(f"""\
                    @${{pred:issue}} tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
                        [${{b32:operand.{commit_names[body_op]}.val}}];
                """)
                continue
            if isinstance(body_op, AwaitOp):
                template += textwrap.dedent(r"""
                    ${label:before_tcgen05_input_wait}:
                    mov.u32 ${b32:input_wait_ticks}, 0x989680;
                    ${label:tcgen05_input_wait}:
                    mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ${pred:input_ready}, [${b32:operand.input_lock.val}], ${b32:operand.input_lock.parity}, ${b32:input_wait_ticks};
                    @!${pred:input_ready} bra ${label:tcgen05_input_wait};
                    xor.b32 ${b32:operand.input_lock.parity}, ${b32:operand.input_lock.parity}, 1;
                    fence.proxy.async.shared::cta;
                    @!${pred:issue} bra ${label:after_tcgen05_issue};
                """)
                continue
            mm = body_op
            i = mma_index
            mma_index += 1
            lhs = self.value_map[mm.lhs]
            rhs = self.value_map[mm.rhs]
            lhs_desc_const = (
                "0xc000401000000000"
                if mm.lhs.type.swizzle == "32B"
                else "0x4000404000000000"
            )
            rhs_desc_const = "0x4000404000000000"
            operands.extend([lhs, rhs])
            operand_names.extend([f"lhs{i}", f"rhs{i}"])
            rhs_ptr = f"${{b32:operand.rhs{i}.ptr}}"
            if hybrid_n_tiles is not None:
                template += (
                    f"add.u32 ${{b32:rhs_ptr{i}}}, {rhs_ptr}, "
                    "${b32:rhs_half_offset};\n"
                )
                rhs_ptr = f"${{b32:rhs_ptr{i}}}"
            template += textwrap.dedent(f"""\
                bfe.u32 ${{b32:lhs_addr{i}}}, ${{b32:operand.lhs{i}.ptr}}, 4, 14;
                cvt.u64.u32 ${{b64:lhs_desc{i}}}, ${{b32:lhs_addr{i}}};
                bfe.u32 ${{b32:rhs_addr{i}}}, {rhs_ptr}, 4, 14;
                cvt.u64.u32 ${{b64:rhs_desc{i}}}, ${{b32:rhs_addr{i}}};
                or.b64 ${{b64:lhs_desc{i}}}, ${{b64:lhs_desc{i}}}, {lhs_desc_const};
                or.b64 ${{b64:rhs_desc{i}}}, ${{b64:rhs_desc{i}}}, {rhs_desc_const};
                @${{pred:issue}} tcgen05.mma.cta_group::{cta_group}.kind::f16
                    [${{b32:operand.dst.ptr}}], ${{b64:lhs_desc{i}}}, ${{b64:rhs_desc{i}}},
                    ${{b32:idesc}},
                    {{{disable_vector}}},
                    ${{pred:use_d}};
            """)
        if cta_group == 1:
            template += textwrap.dedent("""\
                @${pred:issue} tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
                    [${b32:operand.lock.val}];
            """)
        else:
            template += textwrap.dedent("""\
                mov.b16 ${b16:cta_mask}, 0x3;
                @${pred:issue} tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
                    [${b32:operand.lock.val}], ${b16:cta_mask};
            """)
        if op.attrs.get("branch_elected_issue") or input_wait is not None:
            template += "${label:after_tcgen05_issue}:\n"
        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
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
        if (
            isinstance(dst_type, SmemTensorType)
            and isinstance(src_type, SmemTensorType)
            and dst_type.attrs.get("tcgen05_layout") is not None
        ):
            return self._lower_move_to_smem_tcgen05(
                op,
                self.value_map[op.dst],
                self.value_map[op.src],
                dst_type.attrs["tcgen05_layout"],
            )
        if isinstance(dst_type, WrmemTensorType) and isinstance(src_type, SmemTensorType) and dst_type.layout == "mma":
            return self._lower_move_smem_to_wrmem_mma(op)
        if isinstance(dst_type, SmemTensorType) and isinstance(src_type, WrmemTensorType) and src_type.layout == "mma":
            return self._lower_move_wrmem_mma_to_smem(op)
        if (
            isinstance(dst_type, (SmemTensorType, GmemTensorType))
            and isinstance(src_type, WrmemTensorType)
            and src_type.layout == "wgmma_layout_0"
        ):
            return self._lower_move_wrmem_wgmma(op)
        if isinstance(dst_type, GmemTensorType) and isinstance(
            src_type, TmemTensorType,
        ):
            return self._lower_move_tmem_to_gmem(op)
        raise NotImplementedError(
            f"lower_move: unsupported; got "
            f"{type(dst_type).__name__} <- {type(src_type).__name__}"
        )

    def _lower_move_tmem_to_gmem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        dst_type, src_type = op.dst.type, op.src.type
        assert isinstance(dst_type, GmemTensorType)
        assert isinstance(src_type, TmemTensorType)
        if self.target.tensor_core_kind != "tcgen05":
            raise ValueError(
                "tensor-memory load requires a Blackwell target; "
                f"got {self.target.architecture!r}"
            )
        if not (
            src_type.sizes[0] == dst_type.sizes[0]
            and src_type.sizes[0] in (64, 128)
            and src_type.sizes[1] == dst_type.sizes[1]
            and src_type.sizes[1] % 8 == 0
            and 8 <= src_type.sizes[1] <= 256
            and src_type.dtype.name == "f32"
            and dst_type.dtype.name in ("f32", "bf16")
            and dst_type.strides[1] == 1
        ):
            raise NotImplementedError(
                "tcgen05 tmem-to-gmem move currently supports "
                "64|128x8..256 f32 with 8-column granularity "
                "tensor memory to row-strided f32/bf16 global memory"
            )
        hybrid_n_tiles = op.attrs.get("hybrid_n_tiles")
        hybrid_n_width = op.attrs.get("hybrid_n_width")
        predicated_tail_stores = bool(op.attrs.get("predicated_tail_stores"))
        l1_cache_policy = op.attrs.get("l1_cache_policy")
        l2_eviction_priority = op.attrs.get("l2_eviction_priority")
        hybrid_split_rows = op.attrs.get("hybrid_split_rows")
        hybrid_split_tile = op.attrs.get("hybrid_split_tile")
        hybrid_element_offsets = bool(
            op.attrs.get("hybrid_element_offsets")
        )
        if hybrid_n_tiles is not None and (
            not isinstance(hybrid_n_tiles, int)
            or hybrid_n_tiles < 2
            or src_type.sizes[1] not in (232, 240, 248, 256)
            or not isinstance(hybrid_n_width, int)
            or hybrid_n_width % 8
            or not 8 <= hybrid_n_width < src_type.sizes[1]
            or dst_type.dtype.name != "bf16"
        ):
            raise ValueError(
                "hybrid-N output requires at least two N tiles and a "
                "232-, 240-, 248-, or 256-column BF16 output tile with a "
                "valid narrow width"
            )
        if predicated_tail_stores and (
            hybrid_n_tiles is None or src_type.sizes[0] != 128
        ):
            raise ValueError(
                "predicated tail stores require a hybrid-N M128 output"
            )
        if l1_cache_policy not in (
            None, "no_allocate", "evict_first", "evict_last",
        ):
            raise ValueError(
                f"unsupported output L1 cache policy {l1_cache_policy!r}"
            )
        if l2_eviction_priority not in (None, "evict_first", "evict_last"):
            raise ValueError(
                f"unsupported output L2 eviction priority "
                f"{l2_eviction_priority!r}"
            )
        if (
            l1_cache_policy is not None or l2_eviction_priority is not None
        ) and dst_type.dtype.name != "bf16":
            raise ValueError(
                "output cache policy currently requires BF16 stores"
            )
        if (
            hybrid_split_rows is not None
            or hybrid_split_tile is not None
            or hybrid_element_offsets
        ) and (hybrid_n_tiles is None or len(op.indices) != 2):
            raise ValueError(
                "controlled hybrid-N output requires narrow/half controls"
            )
        if (
            hybrid_n_tiles is not None
            and hybrid_split_rows is None
            and hybrid_split_tile is None
            and not hybrid_element_offsets
            and op.indices
        ):
            raise ValueError("full-row hybrid-N output does not accept controls")
        # M=128 uses all 32 lanes in each of the four tensor-memory datapaths.
        # The non-warp-specialized M=64 Layout F instead uses lanes 0..15 in
        # every datapath, so logical row = 16 * warp + lane. All four warps
        # must still issue each collective tcgen05.ld; invalid half-warps only
        # predicate off their global stores after the tensor-memory load.
        # Process 64 columns at a time so the same register bank handles both
        # 64- and 128-column outputs.
        row_stride_bytes = dst_type.strides[0] * dst_type.dtype.byte_size
        value_type = "f32" if dst_type.dtype.name == "bf16" else "b32"
        values = [f"${{{value_type}:value{i}}}" for i in range(64)]
        load_values = ", ".join(values)
        row_mapping = ""
        store_predicate = ""
        if src_type.sizes[0] == 64:
            row_mapping = textwrap.dedent("""\
                and.b32 ${b32:lane}, ${b32:tid}, 31;
                shl.b32 ${b32:row_index}, ${b32:warp}, 4;
                add.u32 ${b32:row_index}, ${b32:row_index}, ${b32:lane};
                setp.lt.u32 ${pred:valid_row}, ${b32:lane}, 16;
            """)
            store_predicate = "@${pred:valid_row} "
        else:
            row_mapping = "mov.u32 ${b32:row_index}, ${b32:tid};\n"
        template = textwrap.dedent("""\
            mov.u32 ${b32:tid}, %tid.x;
            shr.u32 ${b32:warp}, ${b32:tid}, 5;
        """) + row_mapping + textwrap.dedent(f"""\
            shl.b32 ${{b32:lane_base}}, ${{b32:warp}}, 21;
            add.u32 ${{b32:taddr_row}}, ${{b32:operand.src.ptr}}, ${{b32:lane_base}};
            mul.wide.u32 ${{b64:row_offset}}, ${{b32:row_index}}, {row_stride_bytes};
            add.u64 ${{b64:row}}, ${{b64:operand.dst.ptr}}, ${{b64:row_offset}};
        """)
        alternating_store_alignment = src_type.sizes[1] in (232, 248)
        if alternating_store_alignment:
            # N232/N248 tiles have 16-byte-odd strides. Use a CTA-uniform
            # parity branch so each tile gets maximally wide aligned stores:
            # even tiles use a trailing 16B edge, odd tiles use 16B edges
            # around aligned 32B interior transactions.
            if hybrid_element_offsets:
                template += textwrap.dedent("""\
                    and.b64 ${b64:alternating_alignment}, ${b64:row}, 16;
                    setp.ne.u64 ${pred:alternating_odd}, ${b64:alternating_alignment}, 0;
                """)
            else:
                template += textwrap.dedent("""\
                    mov.u32 ${b32:alternating_bid}, %ctaid.x;
                    and.b32 ${b32:alternating_parity}, ${b32:alternating_bid}, 1;
                    setp.ne.u32 ${pred:alternating_odd}, ${b32:alternating_parity}, 0;
                """)
        operands = [self.value_map[op.dst], self.value_map[op.src]]
        operand_names = ["dst", "src"]
        if hybrid_n_tiles is not None:
            if (
                hybrid_split_rows is not None
                or hybrid_split_tile is not None
                or hybrid_element_offsets
            ):
                narrow, half = (
                    self.value_map[value] for value in op.indices
                )
                operands.extend([narrow, half])
                operand_names.extend(["hybrid_narrow", "hybrid_half"])
                template += textwrap.dedent("""\
                    setp.ne.u32 ${pred:hybrid_narrow}, ${b32:operand.hybrid_narrow.val}, 0;
                    mov.u32 ${b32:rhs_half}, ${b32:operand.hybrid_half.val};
                    mul.wide.u32 ${b64:hybrid_col_offset}, ${b32:rhs_half}, 256;
                    add.u64 ${b64:row}, ${b64:row}, ${b64:hybrid_col_offset};
                """)
            else:
                half_instruction = (
                    f"shr.b32 ${{b32:rhs_half}}, ${{b32:hybrid_bid}}, "
                    f"{hybrid_n_tiles.bit_length() - 1};"
                    if hybrid_n_tiles & (hybrid_n_tiles - 1) == 0
                    else (
                        f"div.u32 ${{b32:rhs_half}}, ${{b32:hybrid_bid}}, "
                        f"{hybrid_n_tiles};"
                    )
                )
                template += textwrap.dedent("""\
                    mov.u32 ${b32:hybrid_bid}, %ctaid.x;
                    setp.ge.u32 ${pred:hybrid_narrow}, ${b32:hybrid_bid}, __SPLIT_START__;
                    __HALF_INSTRUCTION__
                    mul.wide.u32 ${b64:hybrid_col_offset}, ${b32:rhs_half}, 256;
                    add.u64 ${b64:row}, ${b64:row}, ${b64:hybrid_col_offset};
                """).replace(
                    "__SPLIT_START__", str(hybrid_n_tiles - 1),
                ).replace("__HALF_INSTRUCTION__", half_instruction)
        for col in range(0, src_type.sizes[1], 64):
            valid_cols = min(64, src_type.sizes[1] - col)
            if dst_type.dtype.name == "f32":
                stores = "\n".join(
                    f"{store_predicate}st.global.v4.b32 "
                    f"[${{b64:row}}+{col * 4 + i * 4}], "
                    f"{{{', '.join(values[i:i + 4])}}};"
                    for i in range(0, valid_cols, 4)
                )
            else:
                conversions = []
                packed = []
                for i in range(0, valid_cols, 2):
                    packed_reg = f"${{b32:packed{i // 2}}}"
                    packed.append(packed_reg)
                    conversions.append(
                        f"cvt.rn.bf16x2.f32 {packed_reg}, "
                        f"{values[i + 1]}, {values[i]};"
                    )
                def vector_stores(
                    starts_and_widths,
                    predicate=store_predicate,
                ):
                    stores = []
                    for start, width in starts_and_widths:
                        # PTX 8.8 permits the inline L2 eviction-priority
                        # qualifier for v8.b32 stores on sm_100+. Narrow v4
                        # edge stores keep the ordinary opcode because that
                        # L2 encoding is not supported for them. L1 policies
                        # are legal for both vector widths.
                        l1_qualifier = (
                            f".L1::{l1_cache_policy}"
                            if l1_cache_policy is not None else ""
                        )
                        l2_qualifier = (
                            f".L2::{l2_eviction_priority}"
                            if l2_eviction_priority is not None and width == 8
                            else ""
                        )
                        stores.append(
                            f"{predicate}st.global{l1_qualifier}"
                            f"{l2_qualifier}.v{width}.b32 "
                            f"[${{b64:row}}+{col * 2 + start * 4}], "
                            f"{{{', '.join(packed[start:start + width])}}};"
                        )
                    return stores

                if alternating_store_alignment:
                    even_groups = [
                        (start, min(8, len(packed) - start))
                        for start in range(0, len(packed), 8)
                    ]
                    odd_groups = [(0, 4)] + [
                        (start, min(8, len(packed) - start))
                        for start in range(4, len(packed), 8)
                    ]
                    stores = "\n".join([
                        *conversions,
                        f"@${{pred:alternating_odd}} bra ${{label:alternating_odd_{col}}};",
                        *vector_stores(even_groups),
                        f"bra ${{label:alternating_done_{col}}};",
                        f"${{label:alternating_odd_{col}}}:",
                        *vector_stores(odd_groups),
                        f"${{label:alternating_done_{col}}}:",
                    ])
                else:
                    groups = [
                        (start, min(8, len(packed) - start))
                        for start in range(0, len(packed), 8)
                    ]
                    if predicated_tail_stores:
                        assert isinstance(hybrid_n_width, int)
                        tail_dwords = max(
                            0,
                            min(len(packed), (hybrid_n_width - col) // 2),
                        )
                        split_groups = []
                        for start, width in groups:
                            end = start + width
                            if start < tail_dwords < end:
                                split_groups.extend([
                                    (start, tail_dwords - start, False),
                                    (tail_dwords, end - tail_dwords, True),
                                ])
                            else:
                                split_groups.append(
                                    (start, width, start >= tail_dwords)
                                )
                        stores = "\n".join([
                            *conversions,
                            *(
                                vector_stores(
                                    [(start, width)],
                                    predicate=(
                                        "@!${pred:hybrid_narrow} "
                                        if tail_only else ""
                                    ),
                                )[0]
                                for start, width, tail_only in split_groups
                            ),
                        ])
                    else:
                        stores = "\n".join([
                            *conversions,
                            *vector_stores(groups),
                        ])
                if (
                    not predicated_tail_stores
                    and
                    isinstance(hybrid_n_width, int)
                    and col < hybrid_n_width < col + valid_cols
                ):
                    narrow_cols = hybrid_n_width - col
                    narrow_conversions = []
                    narrow_packed = []
                    for i in range(0, narrow_cols, 2):
                        packed_reg = f"${{b32:packed{i // 2}}}"
                        narrow_packed.append(packed_reg)
                        narrow_conversions.append(
                            f"cvt.rn.bf16x2.f32 {packed_reg}, "
                            f"{values[i + 1]}, {values[i]};"
                        )
                    narrow_even_groups = [
                        (start, min(8, len(narrow_packed) - start))
                        for start in range(0, len(narrow_packed), 8)
                    ]
                    narrow_odd_groups = [(0, 4)] + [
                        (start, min(8, len(narrow_packed) - start))
                        for start in range(4, len(narrow_packed), 8)
                    ]
                    if alternating_store_alignment and hybrid_element_offsets:
                        narrow_stores = [
                            f"@${{pred:alternating_odd}} bra "
                            f"${{label:hybrid_partial_odd_{col}}};",
                            *vector_stores(narrow_even_groups),
                            f"bra ${{label:hybrid_partial_aligned_{col}}};",
                            f"${{label:hybrid_partial_odd_{col}}}:",
                            *vector_stores(narrow_odd_groups),
                            f"${{label:hybrid_partial_aligned_{col}}}:",
                        ]
                    else:
                        narrow_stores = vector_stores(
                            narrow_odd_groups
                            if alternating_store_alignment
                            else narrow_even_groups
                        )
                    stores = "\n".join([
                        f"@${{pred:hybrid_narrow}} bra ${{label:hybrid_partial_{col}}};",
                        stores,
                        f"bra ${{label:hybrid_partial_done_{col}}};",
                        f"${{label:hybrid_partial_{col}}}:",
                        *narrow_conversions,
                        *narrow_stores,
                        f"${{label:hybrid_partial_done_{col}}}:",
                    ])
            template += textwrap.dedent(f"""\
                add.u32 ${{b32:taddr_chunk}}, ${{b32:taddr_row}}, {col};
                tcgen05.ld.sync.aligned.32x32b.x64.b32 {{{load_values}}}, [${{b32:taddr_chunk}}];
                tcgen05.wait::ld.sync.aligned;
                {stores}
            """)
            if (
                hybrid_n_tiles is not None
                and not predicated_tail_stores
                and isinstance(hybrid_n_width, int)
                and col + valid_cols >= hybrid_n_width
            ):
                template += (
                    "@${pred:hybrid_narrow} bra ${label:hybrid_n_done};\n"
                )
        if hybrid_n_tiles is not None:
            template += "${label:hybrid_n_done}:\n"
        new_op = PTXTemplateOp.build(
            operands=operands,
            result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []

    @lowers(AwaitOp)
    def lower_await(
        self, op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        lock = self.value_map[op.lock]
        kind = op.lock.type.kind
        if kind == "mbar":
            return self._lower_await_mbar(op, lock)
        if kind == "wgmma":
            return self._lower_await_wgmma(op)
        if kind == "tcgen05":
            return self._lower_await_mbar(op, lock)
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
        if scope == "thread":
            return [], []  # per-thread ordering is implicit in program order
        if scope == "block":
            template = "bar.sync 0;\n"
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
        else:
            raise NotImplementedError(
                f"lower_barrier: unsupported barrier scope {scope!r}"
            )
        new_op = PTXTemplateOp.build(
            operands=[],
            result_types=[],
            template=template,
            operand_names=[],
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
        template = textwrap.dedent(r"""
            mov.u32 ${b32:ticks}, 0x989680;
            ${label:wait}:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ${pred:mbar_done}, [${b32:operand.lock.val}], ${b32:operand.lock.parity}, ${b32:ticks};
            @!${pred:mbar_done} bra ${label:wait};
            xor.b32 ${b32:operand.lock.parity}, ${b32:operand.lock.parity}, 1;
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
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]

        is_load = isinstance(op.dst.type, RmemTensorType)
        gmem_ir_type = op.src.type if is_load else op.dst.type
        rmem_type = dst_value.type if is_load else src_value.type
        assert isinstance(gmem_ir_type, GmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        gmem_label = "src" if is_load else "dst"
        dtype_bytes = gmem_ir_type.dtype.byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = f"${{b64:operand.{gmem_label}.ptr}}"

        for flat_idx, idx in enumerate(
            itertools.product(*[range(s) for s in rmem_type.sizes])
        ):
            offset = sum(
                c * s for c, s in zip(idx, gmem_ir_type.strides)
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
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]

        is_load = isinstance(op.dst.type, SmemTensorType)
        smem_type = op.dst.type if is_load else op.src.type
        gmem_type = op.src.type if is_load else op.dst.type
        assert isinstance(smem_type, SmemTensorType)
        assert isinstance(gmem_type, GmemTensorType)
        tcgen05_layout = smem_type.attrs.get("tcgen05_layout")
        if tcgen05_layout is not None:
            if not is_load:
                raise NotImplementedError(
                    "tcgen05 shared layouts currently support loads only"
                )
            return self._lower_move_to_smem_tcgen05(
                op, dst_value, src_value, tcgen05_layout,
            )
        gmem_label = "src" if is_load else "dst"
        smem_label = "dst" if is_load else "src"

        assert gmem_type.dtype == smem_type.dtype, (
            f"_lower_move_gmem_smem: dtype mismatch; "
            f"gmem {gmem_type.dtype} vs smem {smem_type.dtype}"
        )
        dtype_bytes = gmem_type.dtype.byte_size
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
        co_size = smem_type.sizes[co_dim]
        gmem_co_stride_bytes = gmem_type.strides[-1] * dtype_bytes
        smem_co_stride_bytes = smem_type.strides[-1] * dtype_bytes

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
            seq_size = smem_type.sizes[seq_dim]
            gmem_seq_stride_bytes = gmem_type.strides[seq_dim] * dtype_bytes
            smem_seq_stride_bytes = smem_type.strides[seq_dim] * dtype_bytes
            for seq_idx in range(seq_size):
                g_off = seq_idx * gmem_seq_stride_bytes
                s_off = seq_idx * smem_seq_stride_bytes
                template += textwrap.dedent(f"""\
                    add.u64 ${{b64:co_gaddr}}, ${{b64:co_gpar}}, {g_off};
                    add.u32 ${{b32:co_saddr}}, ${{b32:co_spar}}, {s_off};
                    {ld_inst} ${{{suffix}:co_tmp}}, [{ld_addr}];
                    {st_inst} [{st_addr}], ${{{suffix}:co_tmp}};
                """)

        template += textwrap.dedent(f"""\
            add.u32 ${{b32:co_idx}}, ${{b32:co_idx}}, ${{b32:co_ntid}};
            bra ${{label:co_idx_loop}};
            ${{label:co_end}}:
        """)

        new_op = PTXTemplateOp.build(
            operands=operands, result_types=[], template=template,
            operand_names=operand_names, result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_to_smem_tcgen05(
        self,
        op: MoveOp,
        dst: Value,
        src: Value,
        layout: object,
    ) -> tuple[list[Op], list[object]]:
        """Stage one operand in the compact SM100 layout used by tcgen05.

        These two layouts are the first supported 128x64x16 BF16 instruction
        tile. All 128 threads perform equal-width global loads and shared
        stores; there are no dedicated producer warps.
        """
        if self.target.tensor_core_kind != "tcgen05":
            raise ValueError(
                "tcgen05 shared layout requires a Blackwell target; "
                f"got {self.target.architecture!r}"
            )
        dst_type, src_type = op.dst.type, op.src.type
        assert isinstance(dst_type, SmemTensorType)
        assert isinstance(src_type, (GmemTensorType, SmemTensorType))
        source_space = "global" if isinstance(src_type, GmemTensorType) else "shared"
        source_ptr_type = "b64" if isinstance(src_type, GmemTensorType) else "b32"
        if layout == "lhs_128x16" and (
            tuple(dst_type.sizes), tuple(src_type.sizes), dst_type.swizzle
        ) == ((128, 16), (128, 16), "32B"):
            row_stride = src_type.strides[0]
            second_half_bytes = 64 * row_stride * src_type.dtype.byte_size
            if source_space == "global":
                addresses = textwrap.dedent(f"""\
                    mad.wide.u32 ${{b64:g0}}, ${{b32:elem}}, 2, ${{b64:operand.src.ptr}};
                    add.s64 ${{b64:g1}}, ${{b64:g0}}, {second_half_bytes};
                """)
            else:
                addresses = textwrap.dedent(f"""\
                    mul.lo.u32 ${{b32:byte_offset}}, ${{b32:elem}}, 2;
                    add.u32 ${{b32:g0}}, ${{b32:operand.src.ptr}}, ${{b32:byte_offset}};
                    add.u32 ${{b32:g1}}, ${{b32:g0}}, {second_half_bytes};
                """)
            template = textwrap.dedent(f"""\
                mov.u32 ${{b32:tid}}, %tid.x;
                shr.u32 ${{b32:row}}, ${{b32:tid}}, 1;
                and.b32 ${{b32:half}}, ${{b32:tid}}, 1;
                shl.b32 ${{b32:half}}, ${{b32:half}}, 3;
                mad.lo.u32 ${{b32:elem}}, ${{b32:row}}, {row_stride}, ${{b32:half}};
                {addresses}
                ld.{source_space}.v4.b32 {{${{b32:v0}}, ${{b32:v1}}, ${{b32:v2}}, ${{b32:v3}}}}, [${{{source_ptr_type}:g0}}];
                ld.{source_space}.v4.b32 {{${{b32:v4}}, ${{b32:v5}}, ${{b32:v6}}, ${{b32:v7}}}}, [${{{source_ptr_type}:g1}}];
                shl.b32 ${{b32:x4}}, ${{b32:tid}}, 4;
                and.b32 ${{b32:x1}}, ${{b32:x4}}, 1904;
                bfe.s32 ${{b32:x2}}, ${{b32:tid}}, 3, 1;
                and.b32 ${{b32:x2}}, ${{b32:x2}}, 144;
                xor.b32 ${{b32:x3}}, ${{b32:x2}}, ${{b32:x1}};
                add.u32 ${{b32:s0}}, ${{b32:operand.dst.ptr}}, ${{b32:x3}};
                add.u32 ${{b32:s1}}, ${{b32:s0}}, 2048;
                st.shared.v4.b32 [${{b32:s0}}], {{${{b32:v0}}, ${{b32:v1}}, ${{b32:v2}}, ${{b32:v3}}}};
                st.shared.v4.b32 [${{b32:s1}}], {{${{b32:v4}}, ${{b32:v5}}, ${{b32:v6}}, ${{b32:v7}}}};
            """)
        elif layout == "rhs_64x16" and (
            tuple(dst_type.sizes), tuple(src_type.sizes), dst_type.swizzle
        ) == ((64, 16), (64, 16), "128B"):
            row_stride = src_type.strides[0]
            loads = "\n".join(
                (
                    textwrap.dedent(f"""\
                        add.u32 ${{b32:n}}, ${{b32:n_base}}, {i};
                        mad.lo.u32 ${{b32:elem{i}}}, ${{b32:n}}, {row_stride}, ${{b32:k}};
                        mad.wide.u32 ${{b64:g{i}}}, ${{b32:elem{i}}}, 2, ${{b64:operand.src.ptr}};
                        ld.global.b16 ${{b16:v{i}}}, [${{b64:g{i}}}];
                        st.shared.b16 [${{b32:s}}+{i * 2}], ${{b16:v{i}}};
                    """)
                    if source_space == "global"
                    else textwrap.dedent(f"""\
                        add.u32 ${{b32:n}}, ${{b32:n_base}}, {i};
                        mad.lo.u32 ${{b32:elem{i}}}, ${{b32:n}}, {row_stride}, ${{b32:k}};
                        mul.lo.u32 ${{b32:byte{i}}}, ${{b32:elem{i}}}, 2;
                        add.u32 ${{b32:g{i}}}, ${{b32:operand.src.ptr}}, ${{b32:byte{i}}};
                        ld.shared.b16 ${{b16:v{i}}}, [${{b32:g{i}}}];
                        st.shared.b16 [${{b32:s}}+{i * 2}], ${{b16:v{i}}};
                    """)
                ).rstrip()
                for i in range(8)
            )
            template = textwrap.dedent("""\
                mov.u32 ${b32:tid}, %tid.x;
                shr.u32 ${b32:k}, ${b32:tid}, 3;
                and.b32 ${b32:n_base}, ${b32:tid}, 7;
                shl.b32 ${b32:n_base}, ${b32:n_base}, 3;
                shl.b32 ${b32:x4}, ${b32:tid}, 4;
                and.b32 ${b32:x1}, ${b32:x4}, 2032;
                shl.b32 ${b32:x2}, ${b32:tid}, 1;
                and.b32 ${b32:x2}, ${b32:x2}, 112;
                xor.b32 ${b32:x3}, ${b32:x1}, ${b32:x2};
                add.u32 ${b32:s}, ${b32:operand.dst.ptr}, ${b32:x3};
            """) + "\n" + loads + "\n"
        elif layout == "rhs_32x16_cta2" and (
            tuple(dst_type.sizes), tuple(src_type.sizes), dst_type.swizzle
        ) == ((64, 16), (32, 16), "128B"):
            row_stride = src_type.strides[0]
            loads = "\n".join(
                textwrap.dedent(f"""\
                    add.u32 ${{b32:n}}, ${{b32:n_base}}, {i};
                    mad.lo.u32 ${{b32:elem{i}}}, ${{b32:n}}, {row_stride}, ${{b32:k}};
                    mad.wide.u32 ${{b64:g{i}}}, ${{b32:elem{i}}}, 2, ${{b64:operand.src.ptr}};
                    ld.global.b16 ${{b16:v{i}}}, [${{b64:g{i}}}];
                    st.shared.b16 [${{b32:s}}+{i * 2}], ${{b16:v{i}}};
                """).rstrip()
                for i in range(8)
            )
            # Group-2 MMA sources half of B's N columns from each peer CTA.
            # Sixty-four threads populate the local half in the same physical
            # positions used by the first 32 columns of rhs_64x16.
            template = textwrap.dedent("""\
                mov.u32 ${b32:tid}, %tid.x;
                setp.ge.u32 ${pred:done}, ${b32:tid}, 64;
                @${pred:done} bra ${label:end};
                shr.u32 ${b32:k}, ${b32:tid}, 2;
                and.b32 ${b32:n_group}, ${b32:tid}, 3;
                shl.b32 ${b32:n_base}, ${b32:n_group}, 3;
                shl.b32 ${b32:logical_tid}, ${b32:k}, 3;
                or.b32 ${b32:logical_tid}, ${b32:logical_tid}, ${b32:n_group};
                shl.b32 ${b32:x4}, ${b32:logical_tid}, 4;
                and.b32 ${b32:x1}, ${b32:x4}, 2032;
                shl.b32 ${b32:x2}, ${b32:logical_tid}, 1;
                and.b32 ${b32:x2}, ${b32:x2}, 112;
                xor.b32 ${b32:x3}, ${b32:x1}, ${b32:x2};
                add.u32 ${b32:s}, ${b32:operand.dst.ptr}, ${b32:x3};
            """) + "\n" + loads + textwrap.dedent("""\
                ${label:end}:
            """)
        else:
            raise NotImplementedError(
                f"unsupported tcgen05 shared layout {layout!r}: "
                f"{src_type} -> {dst_type}"
            )
        new_op = PTXTemplateOp.build(
            operands=[dst, src], result_types=[], template=template,
            operand_names=["dst", "src"], result_names=[],
            description=str(op),
        )
        return [new_op], []

    def _lower_move_wrmem_mma_to_smem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower wrmem → smem store (per-thread scalar st.shared).

        Each thread stores its fragment registers to smem at addresses
        determined by the mma output layout (gid, lid from laneid).
        Same-dtype only — use convert first for type changes.
        """

        # TODO: this needs unique wrmem layouts
        raise NotImplementedError("this needs unique wrmem layouts")

        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]
        smem_type = dst_value.type
        wrmem_type = src_value.type
        assert isinstance(smem_type, SmemTensorType)
        assert isinstance(wrmem_type, WrmemTensorType)
        assert smem_type.dtype == wrmem_type.dtype, (
            f"_lower_move_wrmem_to_smem: dtype mismatch; "
            f"smem {smem_type.dtype} vs wrmem {wrmem_type.dtype}. "
            f"Use convert first."
        )

        assert wrmem_type.rank == 2, (
            f"_lower_move_wrmem_to_smem: need rank 2; got {wrmem_type.rank}"
        )
        m_size, n_size = wrmem_type.sizes
        m_stride, n_stride = smem_type.strides
        assert n_stride == 1, (
            f"_lower_move_wrmem_to_smem: smem col stride must be 1; got {n_stride}"
        )
        dtype_bytes = smem_type.dtype.byte_size
        suffix = _ptx_suffix(smem_type.dtype)
        m_stride_bytes = m_stride * dtype_bytes
        if m_size != 16 or n_size % 8 != 0:
            raise NotImplementedError(
                f"_lower_move_wrmem_to_smem: need m=16, n%%8=0; got {m_size}x{n_size}"
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



    def _lower_move_wrmem_wgmma(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        register_starts = _stmatrix_register_starts(op)
        if register_starts is None:
            return self._lower_move_wrmem_wgmma_to_addrmem(op)
        return self._lower_move_wrmem_wgmma_to_stmatrix(op, register_starts)

    def _lower_move_wrmem_wgmma_to_addrmem(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Store a wgmma fragment to shared or global address memory.

        The register layout and static element offsets are identical for both
        destinations; only the address width and PTX memory space differ. The
        destination is the writer warp group's own tile, so its shape matches
        the fragment -- the block-wide SMEM staging tile is tiled by warp-group
        id upstream (see promote_buffers._build_tma_writeback), so the row offset
        is already folded into the destination pointer.
        """
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]
        dst_type = op.dst.type
        wrmem_type = op.src.type
        assert isinstance(dst_type, (GmemTensorType, SmemTensorType))
        assert isinstance(wrmem_type, WrmemTensorType)
        assert wrmem_type.layout == "wgmma_layout_0"
        assert wrmem_type.rank == 2 and dst_type.rank == 2
        assert isinstance(wrmem_type.dtype, (BF16Type, F16Type)), (
            f"wgmma store: expected bf16/f16; "
            f"got {wrmem_type.dtype}. Use convert first for f32 accumulators."
        )
        assert dst_type.dtype == wrmem_type.dtype, (
            f"wgmma store: dtype mismatch; "
            f"destination {dst_type.dtype} vs wrmem {wrmem_type.dtype}"
        )

        m_dim, n_dim = wrmem_type.sizes
        col_start = int(op.attrs.get("wgmma_col_start", 0))
        dst_m, dst_n = dst_type.sizes
        assert dst_m == m_dim and 0 <= col_start < n_dim, (
            f"wgmma store: invalid destination shape {list(dst_type.sizes)} "
            f"and column start {col_start} for fragment {[m_dim, n_dim]}"
        )
        assert col_start + dst_n <= n_dim
        assert m_dim % 64 == 0, (
            f"wgmma store: m must be a multiple of 64; got {m_dim}"
        )
        assert n_dim % 16 == 0 and 8 <= n_dim <= 256, (
            f"wgmma store: n must be a multiple of 16 in [8, 256]; got {n_dim}"
        )
        dtype_bytes = dst_type.dtype.byte_size
        row_stride_bytes = dst_type.strides[0] * dtype_bytes

        new_ops, byte_offset = _build_wgmma_layout_0_dynamic_offset(
            row_stride_bytes, dtype_bytes,
        )
        if isinstance(dst_type, GmemTensorType):
            template = textwrap.dedent(r"""
                cvt.u64.u32 ${b64:off64}, ${b32:operand.offset.val};
                add.u64 ${b64:base}, ${b64:operand.dst.ptr}, ${b64:off64};
            """)
            address = "${b64:base}"
            instruction = "st.global.b32"
        else:
            template = (
                "add.u32 ${b32:base}, ${b32:operand.dst.ptr}, "
                "${b32:operand.offset.val};\n"
            )
            address = "${b32:base}"
            instruction = "st.shared.b32"

        num_registers = wrmem_type.num_elements_per_thread // 2
        for reg in range(num_registers):
            di, dj = _get_wgmma_layout_0_static_offsets(reg, n_dim)
            if not col_start <= dj < col_start + dst_n:
                continue
            offset = di * row_stride_bytes + (dj - col_start) * dtype_bytes
            template += (
                f"{instruction} [{address}+{offset}], "
                f"${{b32:operand.src.r{reg}}};\n"
            )

        new_op = PTXTemplateOp.build(
            operands=[dst_value, src_value, byte_offset],
            result_types=[], template=template,
            operand_names=["dst", "src", "offset"], result_names=[],
            description=str(op),
        )
        new_ops.append(new_op)
        return new_ops, []

    def _lower_move_wrmem_wgmma_to_stmatrix(
        self, op: MoveOp, register_starts: list[int],
    ) -> tuple[list[Op], list[object]]:
        dst = self.value_map[op.dst]
        src = self.value_map[op.src]
        dst_type = op.dst.type
        src_type = op.src.type
        assert isinstance(dst_type, SmemTensorType)
        assert isinstance(src_type, WrmemTensorType)
        col_start = int(op.attrs.get("wgmma_col_start", 0))
        n_dim = src_type.sizes[1]
        root_type = root_buffer(op.dst).type
        assert isinstance(root_type, SmemTensorType)
        root_sizes = root_type.sizes
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
        template = writer_setup + textwrap.dedent(f"""\
            mov.u32 ${{b32:st_lane}}, %laneid;
            mov.u32 ${{b32:st_tid}}, %tid.x;
            and.b32 ${{b32:st_tid}}, ${{b32:st_tid}}, 127;
            shr.u32 ${{b32:st_warp}}, ${{b32:st_tid}}, 5;
            and.b32 ${{b32:st_row}}, ${{b32:st_lane}}, 7;
            mad.lo.u32 ${{b32:st_row}}, ${{b32:st_warp}}, 16, ${{b32:st_row}};
            shr.u32 ${{b32:st_col}}, ${{b32:st_lane}}, 3;
            shl.b32 ${{b32:st_col}}, ${{b32:st_col}}, 3;
            mad.lo.u32 ${{b32:st_off}}, ${{b32:st_row}}, 64, ${{b32:st_col}};
            shl.b32 ${{b32:st_off}}, ${{b32:st_off}}, 1;
        """)
        groups: list[tuple[list[int], int]] = []
        for i in range(0, len(register_starts), 2):
            first0 = register_starts[i]
            first1 = register_starts[i + 1]
            di, dj = _get_wgmma_layout_0_static_offsets(first0, n_dim)
            next_di, next_dj = _get_wgmma_layout_0_static_offsets(
                first1, n_dim,
            )
            assert next_di == di and next_dj == dj + 16
            offset0 = _get_stmatrix_store_static_offset(
                first0, n_dim, col_start, root_sizes[0],
            )
            offset1 = _get_stmatrix_store_static_offset(
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

    def _lower_move_smem_to_wrmem_mma(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower smem → wrmem move to ldmatrix.sync (warp-collective).

        All 32 lanes participate: each provides an smem row address,
        and the hardware shuffles data across lanes into the mma
        fragment layout. Uses `.trans` when `attrs={trans: True}`
        (B operands needing col-major fragments from row-major smem).
        """

        # TODO: this needs unique wrmem layouts
        raise NotImplementedError("this needs unique wrmem layouts")

        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]
        wrmem_type = dst_value.type
        smem_type = src_value.type
        assert isinstance(wrmem_type, WrmemTensorType)
        assert isinstance(smem_type, SmemTensorType)

        dtype_bytes = smem_type.dtype.byte_size
        m_stride_bytes = smem_type.strides[0] * dtype_bytes
        trans = ".trans" if op.attrs.get("trans") else ""
        n_regs = wrmem_type.num_elements // (wrmem_type.warps * 32 * 2)

        if n_regs == 4:
            x_count = "x4"
        elif n_regs == 2:
            x_count = "x2"
        elif n_regs == 1:
            x_count = "x1"
        else:
            raise NotImplementedError(
                f"_lower_move_smem_to_wrmem: unsupported register count {n_regs}"
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
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]

        is_load = isinstance(op.dst.type, RmemTensorType)
        smem_ir_type = op.src.type if is_load else op.dst.type
        rmem_type = dst_value.type if is_load else src_value.type
        assert isinstance(smem_ir_type, SmemTensorType)
        assert isinstance(rmem_type, RmemTensorType)
        smem_label = "src" if is_load else "dst"
        dtype_bytes = smem_ir_type.dtype.byte_size
        suffix = _ptx_suffix(rmem_type.dtype)

        operands: list[Value] = [dst_value, src_value]
        operand_names: list[str] = ["dst", "src"]

        template = ""
        base_reg = f"${{b32:operand.{smem_label}.ptr}}"

        for flat_idx, idx in enumerate(
            itertools.product(*[range(s) for s in rmem_type.sizes])
        ):
            offset = sum(
                c * s for c, s in zip(idx, smem_ir_type.strides)
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
        dst_value = self.value_map[op.dst]
        src_value = self.value_map[op.src]
        idx_values = [self.value_map[v] for v in op.indices]
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
        assert lock_value.type.kind == "mbar", (
            f"_lower_async_tma_loads: expected lock kind 'mbar'; got "
            f"{lock_value.type.kind!r}"
        )

        operands: list[Value] = [lock_value]
        operand_names: list[str] = ["lock"]
        total_bytes = 0
        load_lines: list[str] = []
        has_cluster_leader_load = any(
            tma.attrs.get("cluster_leader") for tma in tmas
        )
        for i, tma in enumerate(tmas):
            dst_value = self.value_map[tma.dst]
            desc_value = self.value_map[tma.src]
            assert isinstance(desc_value.type, TmaDescType), (
                f"_lower_async_tma_loads: load src must be a TMA descriptor; "
                f"got {desc_value.type}"
            )
            desc_type = desc_value.type
            total_bytes += (
                desc_type.tile_num_elements * desc_type.gmem_type.dtype.byte_size
            )
            idx_values = [self.value_map[v] for v in tma.indices]
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
            multicast_mask = tma.attrs.get("multicast_mask")
            opcode = (
                f"cp.async.bulk.tensor.{tensor_rank}d.shared::cluster.global."
                "mbarrier::complete_tx::bytes"
            )
            arguments = (
                f"[${{b32:operand.{dn}.ptr}}], "
                f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                f"[${{b32:operand.lock.val}}]"
            )
            if multicast_mask is not None:
                opcode += ".multicast::cluster"
                arguments += f", 0x{int(multicast_mask):x}"
            instruction = self._tma_instruction(
                tma,
                opcode,
                arguments,
                operands,
                operand_names,
                f"cache_policy{i}",
            )
            if tma.attrs.get("cluster_leader"):
                instruction = "@${pred:cluster_leader} " + instruction
            load_lines.append(instruction)

        # Plain-string lines (literal ${...} placeholders); only load_lines and
        # the byte count are interpolated, so we avoid f-string/dedent brace and
        # multi-line-indent pitfalls by joining a line list.
        #
        # The buffer-reuse WAR (t0's reload vs prior readers of the smem tile) is
        # handled by insert_syncs, which emits a CTA barrier before this async.
        # A multicast destination also belongs to neighboring CTAs, so all CTAs
        # in the cluster must rendezvous before the elected CTA can overwrite it.
        # On the first iteration this same rendezvous makes every remote
        # mbarrier initialization visible before the multicast transaction.
        lines = [
            *(
                [
                    "barrier.cluster.arrive.aligned;",
                    "barrier.cluster.wait.aligned;",
                ]
                if has_cluster_leader_load else []
            ),
            "mov.u32 ${b32:tid_x}, %tid.x;",
            "mov.u32 ${b32:tid_y}, %tid.y;",
            "or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_y};",
            "mov.u32 ${b32:tid_z}, %tid.z;",
            "or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_z};",
            "setp.ne.s32 ${pred:not_t0}, ${b32:tid_x}, 0;",
            "@${pred:not_t0} bra ${label:after_load};",
            *(
                [
                    "mov.u32 ${b32:cluster_rank}, %cluster_ctarank;",
                    "setp.eq.u32 ${pred:cluster_leader}, ${b32:cluster_rank}, 0;",
                ]
                if has_cluster_leader_load else []
            ),
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

    tid_op = ThreadIdsOp.build()
    ops.append(tid_op)
    tid_x = tid_op.x

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


def _get_wgmma_layout_0_static_offsets(reg: int, n_dim: int = 128) -> tuple[int, int]:
    """(row, col) offset for register `reg` in wgmma_layout_0 (4 warps).

    Each b32 register holds a pair of adjacent bf16 column elements.
    n_dim determines the number of column groups (n_dim // 16).
    """
    n_groups = n_dim // 16
    m, w = divmod(reg // 4, n_groups)
    di, dj = [(0, 0), (8, 0), (0, 8), (8, 8)][reg % 4]
    return (m * 64 + di, w * 16 + dj)


def _get_stmatrix_store_static_offset(
    reg: int, n_dim: int, col_start: int, dst_m: int,
) -> int:
    """Byte offset for a register in the selected 64-column blocked layout."""
    di, dj = _get_wgmma_layout_0_static_offsets(reg, n_dim)
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
        and isinstance(src_type, WrmemTensorType)
        and src_type.layout == "wgmma_layout_0"
        and isinstance(src_type.dtype, (BF16Type, F16Type))
        and dst_type.dtype == src_type.dtype
        and dst_type.rank == 2
        and dst_type.strides[-1] == 1
        and dst_type.swizzle == "128B"
        and layout == (64, "block_major")
    ):
        return None

    root_type = root_buffer(op.dst).type
    if (
        not isinstance(root_type, SmemTensorType)
        or root_type.rank != 2
        or not root_type.is_canonical_strides
        or root_type.sizes[0] % 64
    ):
        return None

    m_dim, n_dim = src_type.sizes
    col_start = int(op.attrs.get("wgmma_col_start", 0))
    dst_m, dst_n = dst_type.sizes
    if (
        dst_m != m_dim
        or col_start < 0
        or col_start % 16
        or col_start + dst_n > n_dim
    ):
        return None
    if (
        m_dim % 64
        or n_dim % 16
        or dst_n % 64
        or not (8 <= n_dim <= 256)
    ):
        return None

    num_registers = src_type.num_elements_per_thread // 2
    if num_registers % 4:
        return None
    register_starts = []
    for reg in range(0, num_registers, 4):
        _, dj = _get_wgmma_layout_0_static_offsets(reg, n_dim)
        if col_start <= dj < col_start + dst_n:
            register_starts.append(reg)
    return register_starts if len(register_starts) % 2 == 0 else None


def _ptx_suffix(dtype) -> str:
    """PTX type suffix for a dtype. bf16 uses b16 (no native suffix)."""
    name = dtype.name
    return "b16" if name == "bf16" else name


def _ptx_literal(dtype, value) -> str:
    name = dtype.name
    if name == "f32":
        return "0f" + struct.pack(">f", float(value)).hex().upper()
    if name == "f64":
        return "0d" + struct.pack(">d", float(value)).hex().upper()
    if name == "bf16":
        bits = struct.unpack(">I", struct.pack(">f", float(value)))[0] >> 16
        return f"0x{bits:04X}"
    raise NotImplementedError(f"_ptx_literal: dtype {name}")


def lower_to_ptx_template(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToPtxTemplate(module.ptx_target).run(module)
