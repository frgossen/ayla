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

from dataclasses import dataclass

from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BF16Type,
    ConvertOp,
    F16Type,
    F32Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    LockType,
    MatmulOp,
    KernelOp,
    ModuleOp,
    MoveOp,
    PTXTemplateOp,
    PtrAddOp,
    PtrType,
    RmemTensorType,
    SmemTensorType,
    StMatrixStoreOp,
    ThreadIdsOp,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    ToPtrOp,
    WrmemTensorType,
    wgmma_layout_0_static_offsets,
)
from ..base.ir import Op, Value, find_ops
from ..base.lowering_pass import LoweringPass


class LowerToPtxTemplate(LoweringPass):
    """Pass that rewrites high-level ops to ptx_template ops."""

    def __init__(self) -> None:
        super().__init__()
        self.block_dim: tuple[int | None, int | None, int | None] = (
            None, None, None,
        )
        self.cache_policy_values: dict[str, Value] = {}
        self.register(KernelOp, type(self).lower_kernel)
        self.register(AllocOp, type(self).lower_alloc)
        self.register(MoveOp, type(self).lower_move)
        self.register(StMatrixStoreOp, type(self).lower_stmatrix_store)
        self.register(MatmulOp, type(self).lower_matmul)
        self.register(AsyncOp, type(self).lower_async)
        self.register(AwaitOp, type(self).lower_await)
        self.register(BarrierOp, type(self).lower_barrier)
        self.register(ConvertOp, type(self).lower_convert)
        self.register(TileOp, type(self).lower_tile)

    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        block_dim = list(op.block_dim)
        if op.num_warp_groups is not None and op.num_warp_groups > 0:
            block_dim[0] = op.num_warp_groups * 128
        self.block_dim = tuple(block_dim)
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
            op.body.entry_block.ops.insert(0, policy_op)
            self.cache_policy_values = dict(zip(cache_hints, policy_op.results))
        self.lower_region(op.body)
        return [op], []

    def _thread_zero_guard(self, prefix: str, done: str) -> list[str]:
        """Branch nonzero threads away, omitting statically singleton axes."""
        axes = [
            axis for axis, extent in zip("xyz", self.block_dim)
            if extent != 1
        ]
        if not axes:
            return []
        accumulator = f"{prefix}_tid"
        lines = [f"mov.u32 ${{b32:{accumulator}}}, %tid.{axes[0]};"]
        for axis in axes[1:]:
            component = f"{prefix}_tid_{axis}"
            lines.extend([
                f"mov.u32 ${{b32:{component}}}, %tid.{axis};",
                f"or.b32 ${{b32:{accumulator}}}, "
                f"${{b32:{accumulator}}}, ${{b32:{component}}};",
            ])
        predicate = f"{prefix}_not_t0"
        lines.extend([
            f"setp.ne.s32 ${{pred:{predicate}}}, "
            f"${{b32:{accumulator}}}, 0;",
            f"@${{pred:{predicate}}} bra ${{label:{done}}};",
        ])
        return lines


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
            root_type = arg.type
        elif isinstance(arg, Value) and isinstance(arg.type, PtrType):
            ptr = arg
            v = op.arg
            while not isinstance(v.type, AddrMemTensorType):
                assert isinstance(v.definition, TileOp)
                v = v.definition.arg
            root_type = v.type
        else:
            raise NotImplementedError(
                f"lower_tile: unexpected arg type {type(arg)}"
            )

        assert isinstance(root_type, AddrMemTensorType)
        dtype_bytes = root_type.dtype.byte_size
        result_sizes = list(op.result().type.sizes)

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

            tile_size_op = IndexConstantOp.build(result_sizes[dim])
            new_ops.append(tile_size_op)
            elem_off = IndexMulOp.build(idx, tile_size_op.result())
            new_ops.append(elem_off)

            layout = root_type.blocked_layout
            if layout is not None and dim == root_type.rank - 1:
                block_op = IndexConstantOp.build(layout.block)
                chunk = IndexDivOp.build(elem_off.result(), block_op.result())
                inner = IndexModOp.build(elem_off.result(), block_op.result())
                padded_op = IndexConstantOp.build(
                    layout.block + layout.padding
                )
                if layout.order == "row_major":
                    chunk_stride = padded_op.result()
                else:
                    assert layout.outer_size is not None
                    rows_op = IndexConstantOp.build(layout.outer_size)
                    chunk_stride_op = IndexMulOp.build(
                        rows_op.result(), padded_op.result(),
                    )
                    new_ops.extend([rows_op, chunk_stride_op])
                    chunk_stride = chunk_stride_op.result()
                chunk_off = IndexMulOp.build(chunk.result(), chunk_stride)
                physical_off = IndexAddOp.build(
                    chunk_off.result(), inner.result(),
                )
                new_ops.extend([
                    block_op, chunk, inner, padded_op, chunk_off, physical_off,
                ])
                elem_off = physical_off

            stride_bytes = root_type.strides[dim] * dtype_bytes
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
        if isinstance(result_type, LockType):
            return self._lower_alloc_lock(op)
        raise NotImplementedError(
            f"lower_alloc: unsupported result type {type(result_type).__name__}"
        )

    def _lower_alloc_lock(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]]:
        lock_type = op.result().type
        assert isinstance(lock_type, LockType)
        if lock_type.kind != "mbar":
            return [], [None]
        lines = [
            ".shared .align 8 .b64 ${smem:bar}[1];",
            "mov.u32 ${b32:result.dst.val}, ${smem:bar};",
            "mov.u32 ${b32:result.dst.parity}, 0;",
            *self._thread_zero_guard("init", "after_init"),
            "mbarrier.init.shared::cta.b64 [${b32:result.dst.val}], 1;",
            "fence.mbarrier_init.release.cluster;",
            "${label:after_init}:",
            "bar.sync 0;",
        ]
        template = "\n".join(lines) + "\n"
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
        n_elements = smem_type.storage_num_elements
        n_bytes = n_elements * smem_type.dtype.byte_size
        template = textwrap.dedent(f"""\
            .shared .align 128 .b8 ${{smem:sym}}[{n_bytes}];
            mov.u32 ${{b32:result.dst.ptr}}, ${{smem:sym}};
        """)
        new_op = PTXTemplateOp.build(
            operands=[], result_types=[smem_type], template=template,
            operand_names=[], result_names=["dst"],
            description=str(op),
        )
        return [new_op], [new_op.result()]

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



    def lower_async(
        self, op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        body_ops = op.body.entry_block.ops
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
        if body_ops and all(isinstance(o, MatmulOp) for o in body_ops):
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
            delta = value.definition.constant_byte_delta(base.definition)
            return delta if delta is not None and delta >= 0 and delta % 16 == 0 else None

        def _desc(name, base, delta, ptr, const, offset=None):
            if delta is not None:
                return f"add.u64 ${{b64:{name}}}, ${{b64:{base}}}, {delta // 16};\n"
            return _build_desc(name, ptr, const, offset)

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

        desc_template = ""
        scale_template = ""
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
            lhs_name = None if lhs_delta is not None else _operand(self.value_map[mm.lhs], "lhs")
            rhs_name = None if rhs_delta is not None else _operand(self.value_map[mm.rhs], "rhs")
            dst_name = _operand(self.value_map[mm.dst], "dst")

            desc_rhs = f"desc_rhs{j}"
            desc_template += _desc(
                desc_rhs, "desc_rhs0", rhs_delta,
                f"operand.{rhs_name}.ptr", desc_const_rhs,
            )
            for m_it in range(m_iters):
                lhs_offset = m_it * 64 * lhs_row_stride_bytes
                desc_lhs = f"desc_lhs{j}_{m_it}"
                desc_template += _desc(
                    desc_lhs, f"desc_lhs0_{m_it}", lhs_delta,
                    f"operand.{lhs_name}.ptr", desc_const_lhs,
                    lhs_offset if m_it > 0 else None,
                )

            if mm.scale_d is None:
                scale_d = "1"
            else:
                scale_name = _operand(self.value_map[mm.scale_d], "scale_d")
                scale_pred = f"scale_d{j}"
                scale_template += (
                    f"setp.ne.u32 ${{pred:{scale_pred}}}, "
                    f"${{b32:operand.{scale_name}.val}}, 0;\n"
                )
                scale_d = f"${{pred:{scale_pred}}}"
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
                      {scale_d}, 1, 1, 0, 0;
                """)
                reg_offset += num_regs_per_wgmma

        # One commit group for the whole batch: single fence, all mma_async, one
        # commit. Hardware serializes same-accumulator mmas, so the await's lone
        # wait_group covers them all. The await also emits an asymmetric named
        # barrier that closes the cross-warp WAR (thread-0 TMA reload vs
        # warpgroup-wide wgmma reads).
        template = (
            "wgmma.fence.sync.aligned;\n"
            + scale_template
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
        if isinstance(dst_type, WrmemTensorType) and isinstance(src_type, SmemTensorType) and dst_type.layout == "mma":
            return self._lower_move_smem_to_wrmem_mma(op)
        if isinstance(dst_type, SmemTensorType) and isinstance(src_type, WrmemTensorType) and src_type.layout == "mma":
            return self._lower_move_wrmem_mma_to_smem(op)
        if (
            isinstance(dst_type, (SmemTensorType, GmemTensorType))
            and isinstance(src_type, WrmemTensorType)
            and src_type.layout == "wgmma_layout_0"
        ):
            return self._lower_move_wrmem_wgmma_to_addrmem(op)
        raise NotImplementedError(
            f"lower_move: unsupported; got "
            f"{type(dst_type).__name__} <- {type(src_type).__name__}"
        )

    def lower_await(
        self, op: AwaitOp,
    ) -> tuple[list[Op], list[object]]:
        lock = self.value_map[op.lock]
        kind = op.lock.type.kind
        if kind == "mbar":
            return self._lower_await_mbar(op, lock)
        if kind == "wgmma":
            return self._lower_await_wgmma(op)
        if kind == "bulk":
            return self._lower_await_bulk(op)
        raise NotImplementedError(
            f"lower_await: unsupported lock kind {kind!r}"
        )

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
            block_threads = (
                math.prod(self.block_dim)
                if all(dim is not None for dim in self.block_dim)
                else None
            )
            count = (
                str(block_threads) if block_threads is not None
                else "${b32:bl_ntid}"
            )
            lines = ["mov.u32 ${b32:bl_tid}, %tid.x;"]
            if block_threads is None:
                lines.append("mov.u32 ${b32:bl_ntid}, %ntid.x;")
            lines.extend([
                "setp.lt.u32 ${pred:bl_w0}, ${b32:bl_tid}, 32;",
                f"@${{pred:bl_w0}} bar.sync {bar_id}, {count};",
                f"@!${{pred:bl_w0}} bar.arrive {bar_id}, {count};",
            ])
            template = "\n".join(lines) + "\n"
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
            di, dj = wgmma_layout_0_static_offsets(reg, n_dim)
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

    def lower_stmatrix_store(
        self, op: StMatrixStoreOp,
    ) -> tuple[list[Op], list[object]]:
        """Print an already-selected grouped store; all address policy is IR."""
        dst = self.value_map[op.dst]
        src = self.value_map[op.src]
        dst_type = op.dst.type
        assert isinstance(dst_type, SmemTensorType)
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
            and.b32 ${{b32:st_row}}, ${{b32:st_lane}}, 15;
            mad.lo.u32 ${{b32:st_row}}, ${{b32:st_warp}}, 16, ${{b32:st_row}};
            shr.u32 ${{b32:st_col}}, ${{b32:st_lane}}, 4;
            shl.b32 ${{b32:st_col}}, ${{b32:st_col}}, 3;
            mad.lo.u32 ${{b32:st_off}}, ${{b32:st_row}}, {dst_type.strides[0]}, ${{b32:st_col}};
            shl.b32 ${{b32:st_off}}, ${{b32:st_off}}, 1;
            add.u32 ${{b32:base}}, ${{b32:operand.dst.ptr}}, ${{b32:st_off}};
        """)
        for first, offset in zip(op.register_starts, op.offsets):
            regs = ", ".join(
                f"${{b32:operand.src.r{i}}}" for i in range(first, first + 4)
            )
            template += (
                f"{predicate}stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
                f"[${{b32:base}}+{offset}], {{{regs}}};\n"
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
        tensor_rank = len(idx_values)
        if desc_type.physical_layout is not None:
            assert tensor_rank == desc_type.gmem_type.rank + 1
        else:
            assert tensor_rank == desc_type.gmem_type.rank
        indices = ", ".join(
            f"${{b32:operand.idx{i}.val}}"
            for i in reversed(range(len(idx_values)))
        )

        cache_hint = op.l2_cache_hint
        cache_suffix = ".L2::cache_hint" if cache_hint is not None else ""
        cache_operand = ""
        if cache_hint is not None:
            operands.append(self.cache_policy_values[cache_hint])
            operand_names.append("cache_policy")
            cache_operand = ", ${b64:operand.cache_policy.ptr}"

        guard = "\n".join(self._thread_zero_guard("st", "st_done"))
        template = (guard + "\n" if guard else "") + textwrap.dedent(f"""\
            fence.proxy.async.shared::cta;
            cp.async.bulk.tensor.{tensor_rank}d.global.shared::cta.bulk_group{cache_suffix} [${{b64:operand.dst.ptr}}, {{{indices}}}], [${{b32:operand.src.ptr}}]{cache_operand};
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
            indices = ", ".join(
                f"${{b32:operand.{idx_names[j]}.val}}"
                for j in reversed(range(len(idx_values)))
            )
            cache_hint = tma.l2_cache_hint
            if cache_hint is not None:
                policy_name = f"cache_policy{i}"
                operands.append(self.cache_policy_values[cache_hint])
                operand_names.append(policy_name)
            cache_suffix = ".L2::cache_hint" if cache_hint is not None else ""
            cache_operand = (
                f", ${{b64:operand.{policy_name}.ptr}}"
                if cache_hint is not None else ""
            )
            load_lines.append(
                f"cp.async.bulk.tensor.{len(idx_values)}d.shared::cluster.global."
                f"mbarrier::complete_tx::bytes{cache_suffix} "
                f"[${{b32:operand.{dn}.ptr}}], "
                f"[${{b64:operand.{sn}.ptr}}, {{{indices}}}], "
                f"[${{b32:operand.lock.val}}]{cache_operand};"
            )

        # Plain-string lines (literal ${...} placeholders); only load_lines and
        # the byte count are interpolated, so we avoid f-string/dedent brace and
        # multi-line-indent pitfalls by joining a line list.
        #
        # The buffer-reuse WAR (t0's reload vs prior readers of the smem tile) is
        # handled by insert_syncs, which emits a barrier before this async.
        lines = [
            *self._thread_zero_guard("load", "after_load"),
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
    return LowerToPtxTemplate().run(module)
