"""Blackwell split-K, adapted from the library's DSMEM split-K prototype.

The common tensor-core scheduling passes compute disjoint K intervals.
Partials stay in FP32 until a single final output conversion. Both backends
permit paired CTA geometry. The cluster-local alternative uses shared
partials and a rendezvous without a second kernel.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from ..pipelines.ring_schedule import PipelineSchedule

from ..base.ir import find_ops_of_type

from ..gpu_dialect import (
    AllocOp,
    BlockIdOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    F32Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    MatmulOp,
    MemoryEffect,
    PTXTemplateOp,
    RmemTensorType,
    SmemTensorType,
    TileOp,
    gpu_dialect,
)
from ..passes.schedule_async import ScheduleAsync
from ..pipelines.canonicalize import Canonicalize
from ..pipelines.tile_matmul_pipeline import schedule_tiled_matmul


class BlackwellSplitKLauncher:
    """A graph-capturable GEMM with optional caller-owned FP32 workspace.

    A separate workspace is required for concurrent calls on different
    streams. Allocate ``workspace_shape`` once, then call
    ``launch(a, bt, out, workspace)``; both launches are included in capture.
    """

    def __init__(
        self,
        partial_kernel,
        reduction_kernel,
        *,
        m,
        n,
        k,
        dtype,
        transpose_output=False,
        transpose_reduction=False,
    ):
        from ..ptx_runtime import make_ptx_launcher

        self.partial_kernel = partial_kernel
        self.reduction_kernel = reduction_kernel
        self.workspace_shape = (
            tuple(reduction_kernel.signature[0].sizes)
            if reduction_kernel is not None
            else (0,)
        )
        self.transpose_output = transpose_output
        self.transpose_reduction = transpose_reduction
        self.shape = (m, n, k)
        self.dtype = dtype
        self.partial_launch = make_ptx_launcher(partial_kernel)
        self.reduction_launch = (
            make_ptx_launcher(reduction_kernel)
            if reduction_kernel is not None
            else None
        )

    def __call__(self, a, bt, out, workspace=None):
        import torch

        m, n, k = self.shape
        expected_dtype = torch.float16 if self.dtype == "f16" else torch.bfloat16
        tensors = (
            (a, (m, k), expected_dtype),
            (bt, (n, k), expected_dtype),
            (out, (m, n), expected_dtype),
        )
        if self.reduction_launch is not None:
            if workspace is None:
                raise ValueError("split-K requires an FP32 workspace")
            tensors += ((workspace, self.workspace_shape, torch.float32),)
        if any(
            tuple(t.shape) != shape
            or t.dtype != dtype
            or not t.is_contiguous()
            or not t.is_cuda
            or t.device != a.device
            for t, shape, dtype in tensors
        ):
            raise ValueError(
                "split-K operands, output and FP32 workspace must match the compiled shapes, dtypes and CUDA device and be contiguous"
            )
        if self.reduction_launch is None:
            if self.transpose_output:
                self.partial_launch(bt, a, out.T)
            else:
                self.partial_launch(a, bt, out)
            return
        if self.transpose_output:
            self.partial_launch(
                bt, a, workspace if self.transpose_reduction else workspace.T
            )
        else:
            self.partial_launch(a, bt, workspace)
        self.reduction_launch(workspace, out)


def make_blackwell_split_k_launcher(*, m, n, k, dtype="f16", **schedule):
    """Compile a measured Blackwell GEMM; split-K uses caller-owned workspace."""
    from ..ptx_codegen import make_ptx_kernel

    configuration = select_blackwell_split_k_schedule(m, n, k, dtype) or {}
    if "split_k" in schedule and schedule["split_k"] != configuration.get("split_k"):
        configuration = {}
    configuration.update(schedule)
    schedule = configuration
    partial, reduction = blackwell_long_k_gemm_modules(
        m=m, n=n, k=k, dtype=dtype, **schedule
    )
    return BlackwellSplitKLauncher(
        make_ptx_kernel(partial),
        make_ptx_kernel(reduction) if reduction is not None else None,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        transpose_output=schedule.get("transpose_output", False),
        transpose_reduction=schedule.get("transpose_reduction", False),
    )


def blackwell_long_k_gemm_modules(
    *,
    m,
    n,
    k,
    dtype="f16",
    split_k=3,
    native_schedule=None,
    transpose_output=False,
    **schedule,
):
    """Build split-K or direct output when one K partition is selected.

    A split factor of one returns ``(gemm, None)`` and needs no workspace.
    This lets measured schedules avoid partial traffic when output tiles
    already provide sufficient parallelism.
    """
    if type(split_k) is not int or not 1 <= split_k <= 256:
        raise ValueError("split_k must be an integer between one and 256")
    if split_k != 1:
        if native_schedule is not None:
            raise ValueError("native_schedule only applies to a split factor of one")
        return blackwell_split_k_workspace_modules(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            split_k=split_k,
            transpose_output=transpose_output,
            **schedule,
        )
    if native_schedule is None:
        raise ValueError("a split factor of one requires native_schedule")
    from .blackwell_tcgen05 import blackwell_tcgen05_gemm_module

    if schedule:
        raise ValueError("direct output options must be inside native_schedule")
    configuration = dict(native_schedule)
    from .common_warp_schedule import validate_common_warp_schedule

    validate_common_warp_schedule(configuration)
    configuration.setdefault("allow_partial_tiles", True)
    configuration.setdefault("hoist_mma_descriptors", True)
    configuration["output_column_major"] = transpose_output
    return blackwell_tcgen05_gemm_module(
        m=n if transpose_output else m,
        n=m if transpose_output else n,
        k=k,
        dtype=dtype,
        **configuration,
    ), None


def select_blackwell_split_k_schedule(m, n, k, dtype="f16"):
    """Return the measured exact-shape schedule, or None for an unmeasured shape."""
    path = Path(__file__).with_name("blackwell_split_k_schedules.json")
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if document["target"] != "sm_100a" or document["version"] != 1:
        raise ValueError("expected a Blackwell split-K schedule table")
    for row in document["shapes"]:
        if (row["m"], row["n"], row["k"], row["dtype"]) == (m, n, k, dtype):
            from .common_warp_schedule import validate_common_warp_schedule

            validate_common_warp_schedule(row["configuration"])
            return dict(row["configuration"])
    return None


def blackwell_split_k_gemm_module(
    *,
    m: int,
    n: int,
    k: int,
    split_k: int = 4,
    dtype: str = "f16",
    output_m_tile: int = 128,
    output_n_tile: int = 128,
    operand_k_tile: int = 64,
    operand_stages: int = 4,
    cyclic_operand_pipeline: bool | int = True,
    pipeline: PipelineSchedule | dict | None = None,
    hoist_mma_descriptors: bool = True,
    reuse_operand_storage: bool = True,
    tmem_writeback_warps: int = 4,
    tmem_writeback_register_budget: int = 64,
    partial_padding: int = 4,
    cta_group: int = 1,
    allow_partial_m: bool = False,
    bias: bool = False,
    resident_partition: bool = False,
    cluster_bank_pipeline: bool = False,
    rolled_operand_pipeline: bool = False,
    prefetch_tma_descriptors: bool = False,
):
    """Build ``C = A @ B.T`` with FP32 cluster-local K reduction.

    K intervals are rounded up to whole operand banks; the last interval
    uses the original input tensor's TMA bounds to zero its tail. The output
    may have partial N tiles and, with allow_partial_m, partial M tiles.
    Optional vector bias is added after FP32 reduction and before conversion.
    resident_partition requires one full K bank and one stage. Stores are
    predicated on the original bounds.
    With paired CTAs, adjacent ranks share one K partition and own distinct
    M halves. Reduction reads the same M half from each pair.
    cluster_bank_pipeline releases each operand bank to its own pair; the
    final FP32 reduction still rendezvouses across all K partitions.
    """
    if pipeline is not None:
        plan = PipelineSchedule.parse(pipeline, default_pending=1)
        if plan.cross_jobs or plan.reduction.peel_front or plan.reduction.peel_back or plan.reduction.unroll != 1:
            raise ValueError("split-K requires a per-reduction operand pipeline without loop specialization")
        cyclic_operand_pipeline, rolled_operand_pipeline = plan.mma_pending, plan.rolled
    if any(
        type(flag) is not bool
        for flag in (
            allow_partial_m,
            bias,
            resident_partition,
            cluster_bank_pipeline,
            rolled_operand_pipeline,
            prefetch_tma_descriptors,
        )
    ):
        raise ValueError("split-K feature flags must be boolean")
    if type(split_k) is not int or split_k not in (2, 3, 4, 6, 8, 12, 16):
        raise ValueError("split_k must be 2, 3, 4, 6, 8, 12 or 16 K partitions")
    if cta_group not in (1, 2) or cta_group * split_k > 16:
        raise ValueError("CTA groups must be one/two and fit the 16-CTA cluster")
    if cluster_bank_pipeline and cta_group != 2:
        raise ValueError("cluster bank pipelining requires paired CTAs")
    if dtype not in ("f16", "bf16"):
        raise ValueError("dtype must be f16 or bf16")
    if (
        output_m_tile not in (64, 128)
        or m <= 0
        or (m % (cta_group * output_m_tile) and not allow_partial_m)
    ):
        raise ValueError("M must divide the M64/M128 output tile")
    if n <= 0 or n % 8 or output_n_tile not in (32, 64, 96, 128, 192, 256):
        raise ValueError(
            "N must be positive and aligned to eight; N tile is 32/64/96/128/192/256"
        )
    allowed_k_tiles = (64, 128, 256, 384, 512, 768) if resident_partition else (64, 128)
    if k <= 0 or k % 8 or operand_k_tile not in allowed_k_tiles:
        raise ValueError("K must be positive and aligned to eight; K tile is 64/128")
    partial_k = math.ceil(k / (split_k * operand_k_tile)) * operand_k_tile
    if resident_partition and (
        operand_stages != 1 or cyclic_operand_pipeline or partial_k != operand_k_tile
    ):
        raise ValueError(
            "resident partitions require one complete K bank, one stage and no cyclic pipeline"
        )
    if (split_k - 1) * partial_k >= k:
        raise ValueError("every K partition must contain input elements")
    if not 1 <= operand_stages <= partial_k // operand_k_tile:
        raise ValueError("operand stages must fit each K partition")
    if int(cyclic_operand_pipeline) >= operand_stages:
        raise ValueError("cyclic queue depth must be smaller than operand stages")
    if partial_padding not in (0, 4) or tmem_writeback_warps not in (4, 8):
        raise ValueError(
            "partial padding must be zero/four and writeback warps four/eight"
        )
    partial_stride = output_n_tile + partial_padding
    partial_bytes = 4 * output_m_tile * partial_stride
    operand_bytes = (
        2
        * (output_m_tile + output_n_tile // cta_group)
        * operand_k_tile
        * operand_stages
    )
    shared_bytes = (
        max(partial_bytes, operand_bytes)
        if reuse_operand_storage
        else partial_bytes + operand_bytes
    )
    if shared_bytes > 226 * 1024:
        raise ValueError("partials and operand banks exceed shared-memory capacity")
    tiles_m = (
        math.ceil(m / (cta_group * output_m_tile))
        if allow_partial_m
        else m // (cta_group * output_m_tile)
    )
    tiles_n = math.ceil(n / output_n_tile)
    name = f"blackwell_split_k_m{m}_n{n}_k{k}_{dtype}_s{split_k}"
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>{f", %bias: gmem<{n}x{dtype}>" if bias else ""})
          grid_dim=[{tiles_m * tiles_n * split_k * cta_group}, 1, 1] block_dim=[{tmem_writeback_warps * 32}, 1, 1] cluster_dim=[{split_k * cta_group}, 1, 1] {{}}
        }}
    """)
    kernel = module.body.ops[0]
    kernel.attrs["hoist_mma_descriptors"] = hoist_mma_descriptors
    kernel.attrs["integer_index_division"] = True
    kernel.attrs["prefetch_tma_descriptors"] = prefetch_tma_descriptors
    a, b, c, *bias_arguments = kernel.body.arguments
    bx = BlockIdOp.build(CollectiveAxis.X)
    splits = IndexConstantOp.build(split_k)
    group = IndexConstantOp.build(cta_group)
    pair = IndexDivOp.build(bx.value, group.result())
    half = IndexModOp.build(bx.value, group.result())
    mt = IndexConstantOp.build(tiles_m)
    tile = IndexDivOp.build(pair.result(), splits.result())
    part = IndexModOp.build(pair.result(), splits.result())
    group_m = IndexModOp.build(tile.result(), mt.result())
    mi_base = IndexMulOp.build(group_m.result(), group.result())
    mi = IndexAddOp.build(mi_base.result(), half.result())
    ni = IndexDivOp.build(tile.result(), mt.result())
    av = TileOp.build(
        a,
        [mi.result(), part.result()],
        GmemTensorType.build(
            [output_m_tile, partial_k], a.type.dtype, strides=a.type.strides
        ),
        maybe_partial=m % output_m_tile != 0 or k % (split_k * operand_k_tile) != 0,
    )
    bv = TileOp.build(
        b,
        [ni.result(), part.result()],
        GmemTensorType.build(
            [output_n_tile, partial_k], b.type.dtype, strides=b.type.strides
        ),
        maybe_partial=n % output_n_tile != 0 or k % (split_k * operand_k_tile) != 0,
    )
    cv = TileOp.build(
        c,
        [mi.result(), ni.result()],
        GmemTensorType.build(
            [output_m_tile, output_n_tile], c.type.dtype, strides=c.type.strides
        ),
        maybe_partial=m % output_m_tile != 0 or n % output_n_tile != 0,
    )
    partial = AllocOp.build_with_fresh_id(
        SmemTensorType.build([output_m_tile, partial_stride], F32Type())
    )
    index_zero = IndexConstantOp.build(0)
    partial_view = TileOp.build(
        partial.result(),
        [index_zero.result(), index_zero.result()],
        SmemTensorType.build(
            [output_m_tile, output_n_tile], F32Type(), strides=[partial_stride, 1]
        ),
    )
    zero = ConstantOp.build(0.0, RmemTensorType.build([], F32Type()))
    mm = MatmulOp.build(
        av.result(),
        bv.result(),
        partial_view.result(),
        init=zero.result(),
        collective={axis: CollectiveLevel.BLOCK for axis in CollectiveAxis},
        attrs={"label": "mm"},
    )
    reduction = PTXTemplateOp.build(
        operands=[
            partial_view.result(),
            cv.result(),
            ni.result(),
            *([mi.result()] if allow_partial_m else []),
            *bias_arguments,
        ],
        result_types=[],
        template=_cluster_reduction(
            split_k,
            output_m_tile,
            output_n_tile,
            n,
            dtype,
            tmem_writeback_warps * 32,
            partial_stride,
            cta_group,
            m=m if allow_partial_m else None,
            bias=bias,
        ),
        operand_names=[
            "partial",
            "dst",
            "tile_n",
            *(["tile_m"] if allow_partial_m else []),
            *(["bias"] if bias else []),
        ],
        result_names=[],
        description="cluster-local FP32 split-K reduction",
    )
    kernel.body.ops.extend(
        [
            bx,
            splits,
            group,
            pair,
            half,
            mt,
            tile,
            part,
            group_m,
            mi_base,
            mi,
            ni,
            av,
            bv,
            cv,
            partial,
            index_zero,
            partial_view,
            zero,
            mm,
            reduction,
        ]
    )
    module = schedule_tiled_matmul(
        label="mm",
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        operand_k_tile=operand_k_tile,
        operand_stages=operand_stages,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        rolled_operand_pipeline=rolled_operand_pipeline,
        accumulator="tmem",
        cta_group=cta_group,
        output_store="global",
        output_tma_inner_size=None,
        tmem_writeback_register_budget=tmem_writeback_register_budget,
        tmem_writeback_warps=tmem_writeback_warps,
    ).run(module)
    module = ScheduleAsync().run(module)
    if cta_group == 2:
        # Promotion shards B with the partner coordinate. A K-split cluster
        # contains several pairs, so physical ranks 2+ must still use 0/1.
        from ..cluster_layout import ClusterLayout
        from ..passes.map_cluster_tiles import MapClusterTiles

        size = cta_group * split_k
        module = MapClusterTiles(
            ClusterLayout(size, 1, (size, 1, 1), cta_group), (0, 1)
        ).run(module)
        if cluster_bank_pipeline:
            from ..passes.pipeline_cluster_banks import PipelineClusterBanks

            module = Canonicalize().run(module)
            module.body.ops[0].attrs["correlate_uniform_guards"] = True
            # Verify the modeled operand pipeline before attaching the opaque
            # finalizer. It reads only completed FP32 partials, after the TMEM
            # drain; its final cluster rendezvous protects those reads at exit.
            module.body.ops[0].body.ops.remove(reduction)
            module = PipelineClusterBanks().run(module)
            module.body.ops[0].body.ops.append(reduction)
    if reuse_operand_storage:
        # All K MMAs retire before the TMEM drain writes the partial tile.
        # The drain never reads operand SMEM; no operand refill follows it.
        # The final cluster barrier prevents exit while peers read the tile.
        for alloc in find_ops_of_type(module, AllocOp):
            if isinstance(alloc.result().type, SmemTensorType):
                label = alloc.attrs.get("label", "")
                if label.startswith(("mm_lhs_smem", "mm_rhs_smem")):
                    alloc.attrs["reuse_group"] = "split_k_operands"
                elif alloc.result().type.dtype == F32Type():
                    alloc.attrs["reuse_group"] = "split_k_partials"
    return Canonicalize().run(module)


def _cluster_reduction(
    splits,
    m_tile,
    n_tile,
    n,
    dtype,
    block_threads=128,
    partial_stride=None,
    cta_group=1,
    *,
    m=None,
    bias=False,
):
    partial_stride = n_tile if partial_stride is None else partial_stride
    threads = block_threads * splits
    template = """
        bar.sync 0;
        barrier.cluster.arrive.release;
        barrier.cluster.wait.acquire;
        mov.u32 ${b32:rank}, %cluster_ctarank;
        rem.u32 ${b32:half}, ${b32:rank}, CTA_GROUP;
        div.u32 ${b32:part}, ${b32:rank}, CTA_GROUP;
        mov.u32 ${b32:tid}, %tid.x;
        mad.lo.u32 ${b32:flat}, ${b32:part}, BLOCK_THREADS, ${b32:tid};
        mul.lo.u32 ${b32:flat}, ${b32:flat}, 4;
    """
    template = template.replace("BLOCK_THREADS", str(block_threads))
    template = template.replace("CTA_GROUP", str(cta_group))
    for peer in range(splits):
        template += f"add.u32 ${{b32:peer_rank}}, ${{b32:half}}, {peer * cta_group};\nmapa.shared::cluster.u32 ${{b32:peer{peer}}}, ${{b32:operand.partial.ptr}}, ${{b32:peer_rank}};\n"
    for offset in range(0, m_tile * n_tile, threads * 4):
        template += f"""
            add.u32 ${{b32:element}}, ${{b32:flat}}, {offset};
            setp.lt.u32 ${{pred:valid}}, ${{b32:element}}, {m_tile * n_tile};
            @!${{pred:valid}} bra ${{label:skip{offset}}};
            div.u32 ${{b32:row}}, ${{b32:element}}, {n_tile};
            rem.u32 ${{b32:col}}, ${{b32:element}}, {n_tile};
            mad.lo.u32 ${{b32:smem_element}}, ${{b32:row}}, {partial_stride}, ${{b32:col}};
            shl.b32 ${{b32:smem_offset}}, ${{b32:smem_element}}, 2;
        """
        for peer in range(splits):
            template += f"add.u32 ${{b32:address}}, ${{b32:peer{peer}}}, ${{b32:smem_offset}};\n"
            template += "ld.shared::cluster.v4.b32 {${f32:v0}, ${f32:v1}, ${f32:v2}, ${f32:v3}}, [${b32:address}];\n"
            for i in range(4):
                template += (
                    f"mov.f32 ${{f32:sum{i}}}, ${{f32:v{i}}};\n"
                    if peer == 0
                    else f"add.f32 ${{f32:sum{i}}}, ${{f32:sum{i}}}, ${{f32:v{i}}};\n"
                )
        row_guard = (
            ""
            if m is None
            else f"""
            mad.lo.u32 ${{b32:global_row}}, ${{b32:operand.tile_m.val}}, {m_tile}, ${{b32:row}};
            setp.lt.u32 ${{pred:row_valid}}, ${{b32:global_row}}, {m};
            and.pred ${{pred:valid}}, ${{pred:valid}}, ${{pred:row_valid}};
        """
        )
        bias_add = ""
        if bias:
            bias_add = "mul.wide.u32 ${b64:bias_offset}, ${b32:global_col}, 2;\nadd.u64 ${b64:bias_address}, ${b64:operand.bias.ptr}, ${b64:bias_offset};\n"
            for i in range(4):
                bias_add += f"@${{pred:valid}} ld.global.b16 ${{b16:bias{i}}}, [${{b64:bias_address}}+{i * 2}];\n@${{pred:valid}} cvt.f32.{dtype} ${{f32:bias_float{i}}}, ${{b16:bias{i}}};\n@${{pred:valid}} add.f32 ${{f32:sum{i}}}, ${{f32:sum{i}}}, ${{f32:bias_float{i}}};\n"
        template += f"""
            mad.lo.u32 ${{b32:global_col}}, ${{b32:operand.tile_n.val}}, {n_tile}, ${{b32:col}};
            setp.lt.u32 ${{pred:valid}}, ${{b32:global_col}}, {n};
            {row_guard}
            mad.lo.u32 ${{b32:dst_offset}}, ${{b32:row}}, {n}, ${{b32:col}};
            mul.wide.u32 ${{b64:dst_offset64}}, ${{b32:dst_offset}}, 2;
            add.u64 ${{b64:dst_address}}, ${{b64:operand.dst.ptr}}, ${{b64:dst_offset64}};
            {bias_add}
            cvt.rn.{dtype}x2.f32 ${{b32:packed0}}, ${{f32:sum1}}, ${{f32:sum0}};
            cvt.rn.{dtype}x2.f32 ${{b32:packed1}}, ${{f32:sum3}}, ${{f32:sum2}};
            @${{pred:valid}} st.global.v2.b32 [${{b64:dst_address}}], {{${{b32:packed0}}, ${{b32:packed1}}}};
            ${{label:skip{offset}}}:
        """
    return template + "barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n"


def blackwell_split_k_workspace_modules(
    *,
    m: int,
    n: int,
    k: int,
    split_k: int = 3,
    dtype: str = "f16",
    output_m_tile: int = 128,
    output_n_tile: int = 128,
    operand_k_tile: int = 64,
    operand_stages: int = 5,
    cyclic_operand_pipeline: bool | int = True,
    pipeline: PipelineSchedule | dict | None = None,
    cta_group: int = 2,
    hoist_mma_descriptors: bool = True,
    paired_phase_relay: bool = True,
    tmem_writeback_warps: int = 4,
    tmem_writeback_register_budget: int = 64,
    reduction_elements_per_thread: int = 4,
    reduction_threads: int = 256,
    partial_store: str = "global",
    output_layout: str | None = None,
    prefetch_tma_descriptors: bool = False,
    transpose_output: bool = False,
    transpose_reduction: bool = False,
    rolled_operand_pipeline: bool = False,
    operand_issuer_warp: int = 0,
    bias: bool = False,
):
    """Split-K GEMM and vector reduction modules, with FP32 global partials.

    Workspace shape is ``[split_k * padded_m, padded_n]``; callers allocate it once
    and capture both launches. This permits non-power-of-two K partitions
    and paired MMA geometries independently of the reduction topology.
    """
    if pipeline is not None:
        plan = PipelineSchedule.parse(pipeline, default_pending=1)
        if plan.cross_jobs or plan.reduction.peel_front or plan.reduction.peel_back or plan.reduction.unroll != 1:
            raise ValueError("split-K requires a per-reduction operand pipeline without loop specialization")
        cyclic_operand_pipeline, rolled_operand_pipeline = plan.mma_pending, plan.rolled
    if type(split_k) is not int or not 2 <= split_k <= 256:
        raise ValueError("split_k must be an integer between two and 256")
    original_m, original_n = m, n
    if type(transpose_output) is not bool:
        raise ValueError("transpose_output must be boolean")
    if type(transpose_reduction) is not bool or (
        transpose_reduction and not transpose_output
    ):
        raise ValueError("transpose_reduction requires transposed computation")
    if transpose_output:
        if bias:
            raise ValueError("fused split-K bias requires row-major computation")
        m, n = n, m
        if partial_store != "global":
            raise ValueError("transposed partials require global stores")
        if not transpose_reduction:
            output_layout = (
                "tcgen05_16x256b" if output_layout is None else output_layout
            )
        elif original_n % 32 or original_m % 32:
            raise ValueError("tiled transpose reduction needs dimensions aligned to 32")
    if cta_group not in (1, 2) or output_m_tile not in (64, 128):
        raise ValueError("CTA group must be one/two and M tile must be 64/128")
    if type(cyclic_operand_pipeline) not in (int, bool) or cyclic_operand_pipeline < 0:
        raise ValueError("cyclic queue depth must be nonnegative")
    if any(
        type(value) is not bool
        for value in (
            paired_phase_relay,
            hoist_mma_descriptors,
            prefetch_tma_descriptors,
            rolled_operand_pipeline,
        )
    ):
        raise ValueError(
            "phase relay, descriptor hoisting and prefetch must be boolean"
        )
    if partial_store not in ("global", "stmatrix"):
        raise ValueError("partial_store must be global or stmatrix")
    if tmem_writeback_warps not in (4, 8):
        raise ValueError("writeback needs four or eight warps")
    if type(operand_issuer_warp) is not int or operand_issuer_warp != 0:
        raise ValueError("long-K requires TMA and MMA to share issuer warp zero")
    if rolled_operand_pipeline and operand_issuer_warp:
        raise ValueError("rolled operands require the shared issuer")
    if (
        reduction_elements_per_thread not in (2, 4, 8, 16, 32)
        or original_n % reduction_elements_per_thread
        or reduction_threads not in (32, 64, 128, 256, 512)
    ):
        raise ValueError(
            "reduction vector must divide N and use 32/64/128/256/512 threads"
        )
    if min(m, n, k) <= 0 or n % 2 or k % 8:
        raise ValueError("dimensions must be positive and match M, N2 and K8 alignment")
    if dtype not in ("f16", "bf16") or output_n_tile not in (32, 64, 96, 128, 192, 256):
        raise ValueError(
            "dtype must be f16/bf16 and N tile must be 32/64/96/128/192/256"
        )
    if operand_k_tile not in (64, 128, 256):
        raise ValueError("K tile must be 64/128/256")
    partial_k = math.ceil(k / (split_k * operand_k_tile)) * operand_k_tile
    if (split_k - 1) * partial_k >= k:
        raise ValueError("every K partition must contain input elements")
    if (
        not 1 <= operand_stages <= partial_k // operand_k_tile
        or int(cyclic_operand_pipeline) >= operand_stages
    ):
        raise ValueError("stages and queue depth must fit the K partition")
    operand_bytes = (
        2
        * (output_m_tile + output_n_tile // cta_group)
        * operand_k_tile
        * operand_stages
    )
    if operand_bytes > 226 * 1024:
        raise ValueError("operand banks exceed shared-memory capacity")
    padded_n = math.ceil(n / output_n_tile) * output_n_tile
    padded_m = math.ceil(m / (cta_group * output_m_tile)) * cta_group * output_m_tile
    tiles_m, tiles_n = padded_m // output_m_tile, padded_n // output_n_tile
    tiles = tiles_m * tiles_n
    workspace_type = (
        f"{split_k * padded_m}x{padded_n}xf32 strides=[1, {split_k * padded_m}]"
        if transpose_output and not transpose_reduction
        else f"{split_k * padded_m}x{padded_n}xf32"
    )
    name = f"blackwell_split_k_partial_m{m}_n{n}_k{k}_{dtype}_p{split_k}"
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %w: gmem<{workspace_type}>)
          grid_dim=[{tiles * split_k}, 1, 1] block_dim=[{tmem_writeback_warps * 32}, 1, 1] cluster_dim=[{cta_group}, 1, 1] {{}}
        }}
    """)
    kernel = module.body.ops[0]
    kernel.attrs["hoist_mma_descriptors"] = hoist_mma_descriptors
    kernel.attrs["integer_index_division"] = True
    kernel.attrs["prefetch_tma_descriptors"] = prefetch_tma_descriptors
    a, b, w = kernel.body.arguments
    bx = BlockIdOp.build(CollectiveAxis.X)
    tile_count, mt = IndexConstantOp.build(tiles), IndexConstantOp.build(tiles_m)
    part = IndexDivOp.build(bx.value, tile_count.result())
    tile = IndexModOp.build(bx.value, tile_count.result())
    mi = IndexModOp.build(tile.result(), mt.result())
    ni = IndexDivOp.build(tile.result(), mt.result())
    wbase = IndexMulOp.build(part.result(), mt.result())
    wi = IndexAddOp.build(wbase.result(), mi.result())
    av = TileOp.build(
        a,
        [mi.result(), part.result()],
        GmemTensorType.build(
            [output_m_tile, partial_k], a.type.dtype, strides=a.type.strides
        ),
        maybe_partial=m != padded_m or k % (split_k * operand_k_tile) != 0,
    )
    bv = TileOp.build(
        b,
        [ni.result(), part.result()],
        GmemTensorType.build(
            [output_n_tile, partial_k], b.type.dtype, strides=b.type.strides
        ),
        maybe_partial=n % output_n_tile != 0 or k % (split_k * operand_k_tile) != 0,
    )
    wv = TileOp.build(
        w,
        [wi.result(), ni.result()],
        GmemTensorType.build(
            [output_m_tile, output_n_tile], F32Type(), strides=w.type.strides
        ),
    )
    zero = ConstantOp.build(0.0, RmemTensorType.build([], F32Type()))
    mm = MatmulOp.build(
        av.result(),
        bv.result(),
        wv.result(),
        init=zero.result(),
        collective={axis: CollectiveLevel.BLOCK for axis in CollectiveAxis},
        attrs={
            "label": "mm",
            "dst_tiling_scheme": MemoryEffect.fresh_tiling_scheme_id(),
        },
    )
    kernel.body.ops.extend(
        [bx, tile_count, mt, part, tile, mi, ni, wbase, wi, av, bv, wv, zero, mm]
    )
    module = schedule_tiled_matmul(
        label="mm",
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        operand_k_tile=operand_k_tile,
        operand_stages=operand_stages,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        rolled_operand_pipeline=rolled_operand_pipeline,
        operand_move_attrs={
            name: {"issuer_warp": operand_issuer_warp} for name in ("lhs", "rhs")
        },
        accumulator="tmem",
        cta_group=cta_group,
        output_store=partial_store,
        output_tma_inner_size=None,
        output_layout=output_layout,
        tmem_writeback_register_budget=tmem_writeback_register_budget,
        tmem_writeback_warps=tmem_writeback_warps,
    ).run(module)
    module = ScheduleAsync().run(module)
    module = Canonicalize().run(module)
    if cta_group == 2 and paired_phase_relay:
        # Each pair's M/N tile and K partition are aligned; use the native
        # paired phase handoff instead of a cluster-wide gather per refill.
        from ..passes.pipeline_cluster_banks import PipelineClusterBanks
        from ..passes.schedule_warp_waits import ScheduleWarpWaits

        module.body.ops[0].attrs["correlate_uniform_guards"] = True
        module.body.ops[0].attrs["paired_phase_relay"] = True
        module = PipelineClusterBanks().run(module)
        if not module.body.ops[0].attrs.get("cluster_bank_pipeline"):
            module = ScheduleWarpWaits().run(module)
    workspace_shape = (
        f"{padded_n}x{split_k * padded_m}"
        if transpose_output and not transpose_reduction
        else f"{split_k * padded_m}x{padded_n}"
    )
    reduce = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @blackwell_split_k_reduce_m{original_m}_n{original_n}_p{split_k}_{dtype}(%w: gmem<{workspace_shape}xf32>, %c: gmem<{original_m}x{original_n}x{dtype}>{f", %bias: gmem<{original_n}x{dtype}>" if bias else ""})
          grid_dim=[{math.ceil(original_m * original_n / (reduction_threads * reduction_elements_per_thread))}, 1, 1] block_dim=[{reduction_threads}, 1, 1] {{}}
        }}
    """)
    w, c, *bias_arguments = reduce.body.ops[0].body.arguments
    if transpose_reduction:
        kernel = reduce.body.ops[0]
        kernel.block_dim = [256, 1, 1]
        kernel.grid_dim = [original_m // 32 * (original_n // 32), 1, 1]
        scratch = AllocOp.build_with_fresh_id(SmemTensorType.build([32, 33], F32Type()))
        kernel.body.ops.extend(
            [
                scratch,
                PTXTemplateOp.build(
                    operands=[w, c, scratch.result()],
                    result_types=[],
                    template=_transposed_workspace_reduction(
                        split_k, original_m, original_n, padded_m, padded_n, dtype
                    ),
                    operand_names=["workspace", "dst", "scratch"],
                    result_names=[],
                    description="FP32 split-K tiled transpose reduction",
                ),
            ]
        )
        return module, reduce
    reduce.body.ops[0].body.ops.append(
        PTXTemplateOp.build(
            operands=[w, c, *bias_arguments],
            result_types=[],
            template=_workspace_reduction(
                split_k,
                original_m,
                original_n,
                padded_m if transpose_output else padded_m * padded_n,
                split_k * padded_m if transpose_output else padded_n,
                dtype,
                reduction_elements_per_thread,
                reduction_threads,
                bias=bias,
            ),
            operand_names=["workspace", "dst", *(["bias"] if bias else [])],
            result_names=[],
            description="FP32 split-K workspace reduction",
        )
    )
    return module, reduce


def _transposed_workspace_reduction(splits, m, n, padded_m, padded_n, dtype):
    template = f"""
        mov.u32 ${{b32:block}}, %ctaid.x;
        mov.u32 ${{b32:tid}}, %tid.x;
        div.u32 ${{b32:tile_m}}, ${{b32:block}}, {n // 32};
        rem.u32 ${{b32:tile_n}}, ${{b32:block}}, {n // 32};
        shr.u32 ${{b32:local_row}}, ${{b32:tid}}, 3;
        and.b32 ${{b32:local_col}}, ${{b32:tid}}, 7;
        shl.b32 ${{b32:local_col}}, ${{b32:local_col}}, 2;
        mad.lo.u32 ${{b32:src_row}}, ${{b32:tile_n}}, 32, ${{b32:local_row}};
        mad.lo.u32 ${{b32:src_col}}, ${{b32:tile_m}}, 32, ${{b32:local_col}};
        mad.lo.u32 ${{b32:src_element}}, ${{b32:src_row}}, {padded_n}, ${{b32:src_col}};
        mul.wide.u32 ${{b64:src_offset}}, ${{b32:src_element}}, 4;
        add.u64 ${{b64:src_address}}, ${{b64:operand.workspace.ptr}}, ${{b64:src_offset}};
    """
    for part in range(splits):
        template += "ld.global.v4.b32 {${f32:v0}, ${f32:v1}, ${f32:v2}, ${f32:v3}}, [${b64:src_address}];\n"
        for i in range(4):
            template += (
                f"mov.f32 ${{f32:sum{i}}}, ${{f32:v{i}}};\n"
                if part == 0
                else f"add.f32 ${{f32:sum{i}}}, ${{f32:sum{i}}}, ${{f32:v{i}}};\n"
            )
        if part + 1 < splits:
            template += f"add.u64 ${{b64:src_address}}, ${{b64:src_address}}, {padded_m * padded_n * 4};\n"
    template += """
        mad.lo.u32 ${b32:shared_element}, ${b32:local_row}, 33, ${b32:local_col};
        shl.b32 ${b32:shared_offset}, ${b32:shared_element}, 2;
        add.u32 ${b32:shared_address}, ${b32:operand.scratch.ptr}, ${b32:shared_offset};
    """
    for i in range(4):
        template += (
            f"st.shared.b32 [${{b32:shared_address}}+{i * 4}], ${{f32:sum{i}}};\n"
        )
    template += """
        bar.sync 0;
        mad.lo.u32 ${b32:shared_element}, ${b32:local_col}, 33, ${b32:local_row};
        shl.b32 ${b32:shared_offset}, ${b32:shared_element}, 2;
        add.u32 ${b32:shared_address}, ${b32:operand.scratch.ptr}, ${b32:shared_offset};
    """
    for i in range(4):
        template += (
            f"ld.shared.b32 ${{f32:sum{i}}}, [${{b32:shared_address}}+{i * 33 * 4}];\n"
        )
    return (
        template
        + f"""
        mad.lo.u32 ${{b32:dst_row}}, ${{b32:tile_m}}, 32, ${{b32:local_row}};
        mad.lo.u32 ${{b32:dst_col}}, ${{b32:tile_n}}, 32, ${{b32:local_col}};
        mad.lo.u32 ${{b32:dst_element}}, ${{b32:dst_row}}, {n}, ${{b32:dst_col}};
        mul.wide.u32 ${{b64:dst_offset}}, ${{b32:dst_element}}, 2;
        add.u64 ${{b64:dst_address}}, ${{b64:operand.dst.ptr}}, ${{b64:dst_offset}};
        cvt.rn.{dtype}x2.f32 ${{b32:packed0}}, ${{f32:sum1}}, ${{f32:sum0}};
        cvt.rn.{dtype}x2.f32 ${{b32:packed1}}, ${{f32:sum3}}, ${{f32:sum2}};
        st.global.v2.b32 [${{b64:dst_address}}], {{${{b32:packed0}}, ${{b32:packed1}}}};
    """
    )


def _workspace_reduction(
    splits, m, n, partition_stride, row_stride, dtype, vector, threads, *, bias=False
):
    # Each warp reads adjacent four-element vectors, then advances by a
    # whole CTA's stripe. Wider per-thread work keeps every memory sector
    # fully used without requiring a large live register accumulator.
    width = 2 if vector == 2 else 4
    template = f"""
        mov.u32 ${{b32:block}}, %ctaid.x;
        mov.u32 ${{b32:tid}}, %tid.x;
        mul.lo.u32 ${{b32:base}}, ${{b32:block}}, {threads * vector};
        mad.lo.u32 ${{b32:base}}, ${{b32:tid}}, {width}, ${{b32:base}};
    """
    for stripe in range(vector // width):
        template += f"""
            add.u32 ${{b32:element}}, ${{b32:base}}, {stripe * threads * width};
            setp.lt.u32 ${{pred:valid}}, ${{b32:element}}, {m * n};
            @!${{pred:valid}} bra ${{label:skip{stripe}}};
        """
        if row_stride == n:
            template += "mul.wide.u32 ${b64:workspace_offset}, ${b32:element}, 4;\n"
        else:
            template += f"""
                div.u32 ${{b32:row}}, ${{b32:element}}, {n};
                rem.u32 ${{b32:col}}, ${{b32:element}}, {n};
                mad.lo.u32 ${{b32:workspace_element}}, ${{b32:row}}, {row_stride}, ${{b32:col}};
                mul.wide.u32 ${{b64:workspace_offset}}, ${{b32:workspace_element}}, 4;
            """
        template += "add.u64 ${b64:address}, ${b64:operand.workspace.ptr}, ${b64:workspace_offset};\n"
        for part in range(splits):
            values = ", ".join(f"${{f32:v{i}}}" for i in range(width))
            template += f"ld.global.v{width}.b32 {{{values}}}, [${{b64:address}}];\n"
            for lane in range(width):
                template += (
                    f"mov.f32 ${{f32:sum{lane}}}, ${{f32:v{lane}}};\n"
                    if part == 0
                    else f"add.f32 ${{f32:sum{lane}}}, ${{f32:sum{lane}}}, ${{f32:v{lane}}};\n"
                )
            if part + 1 < splits:
                template += f"add.u64 ${{b64:address}}, ${{b64:address}}, {partition_stride * 4};\n"
        if bias:
            template += f"rem.u32 ${{b32:bias_col}}, ${{b32:element}}, {n};\n"
            template += "mul.wide.u32 ${b64:bias_offset}, ${b32:bias_col}, 2;\nadd.u64 ${b64:bias_address}, ${b64:operand.bias.ptr}, ${b64:bias_offset};\n"
            for lane in range(width):
                template += f"ld.global.b16 ${{b16:bias_value}}, [${{b64:bias_address}}+{2 * lane}];\n"
                template += (
                    f"cvt.f32.{dtype} ${{f32:bias_float}}, ${{b16:bias_value}};\n"
                )
                template += f"add.f32 ${{f32:sum{lane}}}, ${{f32:sum{lane}}}, ${{f32:bias_float}};\n"
        template += f"""
            mul.wide.u32 ${{b64:dst_offset}}, ${{b32:element}}, 2;
            add.u64 ${{b64:dst_address}}, ${{b64:operand.dst.ptr}}, ${{b64:dst_offset}};
            cvt.rn.{dtype}x2.f32 ${{b32:packed0}}, ${{f32:sum1}}, ${{f32:sum0}};
        """
        template += (
            f"""
            cvt.rn.{dtype}x2.f32 ${{b32:packed1}}, ${{f32:sum3}}, ${{f32:sum2}};
            st.global.v2.b32 [${{b64:dst_address}}], {{${{b32:packed0}}, ${{b32:packed1}}}};
        """
            if width == 4
            else "st.global.b32 [${b64:dst_address}], ${b32:packed0};\n"
        )
        template += f"${{label:skip{stripe}}}:\n"
    return template
