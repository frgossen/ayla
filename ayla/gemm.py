"""Configurable GEMM module builders."""
import textwrap

from .base.ir import Block, Value
from .gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BF16Type,
    BlockIdsOp,
    DeallocOp,
    F32Type,
    ForOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexModOp,
    KernelOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmemTensorType,
    gpu_dialect,
)
from .targets import BLACKWELL
from .passes.canonicalize import Canonicalize
from .passes.fuse_async import fuse_async
from .passes.licm import licm
from .passes.lower_to_async import lower_to_async
from .passes.multi_buffer import multi_buffer
from .passes.pipeline_loop import pipeline_loop
from .passes.promote_buffers import promote_buffers
from .passes.schedule_async import schedule_async
from .passes.tile_matmul import tile_matmul
from .passes.unroll_loop import unroll_loop


def blackwell_tcgen05_gemm_module(
    *, m: int = 4096, n: int = 4096, k: int = 4096,
    operand_stages: int = 4, tma_staging: str | None = "direct",
    output_m_tile: int = 128, output_n_tile: int = 128,
    ptx_optimization_level: int | None = 1,
    cluster_multicast_a: bool = False,
    branch_tcgen05_issue: bool = True,
) -> ModuleOp:
    """Build the first native SM100 BF16 GEMM pipeline.

    Each CTA computes one MxN output tile. All four warps cooperatively
    stage both operands and read the tensor-memory accumulator; only the
    elected issue sequence required by ``tcgen05.mma`` is single-threaded.
    With ``cluster_multicast_a``, paired CTAs still execute this same pipeline;
    cluster rank 0 only performs the instruction-mandated issue of their shared
    A transaction, while each CTA independently loads B and computes its tile.
    """
    if output_m_tile not in (64, 128):
        raise ValueError("output_m_tile must be 64 or 128")
    if output_n_tile not in (64, 128, 256):
        raise ValueError("output_n_tile must be 64, 128, or 256")
    if m % output_m_tile or n % output_n_tile or k % 16:
        raise ValueError(
            "tcgen05 GEMM dimensions must divide "
            f"{output_m_tile}x{output_n_tile}x16 tiles"
        )
    if operand_stages < 1 or k % (16 * operand_stages):
        raise ValueError("K must divide the requested tcgen05 operand stages")
    if tma_staging not in (None, "repack", "direct"):
        raise ValueError("tma_staging must be None, 'repack', or 'direct'")
    if tma_staging == "direct" and operand_stages != 4:
        raise ValueError("direct TMA staging currently requires four K=16 stages")
    if output_n_tile != 64 and tma_staging != "direct":
        raise ValueError("wide output tiles currently require direct TMA staging")
    if output_m_tile != 128 and tma_staging != "direct":
        raise ValueError("M=64 output tiles currently require direct TMA staging")
    if cluster_multicast_a and (
        tma_staging != "direct" or output_m_tile != 128 or output_n_tile != 128
    ):
        raise ValueError("cluster A multicast requires direct M=128/N=128 tiles")
    if cluster_multicast_a and n % 256:
        raise ValueError("cluster A multicast requires N divisible by 256")

    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build([m, k], bf16))
    bt = Value(name="bt", type=GmemTensorType.build([n, k], bf16))
    c = Value(name="c", type=GmemTensorType.build([m, n], bf16))
    block_ids = BlockIdsOp.build()
    a_block = TileOp.build(
        a, [block_ids.x, 0],
        GmemTensorType.build([output_m_tile, k], bf16, strides=[k, 1]),
    )
    bt_block = TileOp.build(
        bt, [block_ids.y, 0],
        GmemTensorType.build([output_n_tile, k], bf16, strides=[k, 1]),
    )
    c_block = TileOp.build(
        c, [block_ids.x, block_ids.y],
        GmemTensorType.build(
            [output_m_tile, output_n_tile], bf16, strides=[n, 1],
        ),
    )
    accumulator = AllocOp.build_with_fresh_id(
        TmemTensorType.build([output_m_tile, output_n_tile], F32Type()), init=0.0,
    )
    if tma_staging == "direct":
        lhs_smem = [AllocOp.build_with_fresh_id(SmemTensorType.build(
            [output_m_tile, 64], bf16, swizzle="128B",
            attrs={"tcgen05_matrix": True},
        ))]
        rhs_smem = [AllocOp.build_with_fresh_id(SmemTensorType.build(
            [output_n_tile, 64], bf16, swizzle="128B",
            attrs={"tcgen05_matrix": True},
        ))]
    else:
        lhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [128, 16], bf16, swizzle="32B",
                attrs={"tcgen05_layout": "lhs_128x16"},
            ))
            for _ in range(operand_stages)
        ]
        rhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [64, 16], bf16, swizzle="128B",
                attrs={"tcgen05_layout": "rhs_64x16"},
            ))
            for _ in range(operand_stages)
        ]
    lhs_tma_smem = [
        AllocOp.build_with_fresh_id(SmemTensorType.build([128, 16], bf16))
        for _ in range(operand_stages)
    ] if tma_staging == "repack" else []
    rhs_tma_smem = [
        AllocOp.build_with_fresh_id(SmemTensorType.build([64, 16], bf16))
        for _ in range(operand_stages)
    ] if tma_staging == "repack" else []
    mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
    tma_lock = (
        AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
        if tma_staging == "direct" else None
    )
    if tma_staging == "direct":
        loop = ForOp.build(0, k // 64)
        a_tile = TileOp.build(
            a_block.result(), [0, loop.iv],
            GmemTensorType.build([output_m_tile, 64], bf16, strides=[k, 1]),
        )
        bt_tile = TileOp.build(
            bt_block.result(), [0, loop.iv],
            GmemTensorType.build([output_n_tile, 64], bf16, strides=[k, 1]),
        )
        assert tma_lock is not None
        loop.body.ops.extend([
            a_tile,
            bt_tile,
            AsyncOp.build(tma_lock.result(), body=Block(ops=[
                MoveOp.build(
                    lhs_smem[0].result(), a_tile.result(),
                    attrs={
                        "tma": True,
                        **({"multicast_mask": 0x3, "cluster_leader": True}
                           if cluster_multicast_a else {}),
                    },
                ),
                MoveOp.build(
                    rhs_smem[0].result(), bt_tile.result(),
                    attrs={"tma": True},
                ),
            ])),
            AwaitOp.build(tma_lock.result()),
        ])
        mmas = []
        for stage in range(4):
            lhs_tile = TileOp.build(
                lhs_smem[0].result(), [0, stage],
                SmemTensorType.build(
                    [output_m_tile, 16], bf16,
                    strides=[64, 1], swizzle="128B",
                    attrs={"tcgen05_tma_layout": True},
                ),
            )
            rhs_tile = TileOp.build(
                rhs_smem[0].result(), [0, stage],
                SmemTensorType.build(
                    [output_n_tile, 16], bf16,
                    strides=[64, 1], swizzle="128B",
                    attrs={"tcgen05_tma_layout": True},
                ),
            )
            loop.body.ops.extend([lhs_tile, rhs_tile])
            mmas.append(MatmulOp.build(
                lhs_tile.result(), rhs_tile.result(), accumulator.result(),
            ))
    else:
        loop = ForOp.build(0, k // 16, step=operand_stages)
        mmas = []
        for stage in range(operand_stages):
            if stage == 0:
                k_index = loop.iv
            else:
                increment = IndexConstantOp.build(stage)
                k_index_op = IndexAddOp.build(loop.iv, increment.result())
                loop.body.ops.extend([increment, k_index_op])
                k_index = k_index_op.result()
            a_tile = TileOp.build(
                a_block.result(), [0, k_index],
                GmemTensorType.build([128, 16], bf16, strides=[k, 1]),
            )
            bt_tile = TileOp.build(
                bt_block.result(), [0, k_index],
                GmemTensorType.build([64, 16], bf16, strides=[k, 1]),
            )
            loop.body.ops.extend([a_tile, bt_tile])
            if tma_staging == "repack":
                loop.body.ops.extend([
                    MoveOp.build(
                        lhs_tma_smem[stage].result(), a_tile.result(),
                        attrs={"tma": True},
                    ),
                    MoveOp.build(
                        rhs_tma_smem[stage].result(), bt_tile.result(),
                        attrs={"tma": True},
                    ),
                    MoveOp.build(
                        lhs_smem[stage].result(), lhs_tma_smem[stage].result(),
                    ),
                    MoveOp.build(
                        rhs_smem[stage].result(), rhs_tma_smem[stage].result(),
                    ),
                ])
            else:
                loop.body.ops.extend([
                    MoveOp.build(lhs_smem[stage].result(), a_tile.result()),
                    MoveOp.build(rhs_smem[stage].result(), bt_tile.result()),
                ])
            mmas.append(MatmulOp.build(
                lhs_smem[stage].result(), rhs_smem[stage].result(),
                accumulator.result(),
            ))
    loop.body.ops.extend([
        AsyncOp.build(
            mma_lock.result(),
            body=Block(ops=mmas),
            attrs={"branch_elected_issue": True}
            if branch_tcgen05_issue else None,
        ),
        AwaitOp.build(mma_lock.result()),
    ])
    body = Block(arguments=[a, bt, c], ops=[
        block_ids,
        a_block,
        bt_block,
        c_block,
        accumulator,
        *lhs_smem,
        *rhs_smem,
        *lhs_tma_smem,
        *rhs_tma_smem,
        mma_lock,
        *([tma_lock] if tma_lock is not None else []),
        loop,
        MoveOp.build(c_block.result(), accumulator.result()),
        DeallocOp.build(accumulator.result()),
    ])
    return ModuleOp.build(Block(ops=[KernelOp.build(
        "blackwell_tcgen05_4096",
        body,
        grid_dim=(m // output_m_tile, n // output_n_tile, 1),
        block_dim=(128, 1, 1),
        cluster_dim=(1, 2, 1) if cluster_multicast_a else None,
        ptx_optimization_level=ptx_optimization_level,
    )]), target=BLACKWELL)


def blackwell_tcgen05_cta_group2_gemm_module(
    *, m: int = 4096, n: int = 4096, k: int = 4096,
) -> ModuleOp:
    """Build a paired-CTA SM100 GEMM with a shared software pipeline.

    Each neighboring CTA pair computes a 256x64 output region. Both CTAs run
    the same TMA/MMA/store sequence; the single-thread issue mandated by
    ``tcgen05.mma.cta_group::2`` is elected from the pair's even CTA.
    """
    if m % 256 or n % 64 or k % 64:
        raise ValueError("group-2 tcgen05 GEMM dimensions must divide 256x64x64")

    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build([m, k], bf16))
    bt = Value(name="bt", type=GmemTensorType.build([n, k], bf16))
    c = Value(name="c", type=GmemTensorType.build([m, n], bf16))
    block_ids = BlockIdsOp.build()
    two = IndexConstantOp.build(2)
    peer_rank = IndexModOp.build(block_ids.x, two.result())
    a_block = TileOp.build(
        a, [block_ids.x, 0],
        GmemTensorType.build([128, k], bf16, strides=[k, 1]),
    )
    bt_block = TileOp.build(
        bt, [block_ids.y, 0],
        GmemTensorType.build([64, k], bf16, strides=[k, 1]),
    )
    c_block = TileOp.build(
        c, [block_ids.x, block_ids.y],
        GmemTensorType.build([128, 64], bf16, strides=[n, 1]),
    )
    lhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [128, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    rhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [32, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    accumulator = AllocOp.build_with_fresh_id(TmemTensorType.build(
        [128, 64], F32Type(), attrs={"cta_group": 2},
    ), init=0.0)
    tma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
    mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
    loop = ForOp.build(0, k // 64)
    a_tile = TileOp.build(
        a_block.result(), [0, loop.iv],
        GmemTensorType.build([128, 64], bf16, strides=[k, 1]),
    )
    bt_tile = TileOp.build(
        bt_block.result(), [peer_rank.result(), loop.iv],
        GmemTensorType.build([32, 64], bf16, strides=[k, 1]),
    )
    loop.body.ops.extend([
        a_tile,
        bt_tile,
        AsyncOp.build(tma_lock.result(), body=Block(ops=[
            MoveOp.build(lhs_smem.result(), a_tile.result(), attrs={"tma": True}),
            MoveOp.build(rhs_smem.result(), bt_tile.result(), attrs={"tma": True}),
        ])),
        AwaitOp.build(tma_lock.result()),
    ])
    mmas = []
    for stage in range(4):
        lhs_tile = TileOp.build(
            lhs_smem.result(), [0, stage],
            SmemTensorType.build(
                [128, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        rhs_tile = TileOp.build(
            rhs_smem.result(), [0, stage],
            SmemTensorType.build(
                [32, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        loop.body.ops.extend([lhs_tile, rhs_tile])
        mmas.append(MatmulOp.build(
            lhs_tile.result(), rhs_tile.result(), accumulator.result(),
        ))
    loop.body.ops.extend([
        AsyncOp.build(mma_lock.result(), body=Block(ops=mmas)),
        AwaitOp.build(mma_lock.result()),
    ])
    body = Block(arguments=[a, bt, c], ops=[
        block_ids,
        two,
        peer_rank,
        a_block,
        bt_block,
        c_block,
        accumulator,
        lhs_smem,
        rhs_smem,
        tma_lock,
        mma_lock,
        loop,
        MoveOp.build(c_block.result(), accumulator.result()),
        DeallocOp.build(accumulator.result()),
    ])
    return ModuleOp.build(Block(ops=[KernelOp.build(
        "blackwell_tcgen05_cta_group2_4096",
        body,
        grid_dim=(m // 128, n // 64, 1),
        block_dim=(128, 1, 1),
        cluster_dim=(2, 1, 1),
        ptx_optimization_level=1,
    )]), target=BLACKWELL)


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    ptx_optimization_level: int | None = None,
) -> ModuleOp:
    if output_tiles_per_cta not in (1, 2, 4):
        raise ValueError("output_tiles_per_cta must be 1, 2, or 4")
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (128, 192):
        raise ValueError("output_m_tile must be 128 or 192")
    if output_m_tile == 192 and output_tiles_per_cta != 1:
        raise ValueError("M=192 currently requires one output tile per CTA")
    if output_n_tile not in (256, 384):
        raise ValueError("output_n_tile must be 256 or 384")
    if operand_k_tile not in (64, 128):
        raise ValueError("operand_k_tile must be 64 or 128")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile == 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_store == "scalar":
        output_tma_inner_size = None
    problem_name = (
        str(problem_size)
        if m == n == k == problem_size
        else f"m{m}_n{n}_k{k}"
    )
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if warpgroup_n_tile not in (128, 256):
        name += f"_wg{warpgroup_n_tile}"
    if output_tiles_per_cta != 1:
        name += f"_{output_tiles_per_cta}tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"
    m_blocks = (m + output_m_tile - 1) // output_m_tile
    grid_x = (m_blocks + output_tiles_per_cta - 1) // output_tiles_per_cta
    grid_y = (n + output_n_tile - 1) // output_n_tile
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) grid_dim=[{grid_x}, {grid_y}, 1] attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            matmul(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    if output_tiles_per_cta == 1:
        module = tile_matmul(
            module, tile_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            tile_levels=["block.x", "block.y"], tile_labels=[None, None],
            label_filter="mm",
            allow_partial=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        )
    else:
        module = tile_matmul(
            module, tile_dims=["M", "M", "N"],
            tile_sizes=[grid_x * output_m_tile, output_m_tile, 256],
            tile_levels=["for", "block.x", "block.y"],
            tile_labels=["outm", None, None], label_filter="mm",
            allow_partial=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        )
    wg_dim, wg_size = (
        ("M", 64) if output_n_tile == 256
        else ("N", warpgroup_n_tile)
    )
    module = tile_matmul(
        module, tile_dims=[wg_dim], tile_sizes=[wg_size],
        tile_levels=["wg.x"], tile_labels=[None], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"], swizzle=None,
        layout="wgmma_layout_0", warps=4, tma=True, nc=False,
        physical_layout=(
            {
                "swizzle": "128B",
                "block_width": 64,
                "order": "block_major",
            }
            if output_tma_inner_size is not None else None
        ),
        writeback_slices=output_slices,
        writeback_buffers=output_store_buffers,
        move_attrs=operand_move_attrs,
        target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[operand_k_tile],
        tile_levels=["for"], tile_labels=["k"], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None, tma=True, nc=True,
        physical_layout=(
            {
                "block_width": 64,
                "order": "block_major",
            }
            if operand_k_tile > 64 else None
        ),
        tma_load_slices=(
            {"rhs": output_n_tile // warpgroup_n_tile}
            if output_n_tile == 384 else None
        ),
        move_attrs=operand_move_attrs,
        label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"], label_filter="mm",
    )
    module = multi_buffer(
        module, depth=operand_stages, label_filter="mm_lhs_smem",
    )
    module = multi_buffer(
        module, depth=operand_stages, label_filter="mm_rhs_smem",
    )
    module = licm(module)
    module = unroll_loop(module, unroll_factor=operand_stages, label_filter="k")
    module = Canonicalize().run(module)
    module = lower_to_async(module)
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = Canonicalize().run(module)

    module = pipeline_loop(
        module, label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    )
    module = unroll_loop(
        module, unroll_factor=operand_k_tile // 16,
        label_filter=r"kk(?:_tail\d+)?(?:_iter\d+)?",
    )
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = Canonicalize().run(module)

    if output_tiles_per_cta != 1:
        module = unroll_loop(
            module, unroll_factor=output_tiles_per_cta,
            label_filter="outm",
        )
    module = schedule_async(module)
    return Canonicalize().run(module)
