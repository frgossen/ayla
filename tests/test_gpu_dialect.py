import textwrap

import pytest

from ayla import (
    Block,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    Op,
    PTXTemplateOp,
    Value,
)
from ayla.gpu_dialect import (
    AddrMemTensorType,
    AsyncCommitOp,
    AsyncOp,
    DeallocOp,
    F32Type,
    ForOp,
    IndexType,
    LockType,
    MoveOp,
    PartialSplitBlockIdsOp,
    SplitNBlockIdOp,
    SmemTensorType,
    TmaDescType,
    TmaMoveOp,
    TmemTensorType,
    TileOp,
    gpu_dialect,
)
from ayla.targets import BLACKWELL, HOPPER


def assert_print_and_roundtrip(op, expected):
    text = repr(op)
    assert text == expected
    assert repr(gpu_dialect.parse(text)) == text


def test_matmul_program():
    m, k, n = 64, 128, 32
    a_type = GmemTensorType.build((m, k), F32Type())
    b_type = GmemTensorType.build((k, n), F32Type())
    c_type = GmemTensorType.build((m, n), F32Type())

    a = Value(name="a", type=a_type)
    b = Value(name="b", type=b_type)
    c = Value(name="c", type=c_type)
    matmul = MatmulOp.build(a, b, c, attrs={"library": "cublas"})
    block = Block(arguments=[a, b, c], ops=[matmul])
    program = Op(operands=[], results=[], blocks=[block])

    expected = textwrap.dedent("""\
        generic() {
        ^bb0(%a: gmem<64x128xf32>, %b: gmem<128x32xf32>, %c: gmem<64x32xf32>):
          matmul(%a: gmem<64x128xf32>, %b: gmem<128x32xf32>, %c: gmem<64x32xf32>) attrs={library: "cublas"}
        }""")  # noqa: E501
    assert_print_and_roundtrip(program, expected)


def test_kernel_op():
    n = 1 << 20
    buf_type = GmemTensorType.build((n,), F32Type())

    lhs = Value(name="lhs", type=buf_type)
    rhs = Value(name="rhs", type=buf_type)
    dst = Value(name="dst", type=buf_type)
    template = "add.f32 ${f32:s}, ${f32:a}, ${f32:b};\n"
    body = Block(arguments=[lhs, rhs, dst], ops=[
        PTXTemplateOp.build([lhs, rhs, dst], [buf_type], template=template),
    ])
    kernel = KernelOp.build(
        "cwise_add", body, ptx_optimization_level=1, attrs={"version": 1},
    )
    assert kernel.ptx_optimization_level == 1
    module = ModuleOp.build(
        Block(ops=[kernel]),
        attrs={"target": "sm_80"},
    )

    expected = textwrap.dedent("""\
        module attrs={target: "sm_80"} {
          kernel @cwise_add(%lhs: gmem<1048576xf32>, %rhs: gmem<1048576xf32>, %dst: gmem<1048576xf32>) attrs={ptx_optimization_level: 1, version: 1} {
            %0: gmem<1048576xf32> = ptx_template(%lhs: gmem<1048576xf32>, %rhs: gmem<1048576xf32>, %dst: gmem<1048576xf32>) attrs={template: "add.f32 ${f32:s}, ${f32:a}, ${f32:b};\\n"}
          }
        }""")  # noqa: E501
    assert_print_and_roundtrip(module, expected)

    with pytest.raises(ValueError, match="ptx_optimization_level"):
        KernelOp.build("invalid", Block(), ptx_optimization_level=5)


def test_kernel_cluster_dimension_roundtrip():
    kernel = KernelOp.build(
        "clustered", Block(), grid_dim=(2, 1, 1), block_dim=(32, 1, 1),
        cluster_dim=(2, 1, 1),
    )

    assert kernel.cluster_dim == (2, 1, 1)
    assert_print_and_roundtrip(
        kernel,
        "kernel @clustered() grid_dim=[2, 1, 1] block_dim=[32, 1, 1] "
        "cluster_dim=[2, 1, 1] {\n}",
    )
    with pytest.raises(ValueError, match="cluster_dim entries must be positive"):
        KernelOp.build("invalid", Block(), cluster_dim=(0, 1, 1))


def test_partial_split_block_ids_roundtrip_and_validation():
    op = PartialSplitBlockIdsOp.build(
        m_tiles=32, n_tiles=16, split_rows=24,
    )

    assert_print_and_roundtrip(
        op,
        "%split_m: index, %split_n: index, %split_narrow: index, "
        "%split_half: index = partial_split_block_ids() "
        "attrs={m_tiles: 32, n_tiles: 16, split_rows: 24}",
    )
    with pytest.raises(ValueError, match="0 < split_rows < m_tiles"):
        PartialSplitBlockIdsOp.build(
            m_tiles=32, n_tiles=16, split_rows=32,
        )


def test_tile_element_offset_dimensions_roundtrip_and_validation():
    tensor_type = GmemTensorType.build((128, 256), F32Type())
    tensor = Value(name="tensor", type=tensor_type)
    offset = Value(name="offset", type=IndexType())
    tile = TileOp.build(
        tensor, [0, offset],
        GmemTensorType.build((128, 32), F32Type(), strides=(256, 1)),
        attrs={"element_offset_dims": [1]},
    )

    assert_print_and_roundtrip(
        tile,
        "%0: gmem<128x32xf32 strides=[256, 1]> = "
        "tile(%tensor: gmem<128x256xf32>, 0, "
        "%offset: index) attrs={element_offset_dims: [1]}",
    )
    with pytest.raises(ValueError, match="element_offset_dims"):
        TileOp.build(
            tensor, [0, offset],
            GmemTensorType.build((128, 32), F32Type(), strides=(256, 1)),
            attrs={"element_offset_dims": [2]},
        )


def test_split_n_block_id_roundtrip_and_validation():
    op = SplitNBlockIdOp.build(n_tiles=16, split_tile=0)

    assert_print_and_roundtrip(
        op,
        "%split_n: index, %split_narrow: index, %split_half: index = "
        "split_n_block_id() attrs={n_tiles: 16, split_tile: 0}",
    )
    with pytest.raises(ValueError, match="valid split_tile"):
        SplitNBlockIdOp.build(n_tiles=16, split_tile=16)


def test_module_explicit_target():
    module = ModuleOp.build(Block(), target=BLACKWELL)

    assert module.ptx_target == BLACKWELL
    assert repr(module) == textwrap.dedent("""\
        module attrs={target: "sm_100a"} {
        }""")
    assert ModuleOp.build(Block()).ptx_target == HOPPER

    with pytest.raises(ValueError, match="target conflicts with attrs: target"):
        ModuleOp.build(Block(), attrs={"target": "sm_80"}, target=BLACKWELL)


def test_tensor_memory_type_and_dealloc_roundtrip():
    tmem = TmemTensorType.build([128, 64], F32Type())
    value = Value(name="acc", type=tmem)
    op = DeallocOp.build(value)

    assert repr(tmem) == "tmem<128x64xf32>"
    assert repr(gpu_dialect.parse(repr(op))) == repr(op)
    with pytest.raises(AssertionError, match="columns must be a power of two"):
        TmemTensorType.build([128, 48], F32Type())


def test_kernel_num_threads():
    kernel = KernelOp.build(
        "k", Block(),
        block_dim=(256, 1, 1), num_warp_groups=2,
    )

    assert kernel.num_threads == 256


def test_for_loop():
    op = ForOp.build(0, 64, 4)
    op.body.append(Op(operands=[op.iv], results=[]))
    expected = textwrap.dedent("""\
        for (%0: index = 0 to 64 step 4) {
          generic(%0: index)
        }""")
    assert_print_and_roundtrip(op, expected)


def test_async_op():
    t = GmemTensorType.build((64, 64), F32Type())
    lhs0 = Value(name="lhs0", type=t)
    rhs0 = Value(name="rhs0", type=t)
    lhs1 = Value(name="lhs1", type=t)
    rhs1 = Value(name="rhs1", type=t)
    dst = Value(name="dst", type=t)
    lock = Value(name="lock", type=LockType.build(kind="wgmma"))
    body = Block(ops=[
        MatmulOp.build(lhs0, rhs0, dst),
        MatmulOp.build(lhs1, rhs1, dst),
    ])
    op = AsyncOp.build(lock, body=body)
    expected = textwrap.dedent("""\
        async(%lock: lock<wgmma>) {
          matmul(%lhs0: gmem<64x64xf32>, %rhs0: gmem<64x64xf32>, %dst: gmem<64x64xf32>)
          matmul(%lhs1: gmem<64x64xf32>, %rhs1: gmem<64x64xf32>, %dst: gmem<64x64xf32>)
        }""")  # noqa: E501
    assert_print_and_roundtrip(op, expected)


def test_async_commit_op():
    lock = Value(name="lock", type=LockType.build(kind="tcgen05"))
    op = AsyncCommitOp.build(lock)
    assert_print_and_roundtrip(
        op, "async_commit(%lock: lock<tcgen05>)",
    )


def test_async_op_single_move():
    t = GmemTensorType.build((64, 64), F32Type())
    dst = Value(name="dst", type=t)
    src = Value(name="src", type=t)
    lock = Value(name="lock", type=LockType.build(kind="bulk"))
    body = Block(ops=[MoveOp.build(dst, src)])
    op = AsyncOp.build(lock, body=body)
    expected = "async(%lock: lock<bulk>) move(%dst: gmem<64x64xf32>, %src: gmem<64x64xf32>)"  # noqa: E501
    assert_print_and_roundtrip(op, expected)


def test_tma_move_build_accepts_l2_cache_hint():
    gmem_type = GmemTensorType.build((64, 64), F32Type())
    desc = Value(
        name="desc",
        type=TmaDescType.build(0, gmem_type, tile_shape=[64, 64]),
    )
    smem = Value(name="smem", type=SmemTensorType.build((64, 64), F32Type()))
    indices = [Value(name=f"i{i}", type=IndexType()) for i in range(2)]

    op = TmaMoveOp.build(
        smem, desc, indices, l2_cache_hint="evict_last",
    )

    assert op.l2_cache_hint == "evict_last"


def test_tma_descriptor_validates_l2_promotion():
    gmem_type = GmemTensorType.build((64, 64), F32Type())
    desc = TmaDescType.build(
        0, gmem_type, tile_shape=[64, 64],
        l2_promotion="256B",
    )
    assert desc.l2_promotion == "256B"
    with pytest.raises(ValueError, match="unsupported TMA L2 promotion"):
        TmaDescType.build(
            0, gmem_type, tile_shape=[64, 64],
            l2_promotion="512B",
        )


def test_for_loop_default_step():
    op = ForOp.build(0, 4096)
    op.body.append(Op(operands=[op.iv], results=[]))
    expected = textwrap.dedent("""\
        for (%0: index = 0 to 4096) {
          generic(%0: index)
        }""")
    assert_print_and_roundtrip(op, expected)









def test_canonical_strides():
    assert AddrMemTensorType.canonical_strides([4096, 4096]) == [4096, 1]
    assert AddrMemTensorType.canonical_strides([128, 64]) == [64, 1]
    assert AddrMemTensorType.canonical_strides([64, 128]) == [128, 1]
    assert AddrMemTensorType.canonical_strides([16]) == [1]
    assert AddrMemTensorType.canonical_strides([4, 3, 2]) == [6, 2, 1]

    t = GmemTensorType.build([128, 64], F32Type())
    assert list(t.strides) == [64, 1]
    assert list(t.strides) == AddrMemTensorType.canonical_strides([128, 64])

    t_strided = GmemTensorType.build([64, 64], F32Type(), strides=[4096, 1])
    assert list(t_strided.strides) == [4096, 1]
    assert list(t_strided.strides) != AddrMemTensorType.canonical_strides([64, 64])

    t_smem = SmemTensorType.build([128, 64], F32Type())
    assert list(t_smem.strides) == [64, 1]


def test_naive_matmul_4096_module():
    module_str = textwrap.dedent(f"""\
        module {{
          kernel @naive_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {{
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<4096x64xf32 strides=[4096, 1]> = tile(%b: gmem<4096x4096xf32>, 0, %cy: index)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<2x4096xf32> = tile(%a_block: gmem<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem<4096x2xf32 strides=[4096, 1]> = tile(%b_block: gmem<4096x64xf32 strides=[4096, 1]>, 0, %ty: index)
            %c_thread: gmem<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem<2x2xf32> = alloc() attrs={{id: 0, init: 0.0}}
            for (%k: index = 0 to 4096) {{
              %a_sloop: gmem<2x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem<2x4096xf32>, 0, %k: index)
              %b_sloop: gmem<1x2xf32 strides=[4096, 1]> = tile(%b_thread: gmem<4096x2xf32 strides=[4096, 1]>, %k: index, 0)
              %a_sloop_rmem: rmem<2x1xf32> = alloc() attrs={{id: 1}}
              move(%a_sloop_rmem: rmem<2x1xf32>, %a_sloop: gmem<2x1xf32 strides=[4096, 1]>)
              %b_sloop_rmem: rmem<1x2xf32> = alloc() attrs={{id: 2}}
              move(%b_sloop_rmem: rmem<1x2xf32>, %b_sloop: gmem<1x2xf32 strides=[4096, 1]>)
              matmul(%a_sloop_rmem: rmem<2x1xf32>, %b_sloop_rmem: rmem<1x2xf32>, %c_thread_rmem: rmem<2x2xf32>)
            }}
            move(%c_thread: gmem<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem<2x2xf32>)
          }}
        }}""")  # noqa: E501
    module = gpu_dialect.parse(module_str)

    assert_print_and_roundtrip(module, module_str)
