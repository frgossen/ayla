import textwrap

import pytest

from ayla import (
    Block,
    CollectiveAxis,
    CollectiveLevel,
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
    AsyncOp,
    BarrierOp,
    BF16Type,
    ClusterBlockIdsOp,
    CollectiveScope,
    ElementwiseOp,
    F32Type,
    ForOp,
    IfOp,
    IndexLtOp,
    IndexType,
    LockType,
    MoveOp,
    PredicateType,
    ReinterpretOp,
    ReshapeOp,
    SelectOp,
    SmemTensorType,
    TmaDescType,
    TmaMoveOp,
    gpu_dialect,
    has_ordering_effect,
    memory_effects,
    root_buffer,
    root_buffers,
)


def assert_print_and_roundtrip(op, expected):
    text = repr(op)
    assert text == expected
    assert repr(gpu_dialect.parse(text)) == text


def test_reinterpret_op():
    arg_type = SmemTensorType.build((64, 64), BF16Type())
    view_type = SmemTensorType.build((32, 64), BF16Type())
    arg = Value(name="arg", type=arg_type)
    reinterpret = ReinterpretOp.build(arg, view_type, byte_offset=8192)

    assert reinterpret.arg is arg
    assert reinterpret.byte_offset == 8192
    assert root_buffer(reinterpret.result()) is arg
    assert_print_and_roundtrip(
        reinterpret,
        "%0: smem<32x64xbf16> = reinterpret(%arg: smem<64x64xbf16>) "
        "attrs={byte_offset: 8192}",
    )

    with pytest.raises(ValueError, match="byte_offset"):
        ReinterpretOp.build(arg, view_type, byte_offset=-1)
    with pytest.raises(ValueError, match="arg"):
        ReinterpretOp.build(Value(name="value", type=F32Type()), view_type)
    with pytest.raises(ValueError, match="result"):
        ReinterpretOp.build(arg, F32Type())


def test_buffer_roots_support_no_root_and_multiple_roots():
    lhs = Value(name="lhs", type=SmemTensorType.build((1,), F32Type()))
    rhs = Value(name="rhs", type=SmemTensorType.build((1,), F32Type()))
    index = Value(name="index", type=IndexType())
    select = SelectOp.build(index, [lhs, rhs])

    assert root_buffers(select.result()) == {lhs, rhs}
    with pytest.raises(AssertionError, match="multiple root buffers"):
        root_buffer(select.result())
    assert root_buffer(index) is None


def test_barrier_has_an_ordering_effect_and_unknown_memory_effects():
    barrier = BarrierOp.build()
    nested = IfOp(operands=[], results=[], blocks=[Block(ops=[barrier])])

    assert memory_effects(barrier) is None
    assert has_ordering_effect(barrier)
    assert has_ordering_effect(nested)


def test_memory_effect_retains_accessed_view_and_exposes_roots():
    source = Value(
        name="source",
        type=SmemTensorType.build([4], F32Type()),
    )
    view = ReinterpretOp.build(
        source,
        SmemTensorType.build([2], F32Type()),
    ).result()
    dst = Value(name="dst", type=SmemTensorType.build([2], F32Type()))
    effects = memory_effects(MoveOp.build(
        dst,
        view,
        collective={
            axis: CollectiveLevel.THREAD for axis in CollectiveAxis
        },
    ))

    assert effects is not None
    read = next(effect for effect in effects if effect.kind == "R")
    assert read.value is view
    assert read.roots == {source}


def test_reshape_op():
    source = Value(
        name="source",
        type=GmemTensorType.build([2, 3, 4], F32Type()),
    )
    reshape = ReshapeOp.build(
        source, GmemTensorType.build([6, 4], F32Type()),
    )

    assert reshape.arg is source
    assert root_buffer(reshape.result()) is source
    assert_print_and_roundtrip(
        reshape,
        "%0: gmem<6x4xf32> = reshape(%source: gmem<2x3x4xf32>)",
    )


def test_matmul_program():
    m, k, n = 64, 128, 32
    a_type = GmemTensorType.build((m, k), F32Type())
    b_type = GmemTensorType.build((k, n), F32Type())
    c_type = GmemTensorType.build((m, n), F32Type())

    a = Value(name="a", type=a_type)
    b = Value(name="b", type=b_type)
    c = Value(name="c", type=c_type)
    matmul = MatmulOp.build(
        a,
        b,
        c,
        collective={
            CollectiveAxis.X: CollectiveLevel.GRID,
            CollectiveAxis.Y: CollectiveLevel.GRID,
            CollectiveAxis.Z: CollectiveLevel.GRID,
        },
        attrs={"library": "cublas"},
    )
    block = Block(arguments=[a, b, c], ops=[matmul])
    program = Op(operands=[], results=[], blocks=[block])

    expected = textwrap.dedent("""\
        generic() {
        ^bb0(%a: gmem<64x128xf32>, %b: gmem<128x32xf32>, %c: gmem<64x32xf32>):
          matmul(%a: gmem<64x128xf32>, %b: gmem<128x32xf32>, %c: gmem<64x32xf32>) collective=xyz.grid attrs={library: "cublas"}
        }""")  # noqa: E501
    assert_print_and_roundtrip(program, expected)
    assert matmul.collective == {
        CollectiveAxis.X: CollectiveLevel.GRID,
        CollectiveAxis.Y: CollectiveLevel.GRID,
        CollectiveAxis.Z: CollectiveLevel.GRID,
    }
    assert CollectiveLevel.WARP.join(CollectiveLevel.WG) == CollectiveLevel.WG
    assert matmul.is_grid_collective()
    assert matmul.is_block_collective()
    matmul.narrow_collective_level(CollectiveAxis.X, CollectiveLevel.WG)
    assert not matmul.is_grid_collective()
    assert matmul.is_wg_collective()
    assert not matmul.is_block_collective(CollectiveAxis.X)
    assert matmul.is_block_collective(CollectiveAxis.Y, CollectiveAxis.Z)
    with pytest.raises(AssertionError):
        matmul.narrow_collective_level(
            CollectiveAxis.X, CollectiveLevel.BLOCK,
        )


def test_elementwise_syntax():
    tensor_type = GmemTensorType.build((8,), F32Type())
    dst = Value(name="dst", type=tensor_type)
    src = Value(name="src", type=tensor_type)
    op = ElementwiseOp.build(
        dst, [src], fn="sigmoid", attrs={"label": "activation"},
    )

    assert_print_and_roundtrip(
        op,
        'elementwise sigmoid(%dst: gmem<8xf32>, %src: gmem<8xf32>) '
        'attrs={label: "activation"}',
    )
    with pytest.raises(ValueError, match="must not also appear in attrs"):
        gpu_dialect.parse(
            'elementwise sigmoid(%dst: gmem<8xf32>, %src: gmem<8xf32>) '
            'attrs={fn: "convert"}'
        )


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


def test_kernel_num_threads():
    kernel = KernelOp.build(
        "k", Block(),
        block_dim=(256, 1, 1), num_warp_groups=2,
    )

    assert kernel.num_threads == 256


def test_kernel_num_threads_is_unresolved_with_an_unset_axis():
    kernel = KernelOp.build(
        "k", Block(),
        block_dim=(256, None, None), num_warp_groups=2,
    )

    assert kernel.num_threads is None


def test_kernel_cluster_dimensions_round_trip():
    kernel = KernelOp.build(
        "k", Block(), grid_dim=(12, 1, 1), cluster_dim=(3, 1, 1),
    )
    module = ModuleOp.build(Block(ops=[kernel]))

    assert kernel.cluster_dim == (3, 1, 1)
    assert_print_and_roundtrip(module, textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[12, 1, 1] cluster_dim=[3, 1, 1] {
          }
        }"""))


def test_kernel_cluster_dimensions_default_to_unresolved():
    kernel = KernelOp.build("k", Block())
    module = ModuleOp.build(Block(ops=[kernel]))

    assert kernel.cluster_dim == (None, None, None)
    assert_print_and_roundtrip(module, textwrap.dedent("""\
        module {
          kernel @k() {
          }
        }"""))


def test_cluster_block_ids_round_trip():
    ids = ClusterBlockIdsOp.build()

    assert ids.results == [ids.x, ids.y, ids.z]
    assert_print_and_roundtrip(
        ids,
        "%cbx: index, %cby: index, %cbz: index "
        "= cluster_block_ids()",
    )


def test_cluster_collective_scope_is_between_block_and_grid():
    assert (
        CollectiveScope.BLOCK.join(CollectiveScope.CLUSTER)
        == CollectiveScope.CLUSTER
    )
    assert (
        CollectiveScope.CLUSTER.join(CollectiveScope.GRID)
        == CollectiveScope.GRID
    )


def test_cluster_leader_collective_scope_is_directional():
    leader = CollectiveScope.CLUSTER_LEADER
    assert CollectiveScope.BLOCK_LEADER.join(leader) == leader
    assert leader.join(CollectiveScope.BLOCK_LEADER) == leader
    assert CollectiveScope.BLOCK.join(leader) == CollectiveScope.CLUSTER
    assert leader.join(CollectiveScope.BLOCK) == CollectiveScope.CLUSTER
    scopes = list(CollectiveScope)
    assert all(a.join(b) == b.join(a) for a in scopes for b in scopes)
    assert all(
        a.join(b).join(c) == a.join(b.join(c))
        for a in scopes
        for b in scopes
        for c in scopes
    )


def test_for_loop():
    op = ForOp.build(0, 64, 4)
    op.body.append(Op(operands=[op.iv], results=[]))
    expected = textwrap.dedent("""\
        for (%0: index = 0 to 64 step 4) {
          generic(%0: index)
        }""")
    assert_print_and_roundtrip(op, expected)


def test_if_op():
    condition = Value(name="condition", type=PredicateType())
    op = IfOp.build(condition, Block())
    assert_print_and_roundtrip(op, textwrap.dedent("""\
        if(%condition: pred) {
        }"""))

    with pytest.raises(AssertionError, match="expected PredicateType"):
        IfOp.build(Value(name="i", type=IndexType()), Block())


def test_barrier_op():
    sync = Value(name="sync", type=GmemTensorType.build((2,), F32Type()))
    barrier = BarrierOp.build("grid", sync=sync)

    assert_print_and_roundtrip(
        barrier,
        'barrier(%sync: gmem<2xf32>) attrs={scope: "grid"}',
    )


def test_index_lt_op():
    lhs = Value(name="lhs", type=IndexType())
    rhs = Value(name="rhs", type=IndexType())
    op = IndexLtOp.build(lhs, rhs)
    assert_print_and_roundtrip(
        op,
        "%0: pred = index_lt(%lhs: index, %rhs: index)",
    )

    not_index = Value(name="f", type=F32Type())
    with pytest.raises(AssertionError, match="expected IndexType lhs"):
        IndexLtOp.build(not_index, rhs)
    with pytest.raises(AssertionError, match="expected IndexType rhs"):
        IndexLtOp.build(lhs, not_index)


def test_async_op():
    t = GmemTensorType.build((64, 64), F32Type())
    lhs0 = Value(name="lhs0", type=t)
    rhs0 = Value(name="rhs0", type=t)
    lhs1 = Value(name="lhs1", type=t)
    rhs1 = Value(name="rhs1", type=t)
    dst = Value(name="dst", type=t)
    lock = Value(name="lock", type=LockType.build(kind="wgmma"))
    body = Block(ops=[
        MatmulOp.build(
            lhs0,
            rhs0,
            dst,
            collective={
                CollectiveAxis.X: CollectiveLevel.GRID,
                CollectiveAxis.Y: CollectiveLevel.GRID,
                CollectiveAxis.Z: CollectiveLevel.GRID,
            },
        ),
        MatmulOp.build(
            lhs1,
            rhs1,
            dst,
            collective={
                CollectiveAxis.X: CollectiveLevel.GRID,
                CollectiveAxis.Y: CollectiveLevel.GRID,
                CollectiveAxis.Z: CollectiveLevel.GRID,
            },
        ),
    ])
    op = AsyncOp.build(
        lock,
        collective={
            axis: CollectiveLevel.GRID for axis in CollectiveAxis
        },
        body=body,
    )
    expected = textwrap.dedent("""\
        async(%lock: lock<wgmma>) collective=xyz.grid {
          matmul(%lhs0: gmem<64x64xf32>, %rhs0: gmem<64x64xf32>, %dst: gmem<64x64xf32>) collective=xyz.grid
          matmul(%lhs1: gmem<64x64xf32>, %rhs1: gmem<64x64xf32>, %dst: gmem<64x64xf32>) collective=xyz.grid
        }""")  # noqa: E501
    assert_print_and_roundtrip(op, expected)


def test_async_op_single_move():
    t = GmemTensorType.build((64, 64), F32Type())
    dst = Value(name="dst", type=t)
    src = Value(name="src", type=t)
    lock = Value(name="lock", type=LockType.build(kind="bulk"))
    collective = {
        axis: CollectiveLevel.CLUSTER for axis in CollectiveAxis
    }
    body = Block(ops=[MoveOp.build(
        dst, src, collective=collective,
    )])
    op = AsyncOp.build(lock, collective=collective, body=body)
    expected = "async(%lock: lock<bulk>) collective=xyz.cluster move(%dst: gmem<64x64xf32>, %src: gmem<64x64xf32>) collective=xyz.cluster"  # noqa: E501
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
        smem,
        desc,
        indices=indices,
        collective={
            axis: CollectiveLevel.BLOCK for axis in CollectiveAxis
        },
        l2_cache_hint="evict_last",
    )

    assert op.l2_cache_hint == "evict_last"
    assert_print_and_roundtrip(
        op,
        "tma_move(%smem: smem<64x64xf32>, %desc: "
        "tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %i0: index, "
        "%i1: index) collective=xyz.block "
        'attrs={l2_cache_hint: "evict_last"}',
    )


@pytest.mark.parametrize(
    ("smem_is_dst", "expected"),
    [
        (
            True,
            {
                ("W", "smem", CollectiveScope.BLOCK_LEADER, 7),
                ("R", "desc", CollectiveScope.GRID, 8),
            },
        ),
        (
            False,
            {
                ("W", "desc", CollectiveScope.GRID, 7),
                ("R", "smem", CollectiveScope.BLOCK_LEADER, 8),
            },
        ),
    ],
)
def test_tma_move_reports_both_memory_effects(smem_is_dst, expected):
    gmem_type = GmemTensorType.build((64, 64), F32Type())
    desc = Value(
        name="desc",
        type=TmaDescType.build(0, gmem_type, tile_shape=[64, 64]),
    )
    smem = Value(name="smem", type=SmemTensorType.build((64, 64), F32Type()))
    index = Value(name="index", type=IndexType())
    dst, src = (smem, desc) if smem_is_dst else (desc, smem)

    effects = memory_effects(TmaMoveOp.build(
        dst,
        src,
        indices=[index],
        collective={
            axis: CollectiveLevel.BLOCK for axis in CollectiveAxis
        },
        attrs={"dst_tiling_scheme": 7, "src_tiling_scheme": 8},
    ))

    assert effects is not None
    assert {
        (effect.kind, root.name, effect.scope, effect.tiling_scheme)
        for effect in effects
        for root in effect.roots
    } == expected


def test_tma_multicast_reports_cluster_leader_write():
    gmem_type = GmemTensorType.build((64, 64), F32Type())
    desc = Value(
        name="desc",
        type=TmaDescType.build(0, gmem_type, tile_shape=[64, 64]),
    )
    smem = Value(name="smem", type=SmemTensorType.build((64, 64), F32Type()))
    index = Value(name="index", type=IndexType())

    effects = memory_effects(TmaMoveOp.build(
        smem,
        desc,
        indices=[index],
        collective={
            axis: CollectiveLevel.CLUSTER for axis in CollectiveAxis
        },
        multicast=True,
    ))

    assert effects is not None
    assert {
        (effect.kind, root.name, effect.scope)
        for effect in effects
        for root in effect.roots
    } == {
        ("W", "smem", CollectiveScope.CLUSTER_LEADER),
        ("R", "desc", CollectiveScope.CLUSTER_LEADER),
    }


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


def test_for_loop_trip_count():
    assert ForOp.build(0, 5, 2).trip_count == 3
    assert ForOp.build(4, 2).trip_count == 0









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


def test_dynamic_gmem_shape_roundtrips_and_defers_static_properties():
    t = GmemTensorType.build([None, 64], F32Type())

    module = ModuleOp.build(Block(ops=[
        KernelOp.build("k", Block(arguments=[Value(name="a", type=t)])),
    ]))
    expected = textwrap.dedent("""\
        module {
          kernel @k(%a: gmem<?x64xf32>) {
          }
        }""")
    assert_print_and_roundtrip(module, expected)

    assert t.is_dynamic
    assert t.strides == (64, 1)
    assert t.num_elements is None

    trailing_dynamic = GmemTensorType.build([64, None], F32Type())
    assert trailing_dynamic.strides == (None, 1)


def test_naive_matmul_4096_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @naive_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<4096x64xf32 strides=[4096, 1]> = tile(%b: gmem<4096x4096xf32>, 0, %cy: index)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<2x4096xf32> = tile(%a_block: gmem<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem<4096x2xf32 strides=[4096, 1]> = tile(%b_block: gmem<4096x64xf32 strides=[4096, 1]>, 0, %ty: index)
            %c_thread: gmem<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0, init: 0.0}
            for (%k: index = 0 to 4096) {
              %a_sloop: gmem<2x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem<2x4096xf32>, 0, %k: index)
              %b_sloop: gmem<1x2xf32 strides=[4096, 1]> = tile(%b_thread: gmem<4096x2xf32 strides=[4096, 1]>, %k: index, 0)
              %a_sloop_rmem: rmem<2x1xf32> = alloc() attrs={id: 1}
              move(%a_sloop_rmem: rmem<2x1xf32>, %a_sloop: gmem<2x1xf32 strides=[4096, 1]>)
              %b_sloop_rmem: rmem<1x2xf32> = alloc() attrs={id: 2}
              move(%b_sloop_rmem: rmem<1x2xf32>, %b_sloop: gmem<1x2xf32 strides=[4096, 1]>)
              matmul(%a_sloop_rmem: rmem<2x1xf32>, %b_sloop_rmem: rmem<1x2xf32>, %c_thread_rmem: rmem<2x2xf32>)
            }
            move(%c_thread: gmem<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem<2x2xf32>)
          }
        }""")  # noqa: E501
    module = gpu_dialect.parse(module_str)

    assert_print_and_roundtrip(module, module_str)
