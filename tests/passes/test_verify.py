import textwrap

import pytest

from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    AsyncOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    ForOp,
    GmemTensorType,
    IndexType,
    LockType,
    ModuleOp,
    MoveOp,
    ReshapeOp,
    RmemTensorType,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    gpu_dialect,
)
from ayla.passes.verify_gpu_dialect import (
    VerificationStage,
    verify_gpu_dialect,
)
from ayla.ptx_codegen import make_ptx_kernel


def _dynamic_matmul_module() -> ModuleOp:
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @dynamic(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>) block_dim=[128, None, None] {
            matmul(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>) collective=xyz.grid
          }
        }"""))
    assert isinstance(module, ModuleOp)
    return module


def test_default_verification_allows_dynamic_shapes():
    module = _dynamic_matmul_module()

    assert verify_gpu_dialect(module) is module


@pytest.mark.parametrize("step", [0, -1])
def test_default_verification_rejects_nonpositive_loop_step(step):
    module = ModuleOp.build(Block(ops=[ForOp.build(0, 4, step)]))

    with pytest.raises(ValueError, match="positive integer"):
        verify_gpu_dialect(module)


def test_default_verification_checks_tma_move_structure():
    gmem_type = GmemTensorType.build([64, 64], F32Type())
    descriptor = Value(
        name="descriptor",
        type=TmaDescType.build(0, gmem_type, tile_shape=[64, 64]),
    )
    smem = Value(
        name="smem", type=SmemTensorType.build([64, 64], F32Type()),
    )
    gmem = Value(name="gmem", type=gmem_type)
    indices = [Value(name=f"i{i}", type=IndexType()) for i in range(2)]
    collective = {
        axis: CollectiveLevel.BLOCK for axis in CollectiveAxis
    }

    valid = ModuleOp.build(Block(ops=[TmaMoveOp.build(
        smem, descriptor, indices=indices, collective=collective,
    )]))
    assert verify_gpu_dialect(valid) is valid

    wrong_index_count = ModuleOp.build(Block(ops=[TmaMoveOp.build(
        smem, descriptor, indices=indices[:1], collective=collective,
    )]))
    with pytest.raises(ValueError, match="one index per TMA dimension"):
        verify_gpu_dialect(wrong_index_count)

    wrong_memory_space = ModuleOp.build(Block(ops=[TmaMoveOp.build(
        gmem, descriptor, indices=indices, collective=collective,
    )]))
    with pytest.raises(ValueError, match="one TMA descriptor and one smem"):
        verify_gpu_dialect(wrong_memory_space)

    missing_descriptor = ModuleOp.build(Block(ops=[TmaMoveOp.build(
        smem, gmem, indices=indices, collective=collective,
    )]))
    with pytest.raises(ValueError, match="one TMA descriptor and one smem"):
        verify_gpu_dialect(missing_descriptor)


def test_default_verification_checks_tile_result_type():
    source = Value(
        name="source",
        type=GmemTensorType.build([4, 4], F32Type()),
    )
    wrong_memory_space = TileOp.build(
        source,
        [0, 0],
        SmemTensorType.build([4, 4], F32Type()),
    )
    module = ModuleOp.build(Block(ops=[wrong_memory_space]))

    with pytest.raises(ValueError, match="preserve.*memory space"):
        verify_gpu_dialect(module)

    tile = TileOp.build(
        source,
        [0, 0],
        GmemTensorType.build([2, 2], F32Type()),
    )
    module = ModuleOp.build(Block(ops=[tile]))

    with pytest.raises(ValueError, match="result strides.*must match"):
        verify_gpu_dialect(module)


@pytest.mark.parametrize(
    "indices,result_sizes",
    [
        ([0], [4, 4]),
        ([0], [4]),
    ],
)
def test_default_verification_checks_tile_rank(indices, result_sizes):
    source = Value(
        name="source",
        type=GmemTensorType.build([4, 4], F32Type()),
    )
    tile = TileOp.build(
        source,
        indices,
        GmemTensorType.build(result_sizes, F32Type()),
    )

    with pytest.raises(ValueError, match="rank.*number of indices must match"):
        verify_gpu_dialect(ModuleOp.build(Block(ops=[tile])))


def test_async_and_body_collectives_must_match():
    typ = GmemTensorType.build([1], F32Type())
    block_collective = {
        axis: CollectiveLevel.BLOCK for axis in CollectiveAxis
    }
    thread_collective = {
        axis: CollectiveLevel.THREAD for axis in CollectiveAxis
    }
    move = MoveOp.build(
        Value(name="dst", type=typ),
        Value(name="src", type=typ),
        collective=block_collective,
    )
    async_op = AsyncOp.build(
        Value(name="lock", type=LockType.build("bulk")),
        collective=thread_collective,
        body=Block(ops=[move]),
    )
    module = ModuleOp.build(Block(ops=[async_op]))

    with pytest.raises(ValueError, match="async collective must match"):
        verify_gpu_dialect(module)


def test_default_verification_rejects_inconsistent_matmul_shapes():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%a: gmem<7x11xbf16>, %b: gmem<13x12xbf16>, %c: gmem<7x13xbf16>) {
            matmul(%a: gmem<7x11xbf16>, %b: gmem<13x12xbf16>, %c: gmem<7x13xbf16>) collective=xyz.grid
          }
        }"""))

    with pytest.raises(ValueError, match="inconsistent matmul shapes"):
        verify_gpu_dialect(module)


def test_default_verification_checks_reshape():
    source = Value(
        name="source",
        type=GmemTensorType.build([2, 3, 4], F32Type()),
    )
    valid = ModuleOp.build(Block(ops=[
        ReshapeOp.build(
            source, GmemTensorType.build([6, 4], F32Type()),
        ),
    ]))
    assert verify_gpu_dialect(valid) is valid

    register_source = Value(
        name="register_source",
        type=RmemTensorType.build([2, 3, 4], F32Type()),
    )
    register_reshape = ModuleOp.build(Block(ops=[
        ReshapeOp.build(
            register_source, RmemTensorType.build([6, 4], F32Type()),
        ),
    ]))
    assert verify_gpu_dialect(register_reshape) is register_reshape

    wrong_size = ModuleOp.build(Block(ops=[
        ReshapeOp.build(
            source, GmemTensorType.build([5, 4], F32Type()),
        ),
    ]))
    with pytest.raises(ValueError, match="preserve the number of elements"):
        verify_gpu_dialect(wrong_size)

    strided = ModuleOp.build(Block(ops=[
        ReshapeOp.build(
            source,
            GmemTensorType.build([6, 4], F32Type(), strides=[5, 1]),
        ),
    ]))
    with pytest.raises(ValueError, match="canonical strides"):
        verify_gpu_dialect(strided)


def test_elementwise_function_and_arity_are_verified():
    typ = GmemTensorType.build([8], F32Type())
    dst = Value(name="dst", type=typ)
    srcs = [Value(name=f"src{i}", type=typ) for i in range(2)]

    valid = ModuleOp.build(Block(ops=[
        ElementwiseOp.build(dst, srcs=srcs, fn="mul"),
    ]))
    assert verify_gpu_dialect(valid) is valid

    invalid_arity = ModuleOp.build(Block(ops=[
        ElementwiseOp.build(dst, srcs=srcs[:1], fn="mul"),
    ]))
    with pytest.raises(ValueError, match="mul expects 2 sources, got 1"):
        verify_gpu_dialect(invalid_arity)

    unknown = ModuleOp.build(Block(ops=[
        ElementwiseOp.build(dst, srcs=srcs[:1], fn="unknown"),
    ]))
    with pytest.raises(ValueError, match="unknown elementwise function"):
        verify_gpu_dialect(unknown)


@pytest.mark.parametrize("cluster_dim", [(2, 0, 1), (2, 1)])
def test_default_verification_rejects_invalid_cluster_dimensions(cluster_dim):
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @invalid() grid_dim=[4, 1, 1] cluster_dim={list(cluster_dim)} {{
          }}
        }}"""))

    with pytest.raises(ValueError, match="three positive dimensions"):
        verify_gpu_dialect(module)


def test_default_verification_rejects_cluster_that_does_not_divide_grid():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid() grid_dim=[10, 1, 1] cluster_dim=[3, 1, 1] {
          }
        }"""))

    with pytest.raises(ValueError, match="cluster_dim must divide grid_dim"):
        verify_gpu_dialect(module)


@pytest.mark.parametrize(
    "op",
    [
        (
            "%cbx: index, %cby: index, %cbz: index "
            "= cluster_block_ids()"
        ),
        'barrier() attrs={scope: "cluster_to_leader"}',
        'barrier() attrs={scope: "leader_to_cluster"}',
        'barrier() attrs={scope: "cluster"}',
    ],
)
def test_cluster_operations_require_a_clustered_kernel(op):
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @invalid() grid_dim=[1, 1, 1] {{
            {op}
          }}
        }}"""))

    with pytest.raises(ValueError, match="enclosing clustered kernel"):
        verify_gpu_dialect(module)


def test_tma_multicast_accepts_clustered_load():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @valid(%src: gmem<16x16xf32>) grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1] {
            %dst: smem<16x16xf32> = alloc() attrs={id: 0}
            move(%dst: smem<16x16xf32>, %src: gmem<16x16xf32>) collective=xyz.cluster attrs={multicast: True, tma: True}
          }
        }"""))

    assert verify_gpu_dialect(module) is module


@pytest.mark.parametrize(
    "kernel_attrs,move_attrs,error",
    [
        ('grid_dim=[1, 1, 1]', 'multicast: True, tma: True', "clustered kernel"),
        ('grid_dim=[1, 1, 1] cluster_dim=[1, 1, 1]', 'multicast: True, tma: True', "at least two CTAs"),
        ('grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1]', 'multicast: 1, tma: True', "must be a bool"),
        ('grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1]', 'multicast: True', "requires a TMA move"),
    ],
)
def test_tma_multicast_rejects_invalid_context(kernel_attrs, move_attrs, error):
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @invalid(%src: gmem<16x16xf32>) {kernel_attrs} {{
            %dst: smem<16x16xf32> = alloc() attrs={{id: 0}}
            move(%dst: smem<16x16xf32>, %src: gmem<16x16xf32>) collective=xyz.cluster attrs={{{move_attrs}}}
          }}
        }}"""))

    with pytest.raises(ValueError, match=error):
        verify_gpu_dialect(module)


def test_cluster_to_leader_barrier_requires_cluster_mbar_lock():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid() grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1] {
            %lock: lock<mbar> = alloc()
            barrier(%lock: lock<mbar>) attrs={scope: "cluster_to_leader"}
          }
        }"""))

    with pytest.raises(ValueError, match="must be a cluster_mbar lock"):
        verify_gpu_dialect(module)


def test_cluster_mbar_cannot_be_used_as_an_async_lock():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%src: gmem<1xf32>) grid_dim=[1, 1, 1] {
            %dst: smem<1xf32> = alloc()
            %lock: lock<cluster_mbar> = alloc()
            async(%lock: lock<cluster_mbar>) move(%dst: smem<1xf32>, %src: gmem<1xf32>) attrs={tma: True}
          }
        }"""))

    with pytest.raises(ValueError, match="barrier resource, not an async lock"):
        verify_gpu_dialect(module)


def test_tma_multicast_async_collective_must_match_inner():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %dst: smem<16x16xf32> = alloc() attrs={id: 0}
            %lock: lock<mbar> = alloc() attrs={id: 1}
            async(%lock: lock<mbar>) tma_move(%dst: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={multicast: True}
          }
        }"""))

    with pytest.raises(ValueError, match="async collective must match"):
        verify_gpu_dialect(module)


@pytest.mark.parametrize("separate_groups", [False, True])
def test_tma_async_lock_rejects_mixed_multicast_modes(separate_groups):
    middle = """\
          }
          async(%lock: lock<mbar>) collective=xyz.cluster {
    """ if separate_groups else ""
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @invalid(%desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1] {{
            %c0: index = index_constant() attrs={{value: 0}}
            %dst0: smem<16x16xf32> = alloc() attrs={{id: 0}}
            %dst1: smem<16x16xf32> = alloc() attrs={{id: 1}}
            %lock: lock<mbar> = alloc() attrs={{id: 2}}
            async(%lock: lock<mbar>) collective=xyz.cluster {{
              tma_move(%dst0: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster
              {middle}tma_move(%dst1: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={{multicast: True}}
            }}
          }}
        }}"""))

    with pytest.raises(ValueError, match="same multicast mode"):
        verify_gpu_dialect(module)


def test_ptx_lowering_verification_rejects_dynamic_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        verify_gpu_dialect(
            _dynamic_matmul_module(), VerificationStage.PTX_LOWERING,
        )


def test_ptx_codegen_verifies_static_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        make_ptx_kernel(_dynamic_matmul_module())


def test_verification_rejects_parsed_grid_barrier_without_sync():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            barrier() attrs={scope: "grid"}
          }
        }"""))

    with pytest.raises(ValueError, match="grid scope requires one sync operand"):
        verify_gpu_dialect(module)


def test_verification_rejects_undersized_grid_barrier_sync():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%sync: gmem<1xf32>) {
            barrier(%sync: gmem<1xf32>) attrs={scope: "grid"}
          }
        }"""))

    with pytest.raises(ValueError, match="at least 8 bytes"):
        verify_gpu_dialect(module)
