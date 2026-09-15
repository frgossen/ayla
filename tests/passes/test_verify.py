import textwrap

import pytest

from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    ModuleOp,
    ReshapeOp,
    RmemTensorType,
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


def test_ptx_lowering_verification_rejects_dynamic_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        verify_gpu_dialect(
            _dynamic_matmul_module(), VerificationStage.PTX_LOWERING,
        )


def test_ptx_codegen_verifies_static_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        make_ptx_kernel(_dynamic_matmul_module())
