"""Opt-in tensor-map prefetch: placement, default compatibility, and execution."""

import pytest

from ayla.base.ir import Block, Value
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    F16Type,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    TmaDescType,
)
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import check_gemm


def gemm(**options):
    config = dict(m=256, n=128, k=512, dtype="f16", output_n_tile=128)
    config.update(options)
    return make_ptx_kernel(blackwell_tcgen05_gemm_module(**config))


@pytest.mark.parametrize("store,descriptors", [("global", 2), ("stmatrix", 3)])
def test_prefetch_is_once_per_parameter_before_initialization(store, descriptors):
    plain = gemm(output_store=store)
    disabled = gemm(output_store=store, prefetch_tma_descriptors=False)
    enabled = gemm(output_store=store, prefetch_tma_descriptors=True)
    assert plain.ptx == disabled.ptx
    assert "prefetch.tensormap" not in plain.ptx
    assert enabled.ptx.count("prefetch.tensormap") == descriptors
    assert enabled.ptx.rindex("prefetch.tensormap") < enabled.ptx.index("mbarrier.init")
    assert enabled.ptx.rindex("prefetch.tensormap") < enabled.ptx.index("cp.async.bulk.tensor")
    assert enabled.signature == disabled.signature
    assert enabled.grid_dim == disabled.grid_dim
    assert enabled.block_dim == disabled.block_dim
    assert enabled.cluster_dim == disabled.cluster_dim


def test_persistent_and_rolled_loops_do_not_repeat_descriptor_prefetch():
    kernel = gemm(
        k=1024,
        output_grid_size=2,
        rolled_operand_pipeline=True,
        cyclic_operand_pipeline=True,
        prefetch_tma_descriptors=True,
    )
    assert kernel.ptx.count("prefetch.tensormap") == 2
    assert kernel.ptx.rindex("prefetch.tensormap") < kernel.ptx.index("// BEGIN: `for")


def test_prefetch_warp_uses_linear_xyz_thread_index():
    source = Value(type=GmemTensorType.build([128, 64], F16Type()), name="source")
    descriptor = Value(
        type=TmaDescType.build(0, source.type, [128, 64]), name="descriptor"
    )
    kernel = KernelOp.build(
        "prefetch_xyz",
        Block(arguments=[source, descriptor], ops=[]),
        block_dim=(16, 4, 2),
        grid_dim=(1, 1, 1),
        attrs={"prefetch_tma_descriptors": True},
    )
    generated = make_ptx_kernel(
        ModuleOp.build(Block(ops=[kernel]), attrs={"target": "sm_100a"})
    )
    ptx = generated.ptx
    assert "prefetch.tensormap" in ptx
    assert "%tid.x" in ptx and "%tid.y" in ptx and "%tid.z" in ptx
    assert "bra.uni" in ptx
    assert "setp.lt.u32" in ptx
    assert ", 32;" in ptx
    assert ", 4," in ptx and ", 16," in ptx
    import torch

    from ayla.ptx_runtime import make_ptx_launcher

    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0):
        tensor = torch.empty((128, 64), dtype=torch.float16, device="cuda")
        make_ptx_launcher(generated)(tensor)
        torch.cuda.synchronize()


def test_no_descriptor_emits_no_prefetch_or_guard():
    def build(enabled):
        kernel = KernelOp.build(
            "no_tma", Block(ops=[]), grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            attrs={"prefetch_tma_descriptors": enabled},
        )
        return make_ptx_kernel(ModuleOp.build(Block(ops=[kernel]), attrs={"target": "sm_100a"})).ptx

    assert build(True) == build(False)


@pytest.mark.parametrize("value", [None, 0, 1, "auto"])
def test_prefetch_option_requires_boolean(value):
    with pytest.raises(ValueError, match="prefetch_tma_descriptors must be a boolean"):
        gemm(prefetch_tma_descriptors=value)


CASES = [
    dict(),
    dict(output_store="stmatrix", operand_k_tile=128, operand_stages=3),
    dict(m=512, k=528, cta_group=2, allow_partial_tiles=True,
         cyclic_operand_pipeline=True, output_store="stmatrix"),
    dict(dtype="bf16", k=384, operand_k_tile=128, operand_stages=2),
    dict(tmem_writeback_warps=8, output_store="stmatrix"),
    dict(k=1024, operand_k_tile=128, rolled_operand_pipeline=True,
         cyclic_operand_pipeline=True),
    dict(m=512, output_grid_size=2, reuse_tmem_allocation=True,
         output_store="stmatrix", cyclic_operand_pipeline=True),
    dict(m=160, n=96, k=272, output_m_tile=64, output_n_tile=96,
         allow_partial_tiles=True, output_store="stmatrix"),
    dict(m=256, k=256, output_m_tile=64, cta_group=2, output_store="stmatrix",
         tmem_writeback_register_budget=64),
]


@pytest.mark.parametrize("case", CASES)
def test_full_gemm_changed_inputs_gpu(case):
    import torch

    from ayla.ptx_runtime import make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    config = dict(m=256, n=128, k=512, dtype="f16", output_n_tile=128)
    config.update(case)
    m, n, k = (config[key] for key in ("m", "n", "k"))
    launcher = make_ptx_launcher(gemm(**dict(config, prefetch_tma_descriptors=True)))
    assert len(check_gemm(launcher, m, n, k, config["dtype"], graph=True, runs=3)) == 3
