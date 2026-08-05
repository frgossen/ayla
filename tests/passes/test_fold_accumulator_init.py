from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, MatmulOp, gpu_dialect
from ayla.passes.fold_accumulator_init import fold_accumulator_init


def test_folds_zero_fill_into_first_loop_matmul():
    module = gpu_dialect.parse("""
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %a: smem_tensor<64x16xbf16> = alloc()
            %b: smem_tensor<256x16xbf16> = alloc()
            %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 4) {
              matmul(%a: smem_tensor<64x16xbf16>, %b: smem_tensor<256x16xbf16>, %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
              matmul(%a: smem_tensor<64x16xbf16>, %b: smem_tensor<256x16xbf16>, %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
            }
          }
        }
    """)
    fold_accumulator_init(module)
    [alloc] = find_ops(
        module,
        lambda op: isinstance(op, AllocOp)
        and op.result().type.name == "wrmem_tensor",
    )
    matmuls = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert alloc.init is None
    assert matmuls[0].scale_d is not None
    assert matmuls[1].scale_d is None


def test_does_not_fold_when_first_matmul_is_in_a_nested_loop():
    module = gpu_dialect.parse("""
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %a: smem_tensor<64x16xbf16> = alloc()
            %b: smem_tensor<256x16xbf16> = alloc()
            %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 4) {
              for (%kk: index = 0 to 4) {
                matmul(%a: smem_tensor<64x16xbf16>, %b: smem_tensor<256x16xbf16>, %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
              }
            }
          }
        }
    """)
    fold_accumulator_init(module)
    [alloc] = find_ops(
        module,
        lambda op: isinstance(op, AllocOp)
        and op.result().type.name == "wrmem_tensor",
    )
    [matmul] = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert alloc.init == 0.0
    assert matmul.scale_d is None
