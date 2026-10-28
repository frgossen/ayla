# OSS PyTorch GEMM heuristic study

Generated from `/home/arseniivanov/ayla/.benchmarks/oss_gemm/raw.json`. Results are grouped into behavioral regimes; individual measurements remain in the raw CSV/JSON.

Environment: NVIDIA H100 / torch 2.13.0+cu130 / CUDA 13.0

Buckets: `xs` ≤ 128, `small` ≤ 512, `medium` ≤ 2,048, `large` ≤ 8,192, and `xl` above 8,192. Aspect ratio is `tall` or `wide` at 4:1, otherwise `balanced`. Alignment records the largest tested reduction tile dividing K.

Recorded operand layouts are normalized to contiguous A/BT and do not cause rejection. Cluster shape, operand multicast, and two-dimensional workgroup tiling are recorded explicitly; the broad ablations promote them only in regimes where they beat the current best.

## Improvement summary

Aggregates pair each shape's fastest successful Ayla configuration with its measured Torch baseline, weighted by recorded call frequency.

Across all 406 covered shapes, 228 beat Torch/cuBLAS (56.16%). The unweighted median shape reaches 101.71% of Torch/cuBLAS and the unweighted mean reaches 101.71%.

| regime | shapes | calls | covered | Torch TFLOPS | Ayla TFLOPS | % Torch | median shape % | wins | speedup | latency reduction | win rate | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16/canonical_nt/large/balanced/k128 | 23 | 435 | 23/23 | 577.68 | 606.11 | 104.92% | 92.73% | 5 | 1.05x | 4.69% | 21.74% | 264 |
| float16/canonical_nt/large/balanced/k64 | 4 | 31 | 4/4 | 676.68 | 677.62 | 100.14% | 106.52% | 3 | 1.00x | 0.14% | 75.00% | 16 |
| float16/canonical_nt/large/tall/k128 | 51 | 2014 | 51/51 | 553.17 | 572.32 | 103.46% | 102.69% | 35 | 1.03x | 3.35% | 68.63% | 276 |
| float16/canonical_nt/large/tall/k64 | 5 | 84 | 5/5 | 614.53 | 629.43 | 102.42% | 102.84% | 4 | 1.02x | 2.37% | 80.00% | 24 |
| float16/canonical_nt/large/wide/k128 | 1 | 24 | 1/1 | 642.61 | 610.57 | 95.01% | 95.01% | 0 | 0.95x | -5.25% | 0.00% | 10 |
| float16/canonical_nt/large/wide/k64 | 1 | 1 | 1/1 | 119.41 | 144.55 | 121.06% | 121.06% | 1 | 1.21x | 17.40% | 100.00% | 7 |
| float16/canonical_nt/medium/balanced/k128 | 63 | 4538 | 63/63 | 405.96 | 372.72 | 91.81% | 100.00% | 31 | 0.92x | -8.92% | 49.21% | 370 |
| float16/canonical_nt/medium/balanced/k64 | 16 | 62 | 16/16 | 442.16 | 357.59 | 80.87% | 116.35% | 12 | 0.81x | -23.65% | 75.00% | 36 |
| float16/canonical_nt/medium/tall/k128 | 25 | 611 | 25/25 | 385.56 | 359.66 | 93.28% | 96.03% | 7 | 0.93x | -7.20% | 28.00% | 167 |
| float16/canonical_nt/medium/tall/k64 | 3 | 19 | 3/3 | 464.64 | 389.04 | 83.73% | 85.33% | 1 | 0.84x | -19.43% | 33.33% | 9 |
| float16/canonical_nt/medium/wide/k128 | 19 | 920 | 19/19 | 640.11 | 666.29 | 104.09% | 105.38% | 17 | 1.04x | 3.93% | 89.47% | 99 |
| float16/canonical_nt/medium/wide/k64 | 4 | 31 | 4/4 | 703.82 | 727.26 | 103.33% | 105.65% | 4 | 1.03x | 3.22% | 100.00% | 9 |
| float16/canonical_nt/small/balanced/k128 | 24 | 920 | 24/24 | 263.27 | 232.90 | 88.46% | 89.82% | 9 | 0.88x | -13.04% | 37.50% | 52 |
| float16/canonical_nt/small/balanced/k64 | 5 | 19 | 5/5 | 4.28 | 5.55 | 129.50% | 129.49% | 5 | 1.29x | 22.78% | 100.00% | 2 |
| float16/canonical_nt/small/tall/k128 | 1 | 1 | 1/1 | 8.85 | 10.43 | 117.91% | 117.91% | 1 | 1.18x | 15.19% | 100.00% | 0 |
| float16/canonical_nt/small/wide/k128 | 19 | 708 | 19/19 | 419.39 | 396.22 | 94.48% | 97.47% | 5 | 0.94x | -5.85% | 26.32% | 301 |
| float16/canonical_nt/small/wide/k64 | 2 | 18 | 2/2 | 477.35 | 431.51 | 90.40% | 90.97% | 0 | 0.90x | -10.62% | 0.00% | 14 |
| float16/canonical_nt/xl/tall/k128 | 114 | 1179 | 114/114 | 531.09 | 544.67 | 102.56% | 103.49% | 71 | 1.03x | 2.49% | 62.28% | 656 |
| float16/canonical_nt/xl/tall/k64 | 14 | 65 | 14/14 | 197.42 | 218.67 | 110.76% | 106.95% | 9 | 1.11x | 9.72% | 64.29% | 76 |
| float16/canonical_nt/xs/balanced/k64 | 2 | 12 | 2/2 | 0.95 | 1.18 | 123.32% | 123.32% | 2 | 1.23x | 18.91% | 100.00% | 0 |
| float16/canonical_nt/xs/wide/k128 | 10 | 398 | 10/10 | 203.07 | 172.80 | 85.09% | 105.15% | 6 | 0.85x | -17.52% | 60.00% | 27 |
| OVERALL | 406 | 12090 | 406/406 | 509.67 | 504.79 | 99.04% | 101.71% | 228 | 0.99x | -0.97% | 56.16% | 2415 |

## float16/canonical_nt/large/balanced/k128

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=balanced, alignment=k128. 23 shapes, 435 recorded calls. N buckets: {'large': 8, 'medium': 13, 'xl': 2}; K buckets: {'large': 3, 'medium': 13, 'small': 4, 'xl': 1, 'xs': 2}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m192_n96_k64_s4_split2` | 23/23 | 71.5% | 68.8% | 55.5% | 0 |
| 2 | `gap_m192_n96_k64_s4_split2_dbuf2` | 23/23 | 71.3% | 68.1% | 55.2% | 0 |
| 3 | `pipe_m192_n112_k64_s4` | 23/23 | 70.8% | 69.0% | 57.5% | 0 |

Seed details:

- `gap_m192_n96_k64_s4_split2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2_dbuf2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=2, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m192_n112_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=112, warpgroup_m_tile=None, warpgroup_n_tile=112, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_rhs_keep` | ok | 105.09% |
| `probe_lhs_promote_256b` | ok | 99.72% |
| `probe_lhs_keep` | ok | 98.75% |
| `probe_ptx_o1` | ok | 96.36% |
| `probe_dst_keep` | ok | 67.44% |
| `probe_scalar_store` | ok | 63.39% |
| `probe_ktile128_n64_k2` | ok | 60.68% |
| `probe_m192_n64` | ok | 50.91% |

## float16/canonical_nt/large/balanced/k64

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=balanced, alignment=k64. 4 shapes, 31 recorded calls. N buckets: {'large': 1, 'medium': 3}; K buckets: {'large': 2, 'xl': 1, 'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `staticloop132_m128_n128_k64_s4_nfast` | 4/4 | 90.9% | 79.5% | 72.4% | 0 |
| 2 | `pipe_m128_n112_k64_s6` | 4/4 | 89.1% | 76.6% | 69.7% | 0 |
| 3 | `coarse_m128_n112_k64_s4` | 4/4 | 88.9% | 70.9% | 69.7% | 0 |

Seed details:

- `staticloop132_m128_n128_k64_s4_nfast`: output_tiles_per_cta=1, output_cta_count=132, unroll_output_tiles=False, output_tile_fast_axis='n', output_m_tile=128, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m128_n112_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=112, warpgroup_m_tile=None, warpgroup_n_tile=112, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `coarse_m128_n112_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=112, warpgroup_m_tile=None, warpgroup_n_tile=112, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_scalar_store` | ok | 68.66% |
| `probe_dst_keep` | ok | 61.70% |
| `probe_m192_n64` | ok | 40.64% |
| `probe_lhs_keep` | ValueError: TileMatmul: op_dim M size (2304) is not divisible by tile size (640) | - |
| `probe_lhs_promote_256b` | ValueError: TileMatmul: op_dim M size (2304) is not divisible by tile size (640) | - |
| `probe_ptx_o1` | ValueError: TileMatmul: op_dim M size (2304) is not divisible by tile size (640) | - |
| `probe_rhs_keep` | ValueError: TileMatmul: op_dim M size (2304) is not divisible by tile size (640) | - |

## float16/canonical_nt/large/tall/k128

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=tall, alignment=k128. 51 shapes, 2,014 recorded calls. N buckets: {'medium': 28, 'small': 18, 'xs': 5}; K buckets: {'large': 8, 'medium': 19, 'small': 17, 'xl': 2, 'xs': 5}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `refine_m64_n128_k4_dst_keep` | 51/51 | 76.4% | 78.0% | 58.6% | 0 |
| 2 | `refine_m64_n128_k4_rhs_keep` | 51/51 | 76.4% | 77.9% | 58.9% | 0 |
| 3 | `coarse_m192_n128_k64_s3` | 51/51 | 75.0% | 72.8% | 48.1% | 0 |

Seed details:

- `refine_m64_n128_k4_dst_keep`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_last', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `refine_m64_n128_k4_rhs_keep`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `coarse_m192_n128_k64_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=3, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 64.77% |
| `probe_scalar_store` | ok | 54.77% |
| `probe_ktile128_n64_k2` | ok | 54.28% |
| `probe_lhs_keep` | ok | 47.61% |
| `probe_rhs_keep` | ok | 47.45% |
| `probe_lhs_promote_256b` | ok | 47.31% |
| `probe_ptx_o1` | ok | 46.85% |
| `probe_m192_n64` | ok | 45.09% |

## float16/canonical_nt/large/tall/k64

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=tall, alignment=k64. 5 shapes, 84 recorded calls. N buckets: {'medium': 4, 'small': 1}; K buckets: {'large': 2, 'small': 2, 'xl': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `staticloop128_m192_n128_k64_s4_nfast` | 5/5 | 88.0% | 71.4% | 70.5% | 0 |
| 2 | `gap_m192_n96_k64_s4_split2` | 5/5 | 87.5% | 80.6% | 56.4% | 0 |
| 3 | `pipe_m192_n112_k64_s4` | 5/5 | 87.2% | 74.9% | 65.9% | 0 |

Seed details:

- `staticloop128_m192_n128_k64_s4_nfast`: output_tiles_per_cta=1, output_cta_count=128, unroll_output_tiles=False, output_tile_fast_axis='n', output_m_tile=192, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m192_n112_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=112, warpgroup_m_tile=None, warpgroup_n_tile=112, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 91.57% |
| `probe_scalar_store` | ok | 89.41% |
| `probe_m192_n64` | ok | 59.29% |
| `probe_lhs_keep` | ok | 32.12% |
| `probe_lhs_promote_256b` | ok | 32.10% |
| `probe_rhs_keep` | ok | 32.08% |
| `probe_ptx_o1` | ok | 31.79% |

## float16/canonical_nt/large/wide/k128

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=wide, alignment=k128. 1 shapes, 24 recorded calls. N buckets: {'xl': 1}; K buckets: {'medium': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `xfer_m128_n256_k64_s3_c0_7a789e57fd` | 1/1 | 99.7% | 99.7% | 99.7% | 0 |
| 2 | `xfer_m128_n256_k64_s3_c256_556c32cfa0` | 1/1 | 98.0% | 98.0% | 98.0% | 0 |
| 3 | `local_m128_n256_wm0_wn256_k64_s3_sl2_b1_c0_nn_llrfdf_o2_u0_t128` | 1/1 | 97.8% | 97.8% | 97.8% | 0 |

Seed details:

- `xfer_m128_n256_k64_s3_c0_7a789e57fd` (wins at least one shape): output_tiles_per_cta=4, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis='n', output_m_tile=128, output_n_tile=256, warpgroup_m_tile=None, warpgroup_n_tile=256, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `xfer_m128_n256_k64_s3_c256_556c32cfa0`: output_tiles_per_cta=1, output_cta_count=256, unroll_output_tiles=None, output_tile_fast_axis='m', output_m_tile=128, output_n_tile=256, warpgroup_m_tile=None, warpgroup_n_tile=256, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m128_n256_wm0_wn256_k64_s3_sl2_b1_c0_nn_llrfdf_o2_u0_t128`: output_tiles_per_cta=4, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=128, output_n_tile=256, warpgroup_m_tile=None, warpgroup_n_tile=256, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=2, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_rhs_keep` | ok | 92.37% |
| `probe_lhs_keep` | ok | 91.35% |
| `probe_ptx_o1` | ok | 87.04% |
| `probe_lhs_promote_256b` | ok | 87.03% |
| `probe_dst_keep` | ok | 61.52% |
| `probe_ktile128_n64_k2` | ok | 58.53% |
| `probe_scalar_store` | ok | 56.72% |
| `probe_m192_n64` | ok | 46.22% |

## float16/canonical_nt/large/wide/k64

dtype=float16, layout=canonical_nt, m_bucket=large, aspect=wide, alignment=k64. 1 shapes, 1 recorded calls. N buckets: {'xl': 1}; K buckets: {'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `local_m64_n96_wm0_wn96_k64_s4_sl2_b2_c0_nn_lfrldf_o0_u0_t0` | 1/1 | 120.4% | 120.4% | 120.4% | 0 |
| 2 | `local_m64_n96_wm0_wn96_k64_s6_sl2_b1_c0_nn_lfrldf_o0_u0_t0` | 1/1 | 120.4% | 120.4% | 120.4% | 0 |
| 3 | `local_m64_n96_wm0_wn96_k64_s4_sl2_b1_c0_nn_lfrldf_o1_u0_t0` | 1/1 | 120.3% | 120.3% | 120.3% | 0 |

Seed details:

- `local_m64_n96_wm0_wn96_k64_s4_sl2_b2_c0_nn_lfrldf_o0_u0_t0` (wins at least one shape): output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=2, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m64_n96_wm0_wn96_k64_s6_sl2_b1_c0_nn_lfrldf_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m64_n96_wm0_wn96_k64_s4_sl2_b1_c0_nn_lfrldf_o1_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=1, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_rhs_keep` | ok | 102.02% |
| `probe_lhs_keep` | ok | 101.98% |
| `probe_lhs_promote_256b` | ok | 98.87% |
| `probe_scalar_store` | ok | 98.67% |
| `probe_ptx_o1` | ok | 96.41% |
| `probe_m192_n64` | ok | 84.85% |
| `probe_dst_keep` | ok | 80.51% |

## float16/canonical_nt/medium/balanced/k128

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=balanced, alignment=k128. 63 shapes, 4,538 recorded calls. N buckets: {'large': 12, 'medium': 39, 'small': 12}; K buckets: {'large': 12, 'medium': 22, 'small': 13, 'xl': 3, 'xs': 13}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m128_n96_k64_s5_split2_dbuf2` | 63/63 | 75.6% | 79.5% | 55.1% | 0 |
| 2 | `refine_scalar_n64_k4` | 63/63 | 75.0% | 77.2% | 43.2% | 0 |
| 3 | `fallback_scalar_n64_k3` | 63/63 | 74.4% | 77.3% | 42.9% | 0 |

Seed details:

- `gap_m128_n96_k64_s5_split2_dbuf2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=2, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `refine_scalar_n64_k4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='scalar', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `fallback_scalar_n64_k3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=3, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='scalar', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 79.58% |
| `probe_scalar_store` | ok | 77.51% |
| `probe_ktile128_n64_k2` | ok | 69.58% |
| `probe_m192_n64` | ok | 62.68% |
| `probe_rhs_keep` | ok | 46.28% |
| `probe_lhs_keep` | ok | 46.14% |
| `probe_ptx_o1` | ok | 45.67% |
| `probe_lhs_promote_256b` | ok | 45.62% |

## float16/canonical_nt/medium/balanced/k64

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=balanced, alignment=k64. 16 shapes, 62 recorded calls. N buckets: {'medium': 10, 'small': 6}; K buckets: {'large': 3, 'xl': 2, 'xs': 11}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m128_n64_block_k64_s5` | 16/16 | 86.9% | 108.8% | 68.7% | 0 |
| 2 | `pipe_m128_n80_k64_s5` | 16/16 | 84.8% | 106.2% | 66.4% | 0 |
| 3 | `pipe_m128_n80_k64_s6` | 16/16 | 84.3% | 106.5% | 67.0% | 0 |

Seed details:

- `gap_m128_n64_block_k64_s5`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m128_n80_k64_s5`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=80, warpgroup_m_tile=None, warpgroup_n_tile=80, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m128_n80_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=80, warpgroup_m_tile=None, warpgroup_n_tile=80, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 53.59% |
| `probe_scalar_store` | ok | 50.54% |
| `probe_m192_n64` | ok | 45.29% |
| `probe_lhs_promote_256b` | ok | 28.57% |
| `probe_ptx_o1` | ok | 28.56% |
| `probe_lhs_keep` | ok | 28.56% |
| `probe_rhs_keep` | ok | 28.56% |

## float16/canonical_nt/medium/tall/k128

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=tall, alignment=k128. 25 shapes, 611 recorded calls. N buckets: {'small': 21, 'xs': 4}; K buckets: {'large': 6, 'medium': 5, 'small': 8, 'xl': 3, 'xs': 3}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m128_n64_block_k64_s5` | 25/25 | 90.0% | 85.4% | 55.2% | 0 |
| 2 | `gap_m128_n64_block_k128_s4` | 25/25 | 89.0% | 82.6% | 55.3% | 0 |
| 3 | `gap_m128_n64_block_k64_s4` | 25/25 | 88.9% | 84.5% | 49.4% | 0 |

Seed details:

- `gap_m128_n64_block_k64_s5`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m128_n64_block_k128_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m128_n64_block_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 104.99% |
| `probe_scalar_store` | ok | 92.38% |
| `probe_ktile128_n64_k2` | ok | 91.64% |
| `probe_m192_n64` | ok | 81.62% |
| `probe_rhs_keep` | ok | 72.23% |
| `probe_lhs_promote_256b` | ok | 71.83% |
| `probe_ptx_o1` | ok | 71.53% |
| `probe_lhs_keep` | ok | 71.28% |

## float16/canonical_nt/medium/tall/k64

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=tall, alignment=k64. 3 shapes, 19 recorded calls. N buckets: {'small': 2, 'xs': 1}; K buckets: {'xl': 2, 'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `pipe_m128_n80_k64_s5` | 3/3 | 81.7% | 85.1% | 78.5% | 0 |
| 2 | `pipe_m128_n80_k64_s6` | 3/3 | 81.0% | 83.3% | 78.2% | 0 |
| 3 | `pipe_m128_n96_k64_s5` | 3/3 | 78.3% | 84.5% | 74.2% | 0 |

Seed details:

- `pipe_m128_n80_k64_s5` (wins at least one shape): output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=80, warpgroup_m_tile=None, warpgroup_n_tile=80, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m128_n80_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=80, warpgroup_m_tile=None, warpgroup_n_tile=80, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `pipe_m128_n96_k64_s5`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 65.47% |
| `probe_scalar_store` | ok | 56.67% |
| `probe_m192_n64` | ok | 51.67% |
| `probe_lhs_promote_256b` | ok | 32.75% |
| `probe_ptx_o1` | ok | 32.71% |
| `probe_lhs_keep` | ok | 32.70% |
| `probe_rhs_keep` | ok | 32.68% |

## float16/canonical_nt/medium/wide/k128

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=wide, alignment=k128. 19 shapes, 920 recorded calls. N buckets: {'large': 18, 'xl': 1}; K buckets: {'large': 4, 'medium': 5, 'small': 4, 'xl': 3, 'xs': 3}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `coarse_m128_n144_k64_s4` | 19/19 | 86.7% | 93.4% | 71.2% | 0 |
| 2 | `gap_m192_n96_k64_s4_split2` | 19/19 | 84.3% | 90.4% | 68.2% | 0 |
| 3 | `gap_m192_n96_k64_s4_split2_dbuf2` | 19/19 | 84.0% | 89.8% | 68.1% | 0 |

Seed details:

- `coarse_m128_n144_k64_s4` (wins at least one shape): output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=144, warpgroup_m_tile=None, warpgroup_n_tile=144, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2_dbuf2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=2, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_rhs_keep` | ok | 100.59% |
| `probe_lhs_promote_256b` | ok | 97.71% |
| `probe_lhs_keep` | ok | 97.42% |
| `probe_ptx_o1` | ok | 97.22% |
| `probe_dst_keep` | ok | 86.05% |
| `probe_scalar_store` | ok | 75.17% |
| `probe_ktile128_n64_k2` | ok | 69.17% |
| `probe_m192_n64` | ok | 53.99% |

## float16/canonical_nt/medium/wide/k64

dtype=float16, layout=canonical_nt, m_bucket=medium, aspect=wide, alignment=k64. 4 shapes, 31 recorded calls. N buckets: {'large': 4}; K buckets: {'large': 2, 'xl': 1, 'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `coarse_m128_n144_k64_s4` | 4/4 | 98.1% | 92.1% | 86.3% | 0 |
| 2 | `gap_m192_n96_k64_s4_split2_dbuf2` | 4/4 | 96.4% | 91.2% | 84.0% | 0 |
| 3 | `gap_m192_n96_k64_s4_split2` | 4/4 | 95.1% | 90.6% | 82.4% | 0 |

Seed details:

- `coarse_m128_n144_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=144, warpgroup_m_tile=None, warpgroup_n_tile=144, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2_dbuf2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=2, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m192_n96_k64_s4_split2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=192, output_n_tile=96, warpgroup_m_tile=None, warpgroup_n_tile=96, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=2, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_lhs_promote_256b` | ok | 64.81% |
| `probe_lhs_keep` | ok | 64.80% |
| `probe_ptx_o1` | ok | 64.75% |
| `probe_rhs_keep` | ok | 64.73% |
| `probe_dst_keep` | ok | 55.56% |
| `probe_scalar_store` | ok | 52.09% |
| `probe_m192_n64` | ok | 50.72% |

## float16/canonical_nt/small/balanced/k128

dtype=float16, layout=canonical_nt, m_bucket=small, aspect=balanced, alignment=k128. 24 shapes, 920 recorded calls. N buckets: {'medium': 13, 'small': 11}; K buckets: {'large': 4, 'medium': 10, 'small': 8, 'xl': 1, 'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m64_n48_k64_s6` | 24/24 | 84.3% | 81.1% | 61.5% | 0 |
| 2 | `narrow_m64_n48_k64_s7` | 24/24 | 83.0% | 81.1% | 63.6% | 0 |
| 3 | `gap_m64_n72_k128_s3` | 24/24 | 82.4% | 81.1% | 64.4% | 0 |

Seed details:

- `gap_m64_n48_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `narrow_m64_n48_k64_s7`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=7, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m64_n72_k128_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=72, warpgroup_m_tile=None, warpgroup_n_tile=72, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 69.13% |
| `probe_scalar_store` | ok | 67.04% |
| `probe_ktile128_n64_k2` | ok | 65.20% |
| `probe_m192_n64` | ok | 58.39% |
| `probe_rhs_keep` | ok | 39.84% |
| `probe_lhs_keep` | ok | 39.64% |
| `probe_ptx_o1` | ok | 39.50% |
| `probe_lhs_promote_256b` | ok | 39.49% |

## float16/canonical_nt/small/balanced/k64

dtype=float16, layout=canonical_nt, m_bucket=small, aspect=balanced, alignment=k64. 5 shapes, 19 recorded calls. N buckets: {'medium': 1, 'small': 3, 'xs': 1}; K buckets: {'xs': 5}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `local_m64_n48_wm0_wn48_k64_s7_sl1_b1_c0_nn_llrldf_o0_u0_t0` | 5/5 | 124.6% | 122.0% | 115.4% | 0 |
| 2 | `narrow_m64_n48_k64_s7` | 5/5 | 121.4% | 119.5% | 118.1% | 0 |
| 3 | `gap_m64_n56_k64_s6` | 5/5 | 118.7% | 117.3% | 116.5% | 0 |

Seed details:

- `local_m64_n48_wm0_wn48_k64_s7_sl1_b1_c0_nn_llrldf_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=7, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `narrow_m64_n48_k64_s7`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=7, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m64_n56_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=56, warpgroup_m_tile=None, warpgroup_n_tile=56, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

## float16/canonical_nt/small/tall/k128

dtype=float16, layout=canonical_nt, m_bucket=small, aspect=tall, alignment=k128. 1 shapes, 1 recorded calls. N buckets: {'xs': 1}; K buckets: {'small': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `local_m64_n48_wm0_wn48_k128_s4_sl1_b1_c0_nn_lfrfdl_o0_u0_t0` | 1/1 | 113.2% | 113.2% | 113.2% | 0 |
| 2 | `gap_m64_n48_k128_s4` | 1/1 | 113.0% | 113.0% | 113.0% | 0 |
| 3 | `local_m64_n48_wm0_wn48_k128_s4_sl1_b1_c0_nn_llrfdf_o0_u0_t0` | 1/1 | 112.9% | 112.9% | 112.9% | 0 |

Seed details:

- `local_m64_n48_wm0_wn48_k128_s4_sl1_b1_c0_nn_lfrfdl_o0_u0_t0` (wins at least one shape): output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_last', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m64_n48_k128_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m64_n48_wm0_wn48_k128_s4_sl1_b1_c0_nn_llrfdf_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

## float16/canonical_nt/small/wide/k128

dtype=float16, layout=canonical_nt, m_bucket=small, aspect=wide, alignment=k128. 19 shapes, 708 recorded calls. N buckets: {'large': 4, 'medium': 12, 'xl': 3}; K buckets: {'large': 6, 'medium': 8, 'small': 2, 'xl': 3}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m64_n72_k128_s3` | 19/19 | 79.9% | 79.4% | 61.6% | 0 |
| 2 | `narrow_m128_n48_k64_s4` | 19/19 | 77.3% | 66.4% | 50.9% | 0 |
| 3 | `gap_m64_n48_k128_s3` | 19/19 | 73.7% | 57.9% | 48.4% | 0 |

Seed details:

- `gap_m64_n72_k128_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=72, warpgroup_m_tile=None, warpgroup_n_tile=72, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `narrow_m128_n48_k64_s4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m64_n48_k128_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_scalar_store` | ok | 69.46% |
| `probe_dst_keep` | ok | 69.45% |
| `probe_ktile128_n64_k2` | ok | 69.35% |
| `probe_m192_n64` | ok | 64.40% |
| `probe_rhs_keep` | ok | 38.72% |
| `probe_lhs_keep` | ok | 38.59% |
| `probe_lhs_promote_256b` | ok | 38.55% |
| `probe_ptx_o1` | ok | 38.55% |

## float16/canonical_nt/small/wide/k64

dtype=float16, layout=canonical_nt, m_bucket=small, aspect=wide, alignment=k64. 2 shapes, 18 recorded calls. N buckets: {'medium': 2}; K buckets: {'xl': 2}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `pipe_m128_n80_k64_s6` | 2/2 | 81.4% | 84.3% | 76.6% | 0 |
| 2 | `gap_m128_n64_block_k64_s5` | 2/2 | 81.2% | 83.2% | 77.7% | 0 |
| 3 | `xfer_m128_n64_k64_s5_c0_eca091b6eb` | 2/2 | 80.9% | 81.6% | 79.6% | 0 |

Seed details:

- `pipe_m128_n80_k64_s6`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=80, warpgroup_m_tile=None, warpgroup_n_tile=80, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m128_n64_block_k64_s5`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `xfer_m128_n64_k64_s5_c0_eca091b6eb`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis='n', output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=5, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 60.91% |
| `probe_scalar_store` | ok | 54.76% |
| `probe_m192_n64` | ok | 49.36% |
| `probe_lhs_promote_256b` | ok | 31.28% |
| `probe_rhs_keep` | ok | 31.25% |
| `probe_lhs_keep` | ok | 31.24% |
| `probe_ptx_o1` | ok | 31.24% |

## float16/canonical_nt/xl/tall/k128

dtype=float16, layout=canonical_nt, m_bucket=xl, aspect=tall, alignment=k128. 114 shapes, 1,179 recorded calls. N buckets: {'large': 4, 'medium': 52, 'small': 40, 'xs': 18}; K buckets: {'large': 12, 'medium': 39, 'small': 51, 'xs': 12}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `refine_scalar_n64_k2` | 114/114 | 50.8% | 47.3% | 17.6% | 0 |
| 2 | `refine_scalar_n64_ktile128_k2` | 114/114 | 42.9% | 40.9% | 18.0% | 0 |
| 3 | `refine_scalar_n64_k4` | 114/114 | 41.7% | 39.5% | 17.8% | 0 |

Seed details:

- `refine_scalar_n64_k2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=2, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='scalar', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `refine_scalar_n64_ktile128_k2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=2, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='scalar', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `refine_scalar_n64_k4`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='scalar', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_scalar_store` | ok | 58.86% |
| `probe_dst_keep` | ok | 55.11% |
| `probe_ktile128_n64_k2` | ok | 54.52% |
| `probe_m192_n64` | ok | 48.60% |
| `probe_lhs_keep` | ValueError: TileMatmul: op_dim M size (8448) is not divisible by tile size (2176) | - |
| `probe_lhs_promote_256b` | ValueError: TileMatmul: op_dim M size (8448) is not divisible by tile size (2176) | - |
| `probe_ptx_o1` | ValueError: TileMatmul: op_dim M size (8448) is not divisible by tile size (2176) | - |
| `probe_rhs_keep` | ValueError: TileMatmul: op_dim M size (8448) is not divisible by tile size (2176) | - |

## float16/canonical_nt/xl/tall/k64

dtype=float16, layout=canonical_nt, m_bucket=xl, aspect=tall, alignment=k64. 14 shapes, 65 recorded calls. N buckets: {'medium': 2, 'small': 9, 'xs': 3}; K buckets: {'medium': 3, 'small': 7, 'xs': 4}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `refine_m64_n128_k4_dst_keep` | 14/14 | 63.2% | 65.5% | 41.9% | 0 |
| 2 | `refine_m64_n128_k4_rhs_keep` | 14/14 | 61.9% | 63.9% | 41.3% | 0 |
| 3 | `gap_m128_n64_block_k64_s2` | 14/14 | 57.2% | 64.0% | 44.0% | 0 |

Seed details:

- `refine_m64_n128_k4_dst_keep`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_last', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `refine_m64_n128_k4_rhs_keep`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=128, warpgroup_m_tile=None, warpgroup_n_tile=128, warpgroup_fast_axis=None, cluster_m_ctas=None, cluster_n_ctas=None, operand_stages=4, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=128, output_store='stmatrix', unsafe_reuse_smem=None, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m128_n64_block_k64_s2`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=128, output_n_tile=64, warpgroup_m_tile=None, warpgroup_n_tile=64, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=2, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=64, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_lhs_keep` | ok | 95.69% |
| `probe_rhs_keep` | ok | 90.10% |
| `probe_lhs_promote_256b` | ok | 86.92% |
| `probe_ptx_o1` | ok | 83.83% |
| `probe_dst_keep` | ok | 50.50% |
| `probe_scalar_store` | ok | 50.09% |
| `probe_m192_n64` | ok | 45.53% |

## float16/canonical_nt/xs/balanced/k64

dtype=float16, layout=canonical_nt, m_bucket=xs, aspect=balanced, alignment=k64. 2 shapes, 12 recorded calls. N buckets: {'small': 1, 'xs': 1}; K buckets: {'xs': 2}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrldl_o0_u0_t0` | 2/2 | 122.9% | 122.9% | 121.6% | 0 |
| 2 | `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrldf_o0_u0_t0` | 2/2 | 122.2% | 122.2% | 122.1% | 0 |
| 3 | `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrfdl_o0_u0_t0` | 2/2 | 121.5% | 121.5% | 120.7% | 0 |

Seed details:

- `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrldl_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_last', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrldf_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_last', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `local_m64_n48_wm0_wn48_k64_s6_sl1_b1_c0_nn_llrfdl_o0_u0_t0`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=True, output_tile_fast_axis='n', output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=64, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_last', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_last', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

## float16/canonical_nt/xs/wide/k128

dtype=float16, layout=canonical_nt, m_bucket=xs, aspect=wide, alignment=k128. 10 shapes, 398 recorded calls. N buckets: {'large': 3, 'medium': 3, 'small': 2, 'xl': 2}; K buckets: {'large': 3, 'medium': 3, 'small': 3, 'xs': 1}.

| rank | seed configuration | coverage | weighted % Torch | median % Torch | worst % Torch | failures |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gap_m64_n48_k128_s3` | 10/10 | 75.7% | 101.0% | 49.4% | 0 |
| 2 | `narrow_m64_n48_k128_s6` | 10/10 | 73.1% | 98.3% | 37.2% | 0 |
| 3 | `gap_m64_n72_k128_s3` | 10/10 | 72.8% | 93.5% | 47.7% | 0 |

Seed details:

- `gap_m64_n48_k128_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `narrow_m64_n48_k128_s6` (wins at least one shape): output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=48, warpgroup_m_tile=None, warpgroup_n_tile=48, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=6, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False
- `gap_m64_n72_k128_s3`: output_tiles_per_cta=1, output_cta_count=None, unroll_output_tiles=None, output_tile_fast_axis=None, output_m_tile=64, output_n_tile=72, warpgroup_m_tile=None, warpgroup_n_tile=72, warpgroup_fast_axis='n', cluster_m_ctas=1, cluster_n_ctas=1, operand_stages=3, operand_k_tile=128, rhs_tma_load_slices=None, output_slices=1, output_store_buffers=1, output_tma_inner_size=None, output_store='stmatrix', unsafe_reuse_smem=False, lhs_l2_cache_hint='evict_first', rhs_l2_cache_hint='evict_first', dst_l2_cache_hint='evict_first', lhs_l2_promotion=None, ptx_optimization_level=None, multicast=False

Representative-shape knob probes:

| configuration | result | % Torch |
| --- | --- | ---: |
| `probe_dst_keep` | ok | 61.38% |
| `probe_ktile128_n64_k2` | ok | 50.33% |
| `probe_scalar_store` | ok | 50.30% |
| `probe_m192_n64` | ok | 44.72% |
| `probe_rhs_keep` | ok | 35.12% |
| `probe_lhs_keep` | ok | 34.92% |
| `probe_lhs_promote_256b` | ok | 34.89% |
| `probe_ptx_o1` | ok | 34.83% |
