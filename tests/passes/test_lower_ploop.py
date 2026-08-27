"""Tests for the lower_ploop pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.canonicalize import canonicalize
from ayla.passes.lower_ploop import lower_ploop


def test_lower_ploop():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%c: gmem<4096x4096xf32>) {
            ploop block.x (%bx: index = 8 to 64 step 4) {
              ploop wg.x (%w: index = 0 to 2) {
                %t: gmem<128x128xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %bx: index, %w: index)
                %s: smem<128x128xf32> = alloc() attrs={id: 0}
                move(%s: smem<128x128xf32>, %t: gmem<128x128xf32 strides=[4096, 1]>)
              }
            }
          }
        }"""))  # noqa: E501

    # Every ploop leaves `id * step + lb` and its own id op behind, whatever
    # the range, so canonicalize to see past the identities and duplicates.
    expected = textwrap.dedent("""\
        module {
          kernel @k(%c: gmem<4096x4096xf32>) grid_dim=[14, None, None] block_dim=[256, 1, 1] num_warp_groups=2 {
            %bx: index, %by: index, %bz: index = block_ids()
            %c4: index = index_constant() attrs={value: 4}
            %0: index = index_mul(%bx: index, %c4: index)
            %c8: index = index_constant() attrs={value: 8}
            %1: index = index_add(%0: index, %c8: index)
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %t: gmem<128x128xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %1: index, %wgx: index)
            %s: smem<128x128xf32> = alloc() attrs={id: 0}
            move(%s: smem<128x128xf32>, %t: gmem<128x128xf32 strides=[4096, 1]>)
          }
        }""")  # noqa: E501
    assert repr(canonicalize(lower_ploop(module))) == expected
