"""Tests for the loop_peeling pass."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import ForOp, gpu_dialect
from ayla.passes.loop_peeling import loop_peeling


def test_peels_first_iteration_and_advances_loop():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<8xf32>) {
            for (%k: index = 2 to 8 step 2) attrs={label: "k"} {
              %t: gmem<1xf32> = tile(%a: gmem<8xf32>, %k: index)
              move(%a: gmem<8xf32>, %t: gmem<1xf32>) attrs={label: "copy"}
            }
          }
        }"""))

    [loop] = find_ops(module, lambda op: isinstance(op, ForOp))
    module = loop_peeling(module, loops={loop})

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<8xf32>) {
            %c2: index = index_constant() attrs={value: 2}
            %t: gmem<1xf32> = tile(%a: gmem<8xf32>, %c2: index)
            move(%a: gmem<8xf32>, %t: gmem<1xf32>) attrs={label: "copy"}
            for (%k: index = 4 to 8 step 2) attrs={label: "k"} {
              %t1: gmem<1xf32> = tile(%a: gmem<8xf32>, %k: index)
              move(%a: gmem<8xf32>, %t1: gmem<1xf32>) attrs={label: "copy"}
            }
          }
        }""")


def test_single_iteration_has_no_residual_loop():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() {
            for (%k: index = 3 to 4) attrs={label: "k"} {
              index_add(%k: index, %k: index)
            }
          }
        }"""))

    [loop] = find_ops(module, lambda op: isinstance(op, ForOp))
    module = loop_peeling(module, loops={loop})

    assert "for (" not in repr(module)
    assert "index_constant() attrs={value: 3}" in repr(module)
