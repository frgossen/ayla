"""Tests for the DCE pass."""
import textwrap

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.passes.dce import dce


def test_dce_removes_unused_index_ops():
    """Dead index ops are removed, used ones are kept.

    The whole dead chain goes in one pass: %also_dead is unread, and dropping
    it leaves %dead unread too.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index = block_id.x
            %cy: index = block_id.y
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %dead: index = index_constant() attrs={value: 999}
            %also_dead: index = index_mul(%cy: index, %dead: index)
            %val: rmem<1xf32> = alloc() attrs={id: 0}
            move(%val: rmem<1xf32>, %a: gmem<4096x4096xf32>, %cx_offset: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    assert isinstance(module, ModuleOp)
    module = dce(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index = block_id.x
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %val: rmem<1xf32> = alloc() attrs={id: 0}
            move(%val: rmem<1xf32>, %a: gmem<4096x4096xf32>, %cx_offset: index)
          }
        }""")


def test_dce_removes_empty_async_and_await():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test() {
            %lock: lock<mbar> = alloc()
            async(%lock: lock<mbar>) collective=xyz.block {
            }
            await(%lock: lock<mbar>)
          }
        }"""))

    dce(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test() {
          }
        }""")


def test_dce_keeps_async_and_await_with_work():
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%src: gmem<1xf32>, %dst: smem<1xf32>) {
            %lock: lock<mbar> = alloc()
            async(%lock: lock<mbar>) collective=xyz.block move(%dst: smem<1xf32>, %src: gmem<1xf32>) collective=xyz.block attrs={tma: True}
            await(%lock: lock<mbar>)
          }
        }""")
    module = gpu_dialect.parse_module(module_str)
    expected = repr(module)

    dce(module)

    assert repr(module) == expected
