import textwrap

from ayla.gpu_dialect import AllocOp, ElementwiseOp, MatmulOp, gpu_dialect
from ayla.passes.reuse_buffers import reuse_buffers


def test_reuses_output_buffer_for_matmul_temporary():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %tmp: gmem<16x16xf32> = alloc() attrs={id: 0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise(%output: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) attrs={fn: "sigmoid"}
          }
        }"""))

    reuse_buffers(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise(%output: gmem<16x16xf32>, %output: gmem<16x16xf32>) attrs={fn: "sigmoid"}
          }
        }""")


def test_does_not_reuse_buffer_that_remains_live_after_handoff():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %tmp: gmem<16x16xf32> = alloc() attrs={id: 0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise(%output: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) attrs={fn: "sigmoid"}
            move(%tmp: gmem<16x16xf32>, %tmp: gmem<16x16xf32>)
          }
        }"""))

    reuse_buffers(module)

    assert any(isinstance(op, AllocOp) for op in module.body.ops[0].body.ops)
    [elementwise] = [
        op for op in module.body.ops[0].body.ops
        if isinstance(op, ElementwiseOp)
    ]
    assert elementwise.src is not elementwise.dst
    assert any(isinstance(op, MatmulOp) for op in module.body.ops[0].body.ops)


def test_reuse_union_includes_lifetimes_of_previous_members():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %a: gmem<16x16xf32> = alloc() attrs={id: 0, label: "a"}
            %b: gmem<16x16xf32> = alloc() attrs={id: 1, label: "b"}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise(%b: gmem<16x16xf32>, %b: gmem<16x16xf32>) attrs={fn: "sigmoid"}
            elementwise(%output: gmem<16x16xf32>, %a: gmem<16x16xf32>) attrs={fn: "sigmoid"}
          }
        }"""))

    reuse_buffers(module)

    allocations = [
        op for op in module.body.ops[0].body.ops if isinstance(op, AllocOp)
    ]
    assert [op.attrs["label"] for op in allocations] == ["b"]


def test_allocation_starts_a_reuse_set():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>) {
            %a: gmem<16x16xf32> = alloc() attrs={id: 0}
            %b: gmem<16x16xf32> = alloc() attrs={id: 1}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise(%a: gmem<16x16xf32>, %a: gmem<16x16xf32>) attrs={fn: "sigmoid"}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
          }
        }"""))

    reuse_buffers(module)

    kernel = module.body.ops[0]
    allocations = [op for op in kernel.body.ops if isinstance(op, AllocOp)]
    matmuls = [op for op in kernel.body.ops if isinstance(op, MatmulOp)]
    assert len(allocations) == 1
    assert matmuls[0].dst is matmuls[1].dst
