import textwrap

from ayla.gpu_dialect import AllocOp, ElementwiseOp, MatmulOp, gpu_dialect
from ayla.passes.reuse_buffers import reuse_buffers


def test_reuses_output_buffer_for_matmul_temporary():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %tmp: gmem<16x16xf32> = alloc() attrs={id: 0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise sigmoid(%output: gmem<16x16xf32>, %tmp: gmem<16x16xf32>)
          }
        }"""))

    reuse_buffers(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise sigmoid(%output: gmem<16x16xf32>, %output: gmem<16x16xf32>)
          }
        }""")


def test_reuses_output_buffer_between_matmuls():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @two_matmuls(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %tmp: gmem<16x16xf32> = alloc() attrs={id: 0}
            %first_acc: smem<16x16xf32> = alloc() attrs={id: 1, init: 0.0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %first_acc: smem<16x16xf32>) collective=xyz.grid
            move(%tmp: gmem<16x16xf32>, %first_acc: smem<16x16xf32>) attrs={tma: True}
            %second_lhs: smem<16x16xf32> = alloc() attrs={id: 2, init: 0.0}
            move(%second_lhs: smem<16x16xf32>, %tmp: gmem<16x16xf32>) attrs={tma: True}
            %second_acc: smem<16x16xf32> = alloc() attrs={id: 3, init: 0.0}
            matmul(%second_lhs: smem<16x16xf32>, %rhs: gmem<16x16xf32>, %second_acc: smem<16x16xf32>) collective=xyz.grid
            move(%output: gmem<16x16xf32>, %second_acc: smem<16x16xf32>) attrs={tma: True}
          }
        }"""))

    reuse_buffers(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @two_matmuls(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %first_acc: smem<16x16xf32> = alloc() attrs={id: 1, init: 0.0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %first_acc: smem<16x16xf32>) collective=xyz.grid
            move(%output: gmem<16x16xf32>, %first_acc: smem<16x16xf32>) attrs={tma: True}
            %second_lhs: smem<16x16xf32> = alloc() attrs={id: 2, init: 0.0}
            move(%second_lhs: smem<16x16xf32>, %output: gmem<16x16xf32>) attrs={tma: True}
            %second_acc: smem<16x16xf32> = alloc() attrs={id: 3, init: 0.0}
            matmul(%second_lhs: smem<16x16xf32>, %rhs: gmem<16x16xf32>, %second_acc: smem<16x16xf32>) collective=xyz.grid
            move(%output: gmem<16x16xf32>, %second_acc: smem<16x16xf32>) attrs={tma: True}
          }
        }""")


def test_reuses_one_buffer_for_elementwise_chain():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @elementwise_chain(%src: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %a: gmem<16x16xf32> = alloc() attrs={id: 0}
            %b: gmem<16x16xf32> = alloc() attrs={id: 1}
            %c: gmem<16x16xf32> = alloc() attrs={id: 2}
            elementwise relu(%a: gmem<16x16xf32>, %src: gmem<16x16xf32>)
            elementwise exp(%b: gmem<16x16xf32>, %a: gmem<16x16xf32>)
            elementwise tanh(%c: gmem<16x16xf32>, %b: gmem<16x16xf32>)
            elementwise sigmoid(%output: gmem<16x16xf32>, %c: gmem<16x16xf32>)
          }
        }"""))

    reuse_buffers(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @elementwise_chain(%src: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            elementwise relu(%output: gmem<16x16xf32>, %src: gmem<16x16xf32>)
            elementwise exp(%output: gmem<16x16xf32>, %output: gmem<16x16xf32>)
            elementwise tanh(%output: gmem<16x16xf32>, %output: gmem<16x16xf32>)
            elementwise sigmoid(%output: gmem<16x16xf32>, %output: gmem<16x16xf32>)
          }
        }""")


def test_does_not_reuse_buffer_that_remains_live_after_handoff():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %tmp: gmem<16x16xf32> = alloc() attrs={id: 0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %tmp: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise sigmoid(%output: gmem<16x16xf32>, %tmp: gmem<16x16xf32>)
            move(%tmp: gmem<16x16xf32>, %tmp: gmem<16x16xf32>)
          }
        }"""))

    reuse_buffers(module)

    assert any(isinstance(op, AllocOp) for op in module.body.ops[0].body.ops)
    [elementwise] = [
        op for op in module.body.ops[0].body.ops
        if isinstance(op, ElementwiseOp)
    ]
    assert elementwise.src(0) is not elementwise.dst
    assert any(isinstance(op, MatmulOp) for op in module.body.ops[0].body.ops)


def test_reuse_union_includes_lifetimes_of_previous_members():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %output: gmem<16x16xf32>) {
            %a: gmem<16x16xf32> = alloc() attrs={id: 0, label: "a"}
            %b: gmem<16x16xf32> = alloc() attrs={id: 1, label: "b"}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
            elementwise sigmoid(%b: gmem<16x16xf32>, %b: gmem<16x16xf32>)
            elementwise sigmoid(%output: gmem<16x16xf32>, %a: gmem<16x16xf32>)
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
            elementwise sigmoid(%a: gmem<16x16xf32>, %a: gmem<16x16xf32>)
            matmul(%lhs: gmem<16x16xf32>, %rhs: gmem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.grid attrs={init: 0.0}
          }
        }"""))

    reuse_buffers(module)

    kernel = module.body.ops[0]
    allocations = [op for op in kernel.body.ops if isinstance(op, AllocOp)]
    matmuls = [op for op in kernel.body.ops if isinstance(op, MatmulOp)]
    assert len(allocations) == 1
    assert matmuls[0].dst is matmuls[1].dst


def test_does_not_reuse_async_locks():
    module_str = textwrap.dedent("""\
        module {
          kernel @two_tma_loads(%a: gmem<64x64xf32>, %b: gmem<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %b_desc: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>) {
            %a_smem: smem<64x64xf32> = alloc() attrs={id: 0}
            %b_smem: smem<64x64xf32> = alloc() attrs={id: 1}
            %a_lock: lock<mbar> = alloc() attrs={id: 2}
            %b_lock: lock<mbar> = alloc() attrs={id: 3}
            async(%a_lock: lock<mbar>) tma_move(%a_smem: smem<64x64xf32>, %a: gmem<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>)
            async(%b_lock: lock<mbar>) tma_move(%b_smem: smem<64x64xf32>, %b: gmem<64x64xf32>, %b_desc: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>)
            await(%a_lock: lock<mbar>)
            await(%b_lock: lock<mbar>)
          }
        }""")
    module = gpu_dialect.parse(module_str)

    reuse_buffers(module)

    assert repr(module) == module_str
