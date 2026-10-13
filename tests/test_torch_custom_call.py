import torch

from ayla.gpu_dialect import gpu_dialect
from ayla.jit_compiler import JITCompiler
from ayla.torch_custom_call import OperandKind, make_torch_custom_call
from tests.testing_utils import requires_cuda


def _compiler(name: str) -> JITCompiler:
    return JITCompiler(gpu_dialect.parse_module(f"""
        module {{
          kernel @{name}(%a: gmem<4x8xbf16>, %b: gmem<4x16xbf16>) {{
          }}
        }}
    """))


def test_out_only_custom_call_has_meta_kernel():
    compiler = _compiler("out_only_meta_test")
    op = make_torch_custom_call(
        compiler,
        [OperandKind.R, OperandKind.RW],
        namespace="ayla_out_only_meta_test",
    )
    src = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
    out = torch.empty((4, 16), dtype=torch.bfloat16, device="meta")

    assert op(src, out) is None
    assert compiler.cache_size == 0


def test_result_shape_can_be_inferred_from_inputs():
    compiler = _compiler("dynamic_result_meta_test")
    op = make_torch_custom_call(
        compiler,
        [OperandKind.R, OperandKind.RESULT],
        namespace="ayla_dynamic_result_meta_test",
        infer_result_shapes=lambda src: [(src.shape[0], 16)],
    )
    src = torch.empty((7, 8), dtype=torch.bfloat16, device="meta")

    result = op(src)

    assert result.shape == (7, 16)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "meta"
    assert compiler.cache_size == 0


@requires_cuda
def test_jit_custom_call_runs_on_each_device_and_replays_in_a_graph():
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import F32Type, GmemTensorType, KernelOp, ModuleOp, PTXTemplateOp

    typ = GmemTensorType.build([32], F32Type())
    a, b, c = [Value(name, typ) for name in ("a", "b", "c")]
    add = PTXTemplateOp.build(
        operands=[a, b, c], result_types=[],
        operand_names=["a", "b", "c"], result_names=[],
        description="add tensors",
        template="""
            mov.u32 ${b32:idx}, %tid.x;
            mul.wide.u32 ${b64:offset}, ${b32:idx}, 4;
            add.u64 ${b64:a}, ${b64:operand.a.ptr}, ${b64:offset};
            add.u64 ${b64:b}, ${b64:operand.b.ptr}, ${b64:offset};
            add.u64 ${b64:c}, ${b64:operand.c.ptr}, ${b64:offset};
            ld.global.f32 ${f32:a_value}, [${b64:a}];
            ld.global.f32 ${f32:b_value}, [${b64:b}];
            add.f32 ${f32:c_value}, ${f32:a_value}, ${f32:b_value};
            st.global.f32 [${b64:c}], ${f32:c_value};
        """,
    )
    body = Block()
    body.append(KernelOp.build(
        "jit_custom_add_test", Block(arguments=[a, b, c], ops=[add]),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
    ))
    module = ModuleOp.build(body)
    original = repr(module)
    compiler = JITCompiler(module)
    op = make_torch_custom_call(
        compiler, [OperandKind.R, OperandKind.R, OperandKind.RESULT],
        namespace="ayla_jit_custom_call_test",
    )
    assert compiler.cache_size == 0
    for device in range(min(2, torch.cuda.device_count())):
        src = torch.arange(32, dtype=torch.float32, device=f"cuda:{device}")
        other = src * 2
        # Dispatch selects the tensor's device even if another device is current.
        torch.testing.assert_close(op(src, other), src + other)
        assert compiler.cache_size == device + 1

        with torch.cuda.device(src.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = op(src, other)
            graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(output, src + other)
    assert repr(module) == original
