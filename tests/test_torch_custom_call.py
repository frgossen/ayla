from contextlib import nullcontext

import pytest
import torch

from ayla.base.ir import Type
from ayla.gpu_dialect import BF16Type, GmemTensorType
from ayla.jit_compiler import JITCompiler
from ayla.torch_custom_call import (
    OperandKind,
    make_torch_custom_call,
)


def test_out_only_custom_call_has_meta_kernel():
    class Launcher:
        name = "out_only_meta_test"
        signature: list[Type] = [
            GmemTensorType.build([4, 8], BF16Type()),
            GmemTensorType.build([4, 16], BF16Type()),
        ]

        def __call__(self, *args):
            raise AssertionError("Meta dispatch must not launch the kernel")

    op = make_torch_custom_call(
        Launcher(),
        [OperandKind.R, OperandKind.RW],
        namespace="ayla_out_only_meta_test",
    )
    src = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
    out = torch.empty((4, 16), dtype=torch.bfloat16, device="meta")

    assert op(src, out) is None


def test_result_shape_can_be_inferred_from_inputs():
    class Launcher:
        name = "dynamic_result_meta_test"
        signature: list[Type] = [
            GmemTensorType.build([1, 8], BF16Type()),
            GmemTensorType.build([1, 16], BF16Type()),
        ]

        def __call__(self, *args):
            raise AssertionError("Meta dispatch must not launch the kernel")

    op = make_torch_custom_call(
        Launcher(),
        [OperandKind.R, OperandKind.RESULT],
        namespace="ayla_dynamic_result_meta_test",
        infer_result_shapes=lambda src: [(src.shape[0], 16)],
    )
    src = torch.empty((7, 8), dtype=torch.bfloat16, device="meta")

    result = op(src)

    assert result.shape == (7, 16)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "meta"


def test_jit_compiler_dispatches_by_tensor_signature(monkeypatch):
    built = []
    modules = {}

    def compile_module(module):
        rows = modules[id(module)]
        built.append(rows)

        def build_launcher():
            def launch(src, dst):
                dst.value = src.value + rows

            return launch

        return build_launcher

    class Tensor:
        device = torch.device("cuda:0")

        def __init__(self, rows, value=1):
            self.shape = (rows, 8)
            self.dtype = torch.bfloat16
            self.value = value

        def stride(self):
            return (8, 1)

    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    compiler = JITCompiler(
        name="shape_dispatch_test",
        signature=[
            GmemTensorType.build([None, 8], BF16Type()),
            GmemTensorType.build([None, 8], BF16Type()),
        ],
        compile_fn=compile_module,
    )
    for rows in (4, 7):
        signature = (
            ((rows, 8), (8, 1), torch.bfloat16),
            ((rows, 8), (8, 1), torch.bfloat16),
        )
        module = object()
        modules[id(module)] = rows
        compiler.prepare_module(signature, module)  # type: ignore[arg-type]
    for rows in (4, 4, 7):
        src = Tensor(rows)
        dst = Tensor(rows, value=None)
        compiler(src, dst)  # type: ignore[arg-type]
        assert dst.value == 1 + rows

    assert built == [4, 7]


def test_jit_compiler_rejects_cpu_tensors():
    compiler = JITCompiler(
        name="cuda_only_dispatch_test",
        signature=[],
        compile_fn=lambda _module: lambda: lambda *_tensors: None,
    )

    with pytest.raises(ValueError, match="expected CUDA tensors"):
        compiler(torch.empty(1))
