import torch

from ayla.gpu_dialect import gpu_dialect, ModuleOp
from ayla.jit_compiler import JITCompiler
from ayla.torch_custom_call import (
    OperandKind,
    make_torch_custom_call,
)


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


def _compiler(name: str) -> JITCompiler:
    module = gpu_dialect.parse(
        f"module {{ kernel @{name}(%a: gmem<?x8xbf16>, %b: gmem<?x16xbf16>) {{}} }}"
    )
    assert isinstance(module, ModuleOp)
    return JITCompiler(module)
