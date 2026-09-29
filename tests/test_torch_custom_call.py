import torch

from ayla.base.ir import Type
from ayla.gpu_dialect import BF16Type, GmemTensorType
from ayla.torch_custom_call import OperandKind, make_torch_custom_call


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
