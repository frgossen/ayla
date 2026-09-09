"""A Muse Glimmer 30B decoder block body captured from vLLM."""
# ruff: noqa: E702

from __future__ import annotations

import torch
from torch import device

from benchmarks.common import BenchConfig

def _block_body(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1):
    view = torch.ops.aten.view.default(arg0_1, [-1, 4096]);  arg0_1 = None
    permute = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    mm = torch.ops.aten.mm.default(arg2_1, permute);  arg2_1 = permute = None
    sigmoid = torch.ops.aten.sigmoid.default(mm);  mm = None
    mul = torch.ops.aten.mul.Tensor(sigmoid, view);  sigmoid = view = None
    permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    mm_1 = torch.ops.aten.mm.default(mul, permute_1);  mul = permute_1 = None
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add = torch.ops.aten.add.Tensor(mean, 1e-08);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
    mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt);  convert_element_type_4 = rsqrt = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
    add_1 = torch.ops.aten.add.Tensor(convert_element_type_5, 1);  convert_element_type_5 = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
    add_2 = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_6);  arg6_1 = convert_element_type_6 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 2)
    mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    add_3 = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1);  convert_element_type_7 = rsqrt_1 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
    add_4 = torch.ops.aten.add.Tensor(convert_element_type_8, 1);  convert_element_type_8 = None
    mul_4 = torch.ops.aten.mul.Tensor(mul_3, add_4);  mul_3 = add_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
    permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    mm_2 = torch.ops.aten.mm.default(convert_element_type_9, permute_2);  convert_element_type_9 = permute_2 = None
    slice_1 = torch.ops.aten.slice.Tensor(mm_2, 1, 0, 19968)
    convert_element_type_12 = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
    neg = torch.ops.aten.neg.default(convert_element_type_12)
    exp = torch.ops.aten.exp.default(neg);  neg = None
    add_5 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
    div = torch.ops.aten.div.Tensor(convert_element_type_12, add_5);  convert_element_type_12 = add_5 = None
    convert_element_type_13 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
    slice_2 = torch.ops.aten.slice.Tensor(mm_2, 1, 19968, 9223372036854775807);  mm_2 = None
    mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_13, slice_2);  convert_element_type_13 = slice_2 = None
    permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    mm_3 = torch.ops.aten.mm.default(mul_5, permute_3);  mul_5 = permute_3 = None
    convert_element_type_16 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_16, 2)
    mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    add_6 = torch.ops.aten.add.Tensor(mean_2, 1e-08);  mean_2 = None
    rsqrt_2 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_16, rsqrt_2);  convert_element_type_16 = rsqrt_2 = None
    convert_element_type_17 = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
    add_7 = torch.ops.aten.add.Tensor(convert_element_type_17, 1);  convert_element_type_17 = None
    mul_7 = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    convert_element_type_18 = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
    add_8 = torch.ops.aten.add.Tensor(add_2, convert_element_type_18);  add_2 = convert_element_type_18 = None
    convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
    pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2)
    mean_3 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    add_9 = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
    rsqrt_3 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_3);  convert_element_type_19 = rsqrt_3 = None
    convert_element_type_20 = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
    add_10 = torch.ops.aten.add.Tensor(convert_element_type_20, 1);  convert_element_type_20 = None
    mul_9 = torch.ops.aten.mul.Tensor(mul_8, add_10);  mul_8 = add_10 = None
    convert_element_type_21 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
    permute_4 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    mm_4 = torch.ops.aten.mm.default(convert_element_type_21, permute_4);  permute_4 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(mm_4, [4096, 256, 256], -1);  mm_4 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    mul_10 = arg1_1 * 4096
    floordiv = mul_10 // 128;  mul_10 = None
    clone = torch.ops.aten.clone.default(getitem, memory_format=torch.contiguous_format);  getitem = None
    view_1 = torch.ops.aten.view.default(clone, [floordiv, 128]);  clone = floordiv = None
    convert_element_type_24 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
    pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_24, 2)
    mean_4 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    add_11 = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
    rsqrt_4 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt_4);  convert_element_type_24 = rsqrt_4 = None
    convert_element_type_25 = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
    view_2 = torch.ops.aten.view.default(convert_element_type_25, [-1, 4096]);  convert_element_type_25 = None
    mul_23 = torch.ops.aten.mul.Tensor(view_2, 3.87);  view_2 = None
    mul_24 = arg1_1 * 256
    floordiv_1 = mul_24 // 128;  mul_24 = None
    clone_1 = torch.ops.aten.clone.default(getitem_1, memory_format=torch.contiguous_format);  getitem_1 = None
    view_3 = torch.ops.aten.view.default(clone_1, [floordiv_1, 128]);  clone_1 = floordiv_1 = None
    convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
    pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_26, 2)
    mean_5 = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    add_12 = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
    rsqrt_5 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_26, rsqrt_5);  convert_element_type_26 = rsqrt_5 = None
    convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
    view_4 = torch.ops.aten.view.default(convert_element_type_27, [-1, 256]);  convert_element_type_27 = None
    index = torch.ops.aten.index.Tensor(arg14_1, [arg13_1]);  arg14_1 = arg13_1 = None
    split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
    getitem_3 = split[0]
    getitem_4 = split[1];  split = None
    view_5 = torch.ops.aten.view.default(mul_23, [arg1_1, -1, 128]);  mul_23 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(getitem_3, -2)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(getitem_4, -2)
    split_1 = torch.ops.aten.split.Tensor(view_5, 64, -1);  view_5 = None
    getitem_5 = split_1[0]
    getitem_6 = split_1[1];  split_1 = None
    mul_39 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze)
    mul_40 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_1)
    sub = torch.ops.aten.sub.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    mul_41 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze);  getitem_6 = unsqueeze = None
    mul_42 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_1);  getitem_5 = unsqueeze_1 = None
    add_13 = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    cat = torch.ops.aten.cat.default([sub, add_13], -1);  sub = add_13 = None
    view_7 = torch.ops.aten.view.default(view_4, [arg1_1, -1, 128]);  view_4 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_3, -2);  getitem_3 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_4, -2);  getitem_4 = None
    split_2 = torch.ops.aten.split.Tensor(view_7, 64, -1);  view_7 = None
    getitem_7 = split_2[0]
    getitem_8 = split_2[1];  split_2 = None
    mul_47 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_2)
    mul_48 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_3)
    sub_1 = torch.ops.aten.sub.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    mul_49 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_2);  getitem_8 = unsqueeze_2 = None
    mul_50 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_3);  getitem_7 = unsqueeze_3 = None
    add_14 = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    cat_1 = torch.ops.aten.cat.default([sub_1, add_14], -1);  sub_1 = add_14 = None
    empty = torch.ops.aten.empty.memory_format([arg1_1, 4096], dtype=torch.bfloat16, device=device(type="cuda", index=0), pin_memory=False);  arg1_1 = None
    view_10 = torch.ops.aten.view.default(empty, [-1, 32, 128]);  empty = None
    view_12 = torch.ops.aten.view.default(getitem_2, [-1, 2, 128]);  getitem_2 = None
    return (cat_1, view_12, cat, view_10, convert_element_type_21, add_8)


_DECODE_M = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_PREFILL_M = (512, 2048, 4096, 8192)


def _build_config(num_tokens: int, phase: str) -> BenchConfig:
    def make_inputs() -> list[torch.Tensor | int]:
        torch.manual_seed(0)
        return [
            torch.randn(
                num_tokens, 32, 128, dtype=torch.bfloat16, device="cuda",
            ),
            num_tokens,
            torch.randn(
                num_tokens, 6656, dtype=torch.bfloat16, device="cuda",
            ),
            torch.randn(4096, 6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(6656, 4096, dtype=torch.bfloat16, device="cuda"),
            torch.randn(6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(
                num_tokens, 6656, dtype=torch.bfloat16, device="cuda",
            ),
            torch.randn(6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(39936, 6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(6656, 19968, dtype=torch.bfloat16, device="cuda"),
            torch.randn(6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(6656, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4608, 6656, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
            torch.randn(131072, 128, dtype=torch.bfloat16, device="cuda"),
        ]

    return BenchConfig(
        name=f"muse_glimmer_block {phase} M={num_tokens} bf16",
        make_inputs=make_inputs,
        compute=_block_body,
    )


def build_configs() -> list[BenchConfig]:
    """What this module contributes to the corpus."""
    return [
        *(_build_config(m, "decode") for m in _DECODE_M),
        *(_build_config(m, "prefill") for m in _PREFILL_M),
    ]
