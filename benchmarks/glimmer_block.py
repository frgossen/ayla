"""Standalone perf repro for one Muse Glimmer 30B decoder block body.

Self-contained: needs only torch, no vllm import and no sibling modules, so it
can be copied anywhere. It does need a CUDA device and ~1 GB of weights.

PROVENANCE
    Everything from `s59 = 8192` down to `mod = Repro()` is a verbatim tlparse
    dump -- do not hand-edit it; regenerate instead. It came from:

        cd ../vllm
        TORCH_TRACE=<dir> python dbg/run_model.py \\
            --prefill 512 --batch 2 --decode 12
        tlparse <dir>/*.log -o <out>
        cp <out>/-_-_-_-/fx_graph_runnable_10.txt \\
            ../pytorch/dbg/repro_glimmer_body.py

    `fx_graph_runnable_10` is the graph with 15 args and 5 `aten.mm` calls: the
    RoPE block body, which serves ~39 of the model's 52 layers. Its five GEMMs
    are gate (6656->4096), o_proj (4096->6656), gate_up (6656->39936),
    down (19968->6656) and qkv (6656->4608). Attention is absent because vLLM's
    piecewise compile splits the graph at the attention op, so this needs no KV
    cache and no paged-attention metadata.

    Prefer `_10` over its identical twin `_40`: `_10` keeps the input allocation
    symbolic in s59, whereas `_40` hardcodes it (109051904 = 13312 x 8192) and
    would ignore --tokens.

MODIFICATIONS to the dump (all others are hand-added below it)
    - dropped `os.environ` pins for TORCH_TRACE / TORCHINDUCTOR_CACHE_DIR /
      TRITON_CACHE_DIR, which pointed at the originating vLLM run's 19 GB cache
    - `reader.storage(None, 65536, ... torch.int64)` -> `8*s59`; that buffer had
      its size baked for s59=8192 and would not scale with --tokens

LIMITATIONS
    This always compiles with M specialized to --tokens (zero symbols in the
    graph), whereas vLLM keeps M dynamic. Dynamic mode was tried and dropped:

      - `dynamic=True` makes *every* dim symbolic, including the hidden size, so
        K and N of all five GEMMs become symbolic. vLLM never does that.
      - Marking only dim 0 of the token-shaped inputs raises
        ConstraintViolationError: arg1_1 is the Sym(s59) input and _Reader can
        only supply a plain int, which Dynamo specializes on, contradicting the
        mark. The symbol has to originate upstream in the outer module's trace,
        which a standalone post-AOT submodule cannot recreate.

    It costs little: static vs dynamic measured within ~2% (523.5 vs 535.8 us at
    M=2), because every GEMM leaves as extern_kernels.mm and cuBLAS is handed
    the real M at call time either way.

    The config block from the dump also pins vLLM's inductor settings
    (assume_static_by_default, automatic_dynamic_shapes=False, graph_partition,
    triton.autotune_at_compile_time). Good for matching vLLM, but this is not a
    neutral stock-PyTorch baseline -- strip that block if you want one.

USAGE
    python dbg/repro_glimmer_body.py --tokens 2
    python dbg/repro_glimmer_body.py --tokens 256 --max-autotune
    python dbg/repro_glimmer_body.py --tokens 2 --profile /tmp/p --share
    python dbg/repro_glimmer_body.py --tokens 2 --custom-gemm --profile /tmp/p
"""

s59 = 8192


import os
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TRITON_CACHE_AUTOTUNING'] = '1'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 16
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.enable_cpp_symbolic_shape_guards = False
torch._dynamo.config.nested_graph_breaks = False
torch._dynamo.config.enable_aot_compile = True
torch._inductor.config.enable_auto_functionalized_v2 = False
torch._inductor.config.graph_partition = True
torch._inductor.config.deterministic = False
torch._inductor.config.combo_kernels = True
torch._inductor.config.benchmark_combo_kernel = True
torch._inductor.config.assume_32bit_indexing = False
torch._inductor.config.triton.autotune_at_compile_time = True
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'cutlass.cutlass_dir', 'xpu.cutlass_dir', 'worker_start_method', 'compile_threads', 'compile_worker_watchdog_interval_seconds', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass', '_fuse_ddp_communication_passes', '_pre_fusion_custom_pass', 'cudagraph_policy', 'force_disable_cudagraph_TESTING_ONLY', 'pre_grad_pass_timing', 'fx_graph_cache', 'fx_graph_remote_cache', 'autotune_local_cache', 'autotune_remote_cache', 'pre_grad_custom_pass']
torch._inductor.config.triton.store_cubin = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.bundled_autograd_cache = True
torch._functorch.config.bypass_autograd_cache_key = True
torch._functorch.config.autograd_cache_normalize_inputs = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.force_non_lazy_backward_lowering = True
torch._functorch.config.strict_autograd_cache = True
torch._functorch.config.force_autograd_cache = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.15.0a0+gitfffac88
# torch cuda version: 13.0
# torch git version: fffac8868260473ca9a496b802835ec5cbda2692


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025 
# Cuda compilation tools, release 13.0, V13.0.88 
# Build cuda_13.0.r13.0/compiler.36424714_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1):
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
        clone = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format);  getitem = None
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
        clone_1 = torch.ops.aten.clone.default(getitem_1, memory_format = torch.contiguous_format);  getitem_1 = None
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
        empty = torch.ops.aten.empty.memory_format([arg1_1, 4096], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False);  arg1_1 = None
        view_10 = torch.ops.aten.view.default(empty, [-1, 32, 128]);  empty = None
        view_12 = torch.ops.aten.view.default(getitem_2, [-1, 2, 128]);  getitem_2 = None
        return (cat_1, view_12, cat, view_10, convert_element_type_21, add_8)

def load_args(reader):
    buf0 = reader.storage(None, 8192*s59, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (s59, 32, 128), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    reader.symint(8192, expr='s59')  # arg1_1
    buf1 = reader.storage(None, 13312*s59, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (s59, 6656), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf2 = reader.storage(None, 54525952, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (4096, 6656), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 54525952, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (6656, 4096), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf4 = reader.storage(None, 13312, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (6656,), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf5 = reader.storage(None, 13312*s59, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (s59, 6656), dtype=torch.bfloat16, is_leaf=True)  # arg6_1
    buf6 = reader.storage(None, 13312, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (6656,), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf7 = reader.storage(None, 531628032, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (39936, 6656), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf8 = reader.storage(None, 265814016, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (6656, 19968), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf9 = reader.storage(None, 13312, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (6656,), dtype=torch.bfloat16, is_leaf=True)  # arg10_1
    buf10 = reader.storage(None, 13312, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (6656,), dtype=torch.bfloat16, is_leaf=True)  # arg11_1
    buf11 = reader.storage(None, 61341696, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (4608, 6656), dtype=torch.bfloat16, is_leaf=True)  # arg12_1
    buf12 = reader.storage(None, 8*s59, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf12, (s59,), dtype=torch.int64, is_leaf=True)  # arg13_1
    buf13 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (131072, 128), dtype=torch.bfloat16, is_leaf=True)  # arg14_1
load_args._version = 0
mod = Repro()


# ---------------------------------------------------------------------------
# Harness (hand-added; everything above is the verbatim tlparse dump).
# ---------------------------------------------------------------------------

# The five (K, N) pairs this block multiplies. Weight bytes dominate a decode
# step, so this is also the block's traffic floor.
GEMM_SHAPES = (
    ("gate", 6656, 4096),
    ("o_proj", 4096, 6656),
    ("gate_up", 6656, 39936),
    ("down", 19968, 6656),
    ("qkv", 6656, 4608),
)

WEIGHT_BYTES = sum(k * n * 2 for _, k, n in GEMM_SHAPES)


class _Reader:
    """Stand-in for the repro reader, allocating fresh tensors."""

    def __init__(self) -> None:
        self.args: list = []

    def storage(self, _hash, _nbytes, device=None, dtype_hint=None):
        return device, dtype_hint

    def symint(self, _hint, expr=None):
        # arg1_1 is the Sym(s59) input; pass the size we want, not the 8192
        # hint baked into the dump.
        self.args.append(s59)
        return s59

    def tensor(self, storage, shape, dtype=None, **_kwargs):
        device, dtype_hint = storage
        dtype = dtype or dtype_hint
        if dtype.is_floating_point:
            t = torch.randn(shape, dtype=dtype, device=device)
        else:
            # arg13_1 indexes the rope table; zeros is in range.
            t = torch.zeros(shape, dtype=dtype, device=device)
        self.args.append(t)
        return t


def make_inputs(tokens: int) -> list:
    """Fresh inputs for a block with `tokens` rows.

    In decode that is the number of concurrent sequences: one row per sequence
    per step.
    """
    global s59
    s59 = tokens
    reader = _Reader()
    load_args(reader)
    return reader.args


def compile_block(*, use_ayla: bool):
    """The block, compiled. With `use_ayla`, matmuls ayla covers go to ayla."""
    import torch._inductor.config as inductor_config

    if use_ayla:
        from ayla import torch_inductor

        torch_inductor.install()
    # The backend is not part of the Inductor cache key, so a warm cache would
    # serve a stock graph and the lowering would never run.
    inductor_config.force_disable_caches = True
    return torch.compile(mod, dynamic=False)


def _kernel_times(prof, iters: int) -> list[tuple[str, float, int]]:
    """(name, us per iteration, launches per iteration), slowest first."""
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    rows = [
        (e.key, e.device_time_total / iters, e.count // iters) for e in events
    ]
    return sorted(rows, key=lambda row: -row[1])


def report(rows: list[tuple[str, float, int]]) -> None:
    """Print the per-kernel breakdown, and how the GEMMs use the memory bus."""
    total = sum(us for _, us, _ in rows)
    gemm = sum(us for name, us, _ in rows if name.startswith("nvjet_")
               or name.startswith("ayla_") or "matmul" in name)
    print(f"\n{'us/iter':>9} {'share':>7} {'n':>3}  kernel")
    for name, us, count in rows:
        print(f"{us:9.1f} {us / total * 100:6.1f}% {count:4}  {name[:64]}")
    print(f"\n  total          {total:8.1f} us/iter")
    print(f"  matmul kernels {gemm:8.1f} us/iter  ({gemm / total * 100:.1f}%)")
    if gemm:
        print(f"  weight traffic {WEIGHT_BYTES / 1e6:8.0f} MB"
              f"  -> {WEIGHT_BYTES / (gemm * 1e-6) / 1e12:.2f} TB/s effective")


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Muse Glimmer 30B decoder block body: 5 GEMMs, no attention.",
    )
    parser.add_argument(
        "--tokens", type=int, default=256,
        help="rows into the block; in decode, the number of concurrent "
             "sequences (vLLM's max_num_seqs defaults to 256)",
    )
    parser.add_argument("--ayla", action="store_true",
                        help="route matmuls ayla covers through ayla")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", metavar="DIR", default=None,
                        help="write a chrome trace here")
    parser.add_argument("--share", action="store_true",
                        help="upload the trace to the internal Perfetto UI")
    args = parser.parse_args()

    from torch.profiler import ProfilerActivity, profile

    inputs = make_inputs(args.tokens)
    compiled = compile_block(use_ayla=args.ayla)

    with torch.no_grad():
        for _ in range(args.warmup):
            compiled(*inputs)
        torch.cuda.synchronize()

        # Time and profile separately: the profiler adds enough per-launch
        # overhead to swamp the measurement.
        started = time.perf_counter()
        for _ in range(args.iters):
            compiled(*inputs)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - started) / args.iters * 1e6

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                compiled(*inputs)
            torch.cuda.synchronize()

    # What ayla claimed shows up in the breakdown below as an `ayla_*` kernel.
    print(f"\ntokens={args.tokens}  wall {wall:.1f} us/iter")
    report(_kernel_times(prof, args.iters))

    if args.profile:
        import os

        os.makedirs(args.profile, exist_ok=True)
        tag = "ayla" if args.ayla else "stock"
        path = os.path.join(args.profile, f"glimmer_{tag}_{args.tokens}.json")
        prof.export_chrome_trace(path)
        print(f"\ntrace: {path}")
        if args.share:
            share(path)


def share(path: str) -> None:
    """Upload a trace to the internal Perfetto UI, printing the URL."""
    import os
    import subprocess

    tool = os.path.expanduser("~/fbsource/arvr/scripts/perfetto/share_trace.py")
    if not os.path.exists(tool):
        print(f"!! no share tool at {tool}")
        return
    done = subprocess.run([tool, path], capture_output=True, text=True)
    urls = [line.strip() for line in (done.stdout + done.stderr).splitlines()
            if line.strip().startswith("https://")]
    if urls:
        print(f"perfetto: {urls[-1]}")
    else:
        print(f"!! upload failed (rc={done.returncode})")
        print((done.stderr or done.stdout).strip()[-400:])


if __name__ == "__main__":
    main()
