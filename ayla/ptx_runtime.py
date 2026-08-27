"""PTX custom calls: host-side glue for launching generated PTX kernels.

A :class:`PtxLauncher` bundles:
    - the PTX source (driver-JIT'd via ``cuModuleLoadDataEx``),
    - a :class:`~ayla.ptx.PtxKernel` (entry point, grid_dim/block_dim, shmem,
      and — in the future — tensor-core / TMA / cluster setup),
    - the host-side wrapper that converts ``torch.Tensor`` args into kernel
      params and calls ``cuLaunchKernel`` on the current torch CUDA stream.

From the framework's perspective each instance is opaque: tensors in,
side-effects out.
"""
from __future__ import annotations

from functools import cache

import torch
from torch.utils.cpp_extension import load_inline

from .gpu_dialect import GmemTensorType, TmaDescType, physical_layout
from .ptx_codegen import PtxKernel



_LAUNCHER_CPP = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <map>
#include <tuple>
#include <vector>

namespace py = pybind11;

static void cu_check(CUresult res, const char* expr) {
    if (res == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    cuGetErrorString(res, &msg);
    throw std::runtime_error(
        std::string("CUDA driver error (") + expr + "): "
        + (msg ? msg : "unknown")
    );
}

#define CU_CHECK(expr) cu_check((expr), #expr)

static uintptr_t module_load_data(
    const std::string& ptx,
    int optimization_level
) {
    TORCH_CHECK(
        optimization_level >= -1 && optimization_level <= 4,
        "optimization_level must be unset (-1) or in [0, 4]"
    );
    CUmodule mod = nullptr;
    // Emit line info during JIT so Nsight Compute can correlate the
    // generated PTX with the SASS in its source view.
    CUjit_option options[] = {
        CU_JIT_GENERATE_LINE_INFO,
        CU_JIT_OPTIMIZATION_LEVEL,
    };
    void* values[] = {
        reinterpret_cast<void*>(1),
        reinterpret_cast<void*>(static_cast<uintptr_t>(optimization_level)),
    };
    unsigned int num_options = optimization_level < 0 ? 1 : 2;
    CU_CHECK(cuModuleLoadDataEx(
        &mod, ptx.c_str(), num_options, options, values
    ));
    return reinterpret_cast<uintptr_t>(mod);
}

static void module_unload(uintptr_t handle) {
    auto mod = reinterpret_cast<CUmodule>(handle);
    CU_CHECK(cuModuleUnload(mod));
}

static uintptr_t module_get_function(
    uintptr_t module_handle, const std::string& name
) {
    auto mod = reinterpret_cast<CUmodule>(module_handle);
    CUfunction fn = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn, mod, name.c_str()));
    return reinterpret_cast<uintptr_t>(fn);
}

static CUtensorMapL2promotion parse_l2_promotion(
    const std::string& l2_promotion
) {
    if (l2_promotion == "64B") return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    if (l2_promotion == "128B") return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    if (l2_promotion == "256B") return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    TORCH_CHECK(l2_promotion.empty(), "unsupported TMA L2 promotion");
    return CU_TENSOR_MAP_L2_PROMOTION_NONE;
}

static CUtensorMap tensor_map_encode_2d(
    const at::Tensor& tensor,
    const std::vector<int64_t>& tile_shape,
    CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle,
    const std::string& l2_promotion
) {
    cuuint64_t global_dim[2] = {
        static_cast<cuuint64_t>(tensor.size(1)),
        static_cast<cuuint64_t>(tensor.size(0)),
    };
    cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(tensor.stride(0) * tensor.element_size()),
    };
    cuuint32_t box_dim[2] = {
        static_cast<cuuint32_t>(tile_shape[1]),
        static_cast<cuuint32_t>(tile_shape[0]),
    };
    cuuint32_t elem_strides[2] = {1, 1};
    CUtensorMap map{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &map, dtype, 2, tensor.data_ptr(),
        global_dim, global_strides, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        parse_l2_promotion(l2_promotion),
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    return map;
}

static CUtensorMap tensor_map_encode_3d(
    const at::Tensor& tensor,
    const std::vector<int64_t>& tile_shape,
    CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle,
    int64_t block_width,
    const std::string& block_order,
    const std::string& l2_promotion
) {
    TORCH_CHECK(block_width > 0, "TMA block width must be positive");
    TORCH_CHECK(tensor.size(1) % block_width == 0,
                "tensor columns must be divisible by TMA block width");
    TORCH_CHECK(tile_shape[1] % block_width == 0,
                "tile columns must be divisible by TMA block width");
    TORCH_CHECK(block_order == "row_major" || block_order == "block_major",
                "unsupported TMA blocked layout order");

    bool block_major = block_order == "block_major";
    auto rows = tensor.size(0);
    auto blocks = tensor.size(1) / block_width;
    auto tile_rows = tile_shape[0];
    auto tile_blocks = tile_shape[1] / block_width;
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(block_width),
        static_cast<cuuint64_t>(block_major ? rows : blocks),
        static_cast<cuuint64_t>(block_major ? blocks : rows),
    };
    cuuint64_t global_strides[2] = {
        static_cast<cuuint64_t>((block_major ? tensor.stride(0) : block_width)
                                * tensor.element_size()),
        static_cast<cuuint64_t>((block_major ? block_width : tensor.stride(0))
                                * tensor.element_size()),
    };
    cuuint32_t box_dim[3] = {
        static_cast<cuuint32_t>(block_width),
        static_cast<cuuint32_t>(block_major ? tile_rows : tile_blocks),
        static_cast<cuuint32_t>(block_major ? tile_blocks : tile_rows),
    };
    cuuint32_t elem_strides[3] = {1, 1, 1};
    CUtensorMap map{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &map, dtype, 3, tensor.data_ptr(),
        global_dim, global_strides, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        parse_l2_promotion(l2_promotion),
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    return map;
}

// Return a stable pointer to a cached CUtensorMap, encoding it once per
// distinct (ptr, dtype, shape, row-stride, tile, swizzle). The descriptor
// is a pure function of those inputs, so a cache hit is byte-identical to a
// fresh encode — safe even when a device pointer is reused for a new tensor
// of the same layout (the map describes memory layout, not contents).
// std::map keeps element references stable across inserts, so the returned
// pointer stays valid for use as a kernel param. Calls are serialized by the
// Python GIL (this function never releases it).
static const CUtensorMap* tensor_map_cached(
    at::Tensor tensor,
    std::vector<int64_t> tile_shape,
    std::string swizzle,
    int64_t block_width,
    std::string block_order,
    std::string l2_promotion
) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be on CUDA");
    TORCH_CHECK(tensor.dim() == 2, "only 2D tensors supported");
    TORCH_CHECK(tile_shape.size() == 2, "tile_shape must have 2 elements");

    CUtensorMapDataType dtype;
    auto scalar_type = tensor.scalar_type();
    if (scalar_type == at::kFloat) dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    else if (scalar_type == at::kHalf) dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    else if (scalar_type == at::kBFloat16) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (scalar_type == at::kDouble) dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    else TORCH_CHECK(false, "unsupported dtype for TMA descriptor");

    CUtensorMapSwizzle tensor_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    if (swizzle == "32B") tensor_swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
    else if (swizzle == "64B") tensor_swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
    else if (swizzle == "128B") tensor_swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    auto key = std::make_tuple(
        reinterpret_cast<uintptr_t>(tensor.data_ptr()),
        static_cast<int>(tensor.scalar_type()),
        static_cast<int64_t>(tensor.size(0)),
        static_cast<int64_t>(tensor.size(1)),
        static_cast<int64_t>(tensor.stride(0)),
        tile_shape[0], tile_shape[1],
        swizzle, block_width, block_order, l2_promotion
    );
    static std::map<decltype(key), CUtensorMap> cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
        CUtensorMap map = block_width > 0
            ? tensor_map_encode_3d(
                tensor, tile_shape, dtype, tensor_swizzle,
                block_width, block_order, l2_promotion)
            : tensor_map_encode_2d(
                tensor, tile_shape, dtype, tensor_swizzle, l2_promotion);
        it = cache.emplace(key, map).first;
    }
    return &it->second;
}

static void launch_kernel(
    uintptr_t function_handle,
    std::tuple<int, int, int> grid_dim,
    std::tuple<int, int, int> block_dim,
    py::list params,
    int64_t shared_mem_bytes
) {
    auto fn = reinterpret_cast<CUfunction>(function_handle);

    // reserve() guarantees no reallocation, so &vec.back() pointers
    // stored in arg_ptrs remain valid for the lifetime of the vectors.
    size_t n = params.size();
    std::vector<int64_t> ptr_storage;
    std::vector<void*> arg_ptrs;
    ptr_storage.reserve(n);
    arg_ptrs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        auto obj = params[i];
        if (py::isinstance<py::tuple>(obj)) {
            auto tup = obj.cast<py::tuple>();
            auto tensor = tup[0].cast<at::Tensor>();
            auto tile_shape = tup[1].cast<std::vector<int64_t>>();
            auto swizzle = tup[2].cast<std::string>();
            auto block_width = tup[3].cast<int64_t>();
            auto block_order = tup[4].cast<std::string>();
            auto l2_promotion = tup[5].cast<std::string>();
            arg_ptrs.push_back(const_cast<CUtensorMap*>(
                tensor_map_cached(
                    tensor, tile_shape, swizzle, block_width, block_order,
                    l2_promotion)));
        } else {
            auto t = obj.cast<at::Tensor>();
            ptr_storage.push_back(reinterpret_cast<int64_t>(t.data_ptr()));
            arg_ptrs.push_back(&ptr_storage.back());
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto [gx, gy, gz] = grid_dim;
    auto [bx, by, bz] = block_dim;
    CU_CHECK(cuLaunchKernel(
        fn, gx, gy, gz, bx, by, bz,
        static_cast<unsigned int>(shared_mem_bytes),
        stream, arg_ptrs.data(), /*extra=*/nullptr
    ));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("module_load_data", &module_load_data,
          py::arg("ptx"), py::arg("optimization_level") = -1);
    m.def("module_unload", &module_unload);
    m.def("module_get_function", &module_get_function);
    m.def("launch_kernel", &launch_kernel,
          py::arg("function_handle"),
          py::arg("grid_dim"),
          py::arg("block_dim"),
          py::arg("params"),
          py::arg("shared_mem_bytes") = 0);
}
"""


@cache
def _get_ext():
    """JIT-build the C++ launcher extension (cached after first call)."""
    return load_inline(
        name="ayla_ptx_launcher",
        cpp_sources=[_LAUNCHER_CPP],
        extra_ldflags=["-lcuda"],  # CUDA Driver API
        with_cuda=True,
        verbose=False,
    )




class PtxLauncher:
    """Opaque host-side function that launches a PTX kernel.

    Constructed from a :class:`PtxKernel`. Calling the instance with
    tensors performs all required host-side setup and issues the launch on
    the current torch CUDA stream.
    """

    def __init__(
        self,
        kernel: PtxKernel,
        *,
        optimization_level: int | None = None,
    ):
        if optimization_level is not None and optimization_level not in range(5):
            raise ValueError("optimization_level must be in [0, 4]")
        self._ext = None
        self._kernel = kernel
        self._optimization_level = optimization_level
        self._module = None
        self._function = None
        # Map kernel arg index → user tensor index.
        self._arg_to_tensor: dict[int, int] = {}
        ti = 0
        for i, typ in enumerate(kernel.signature):
            if isinstance(typ, GmemTensorType):
                self._arg_to_tensor[i] = ti
                ti += 1

    def _ensure_loaded(self):
        if self._module is None:
            self._ext = _get_ext()
            self._module = self._ext.module_load_data(
                self._kernel.ptx,
                self._optimization_level
                if self._optimization_level is not None else -1,
            )
            self._function = self._ext.module_get_function(
                self._module, self._kernel.name,
            )

    @property
    def name(self):
        return self._kernel.name

    @property
    def signature(self):
        return self._kernel.signature

    def __call__(self, *tensors) -> None:
        self._ensure_loaded()
        params = self._build_params(tensors)
        self._assert_params(params)
        self._ext.launch_kernel(
            self._function,
            self._kernel.grid_dim,
            self._kernel.block_dim,
            params,
            self._kernel.shared_mem_bytes,
        )

    def _build_params(self, tensors):
        params = []
        for i, typ in enumerate(self._kernel.signature):
            if isinstance(typ, GmemTensorType):
                params.append(tensors[self._arg_to_tensor[i]])
            elif isinstance(typ, TmaDescType):
                layout = physical_layout(typ)
                block_width, block_order = (
                    (0, "row_major") if layout is None else layout
                )
                params.append((
                    tensors[self._arg_to_tensor[typ.arg_idx]],
                    list(typ.tile_sizes),
                    typ.swizzle or "",
                    block_width,
                    block_order,
                    typ.l2_promotion or "",
                ))
        return params

    @staticmethod
    def _assert_params(params: list) -> None:
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                assert p.is_cuda, f"param {i}: expected CUDA tensor, got {p.device}"
            elif isinstance(p, tuple):
                assert len(p) == 6, f"param {i}: expected 6-tuple, got {len(p)}"
                t, tile, swz, block_width, block_order, promotion = p
                assert isinstance(t, torch.Tensor) and t.is_cuda, (
                    f"param {i}: first element must be a CUDA tensor"
                )
                assert isinstance(tile, list) and all(isinstance(x, int) for x in tile), (
                    f"param {i}: second element must be list[int]"
                )
                assert isinstance(swz, str), f"param {i}: third element must be str"
                assert isinstance(block_width, int) and block_width >= 0, (
                    f"param {i}: fourth element must be a nonnegative int"
                )
                assert block_order in ("row_major", "block_major"), (
                    f"param {i}: fifth element must be a supported block order"
                )
                assert promotion in ("", "64B", "128B", "256B"), (
                    f"param {i}: sixth element must be a supported L2 promotion"
                )
            else:
                raise TypeError(f"param {i}: expected Tensor or tuple, got {type(p)}")

    def __del__(self):
        if self._module is None:
            return
        try:
            self._ext.module_unload(self._module)
        except Exception:
            pass


def make_ptx_launcher(
    kernel: PtxKernel,
    *,
    optimization_level: int | None = None,
) -> PtxLauncher:
    return PtxLauncher(kernel, optimization_level=optimization_level)
