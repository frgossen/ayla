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

from dataclasses import dataclass
from functools import cache

import torch
from torch.utils.cpp_extension import load_inline

from .gpu_dialect import GmemTensorType, TmaDescType
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
    int optimization_level,
    int max_registers,
    int threads_per_block,
    int cache_mode,
    bool generate_line_info
) {
    CUmodule mod = nullptr;
    TORCH_CHECK(
        optimization_level >= -1 && optimization_level <= 4,
        "optimization_level must be unset (-1) or in [0, 4]"
    );
    TORCH_CHECK(max_registers == -1 || max_registers > 0,
                "max_registers must be unset (-1) or positive");
    TORCH_CHECK(threads_per_block == -1 || threads_per_block > 0,
                "threads_per_block must be unset (-1) or positive");
    TORCH_CHECK(cache_mode >= -1 && cache_mode <= 2,
                "cache_mode must be unset (-1), none (0), cg (1), or ca (2)");

    std::vector<CUjit_option> options;
    std::vector<void*> values;
    auto add_uint_option = [&](CUjit_option option, int value) {
        options.push_back(option);
        values.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(value)));
    };
    // Line info lets Nsight Compute correlate generated PTX with SASS.
    if (generate_line_info)
        add_uint_option(CU_JIT_GENERATE_LINE_INFO, 1);
    if (optimization_level >= 0)
        add_uint_option(CU_JIT_OPTIMIZATION_LEVEL, optimization_level);
    if (max_registers > 0)
        add_uint_option(CU_JIT_MAX_REGISTERS, max_registers);
    if (threads_per_block > 0)
        add_uint_option(CU_JIT_THREADS_PER_BLOCK, threads_per_block);
    if (cache_mode >= 0)
        add_uint_option(CU_JIT_CACHE_MODE, cache_mode);

    std::vector<char> error_log(16384, '\0');
    options.push_back(CU_JIT_ERROR_LOG_BUFFER);
    values.push_back(error_log.data());
    add_uint_option(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, error_log.size());

    CUresult load_result = cuModuleLoadDataEx(
        &mod, ptx.c_str(), options.size(), options.data(), values.data()
    );
    if (load_result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(load_result, &message);
        throw std::runtime_error(
            std::string("CUDA driver error (cuModuleLoadDataEx): ")
            + (message ? message : "unknown")
            + (error_log[0] ? std::string("\n") + error_log.data() : "")
        );
    }
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

static py::dict function_attributes(uintptr_t function_handle) {
    auto fn = reinterpret_cast<CUfunction>(function_handle);
    py::dict attrs;
    auto add = [&](const char* name, CUfunction_attribute attribute) {
        int value = 0;
        CU_CHECK(cuFuncGetAttribute(&value, attribute, fn));
        attrs[name] = value;
    };
    add("num_regs", CU_FUNC_ATTRIBUTE_NUM_REGS);
    add("local_size_bytes", CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    add("shared_size_bytes", CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    add("const_size_bytes", CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    add("max_threads_per_block", CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    add("binary_version", CU_FUNC_ATTRIBUTE_BINARY_VERSION);
    add("ptx_version", CU_FUNC_ATTRIBUTE_PTX_VERSION);
    return attrs;
}

static CUtensorMap tensor_map_encode_tiled(
    at::Tensor tensor,
    std::vector<int64_t> tile_shape,
    std::string swizzle,
    int64_t inner_size,
    int64_t inner_padding,
    std::string block_order,
    std::string l2_promotion
) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be on CUDA");
    TORCH_CHECK(tensor.dim() == 2, "only 2D tensors supported");
    TORCH_CHECK(tile_shape.size() == 2, "tile_shape must have 2 elements");

    auto dtype = tensor.scalar_type();
    CUtensorMapDataType cu_dtype;
    if (dtype == at::kFloat) cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    else if (dtype == at::kHalf) cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    else if (dtype == at::kBFloat16) cu_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (dtype == at::kDouble) cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    else TORCH_CHECK(false, "unsupported dtype for TMA descriptor");

    CUtensorMapSwizzle cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    if (swizzle == "32B") cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
    else if (swizzle == "64B") cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
    else if (swizzle == "128B") cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    CUtensorMapL2promotion cu_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    if (l2_promotion == "64B")
        cu_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    else if (l2_promotion == "128B")
        cu_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    else if (l2_promotion == "256B")
        cu_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    else TORCH_CHECK(l2_promotion.empty(), "unsupported TMA L2 promotion");

    CUtensorMap map{};
    if (inner_size > 0) {
        TORCH_CHECK(inner_padding >= 0, "TMA inner padding must be non-negative");
        TORCH_CHECK(tensor.size(1) % inner_size == 0,
                    "tensor inner dimension must be divisible by TMA inner size");
        TORCH_CHECK(tile_shape[1] % inner_size == 0,
                    "tile inner dimension must be divisible by TMA inner size");
        // Reshape row-major [M,N] to [M,N/inner,inner].  Making the
        // innermost box slightly wider than the global inner dimension adds
        // shared-memory-only padding which TMA skips in global memory.
        TORCH_CHECK(block_order == "row_major" || block_order == "block_major",
                    "unsupported TMA blocked layout order");
        bool block_major = block_order == "block_major";
        cuuint64_t global_dim[3] = {
            static_cast<cuuint64_t>(inner_size),
            static_cast<cuuint64_t>(block_major
                ? tensor.size(0) : tensor.size(1) / inner_size),
            static_cast<cuuint64_t>(block_major
                ? tensor.size(1) / inner_size : tensor.size(0)),
        };
        cuuint64_t global_strides[2] = {
            static_cast<cuuint64_t>((block_major ? tensor.stride(0) : inner_size)
                                    * tensor.element_size()),
            static_cast<cuuint64_t>((block_major ? inner_size : tensor.stride(0))
                                    * tensor.element_size()),
        };
        cuuint32_t box_dim[3] = {
            static_cast<cuuint32_t>(inner_size + inner_padding),
            static_cast<cuuint32_t>(block_major
                ? tile_shape[0] : tile_shape[1] / inner_size),
            static_cast<cuuint32_t>(block_major
                ? tile_shape[1] / inner_size : tile_shape[0]),
        };
        cuuint32_t elem_strides[3] = {1, 1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &map, cu_dtype, 3, tensor.data_ptr(),
            global_dim, global_strides, box_dim, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, cu_swizzle,
            cu_l2_promotion,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    } else {
        // TMA uses fastest-varying-first ordering.  For row-major [M,N],
        // dimension zero is N and dimension one is M.
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
        CU_CHECK(cuTensorMapEncodeTiled(
            &map, cu_dtype, 2, tensor.data_ptr(),
            global_dim, global_strides, box_dim, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, cu_swizzle,
            cu_l2_promotion,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    }
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
    int64_t inner_size,
    int64_t inner_padding,
    std::string block_order,
    std::string l2_promotion
) {
    TORCH_CHECK(tile_shape.size() == 2, "tile_shape must have 2 elements");
    auto key = std::make_tuple(
        reinterpret_cast<uintptr_t>(tensor.data_ptr()),
        static_cast<int>(tensor.scalar_type()),
        static_cast<int64_t>(tensor.size(0)),
        static_cast<int64_t>(tensor.size(1)),
        static_cast<int64_t>(tensor.stride(0)),
        tile_shape[0], tile_shape[1],
        swizzle, inner_size, inner_padding, block_order, l2_promotion
    );
    static std::map<decltype(key), CUtensorMap> cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(
            key, tensor_map_encode_tiled(
                tensor, tile_shape, swizzle, inner_size, inner_padding,
                block_order, l2_promotion)
        ).first;
    }
    return &it->second;
}

// Launch a kernel. `params` is a list where each element is either:
//   - an at::Tensor (passed as an 8-byte device pointer)
//   - a tuple (tensor, tile_shape, swizzle, inner_size, inner_padding,
//              block_order, l2_promotion) for TMA
//     (a 128-byte CUtensorMap, cached across launches)
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
    // TMA descriptors come from a persistent cache, so their pointers are
    // stable without per-launch storage.
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
            auto inner_size = tup[3].cast<int64_t>();
            auto inner_padding = tup[4].cast<int64_t>();
            auto block_order = tup[5].cast<std::string>();
            auto l2_promotion = tup[6].cast<std::string>();
            arg_ptrs.push_back(const_cast<CUtensorMap*>(
                tensor_map_cached(
                    tensor, tile_shape, swizzle, inner_size, inner_padding,
                    block_order, l2_promotion)));
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
          py::arg("ptx"),
          py::arg("optimization_level") = -1,
          py::arg("max_registers") = -1,
          py::arg("threads_per_block") = -1,
          py::arg("cache_mode") = -1,
          py::arg("generate_line_info") = true);
    m.def("module_unload", &module_unload);
    m.def("module_get_function", &module_get_function);
    m.def("function_attributes", &function_attributes);
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


@dataclass(frozen=True)
class PtxJitOptions:
    """CUDA Driver PTX-JIT controls.

    ``None`` leaves an option unspecified so the driver uses its default.
    CUDA documents optimization levels 0 through 4, with 4 as the default.
    """

    optimization_level: int | None = None
    max_registers: int | None = None
    threads_per_block: int | None = None
    cache_mode: str | None = None
    generate_line_info: bool = True

    def __post_init__(self) -> None:
        if (
            self.optimization_level is not None
            and self.optimization_level not in range(5)
        ):
            raise ValueError("optimization_level must be in [0, 4]")
        if self.max_registers is not None and self.max_registers <= 0:
            raise ValueError("max_registers must be positive")
        if self.threads_per_block is not None and self.threads_per_block <= 0:
            raise ValueError("threads_per_block must be positive")
        if self.cache_mode not in (None, "none", "cg", "ca"):
            raise ValueError("cache_mode must be None, 'none', 'cg', or 'ca'")


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
        jit_options: PtxJitOptions | None = None,
    ):
        self._ext = _get_ext()
        self._kernel = kernel
        self._jit_options = jit_options or PtxJitOptions()
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
            options = self._jit_options
            self._module = self._ext.module_load_data(
                self._kernel.ptx,
                options.optimization_level
                if options.optimization_level is not None else -1,
                options.max_registers if options.max_registers is not None else -1,
                options.threads_per_block
                if options.threads_per_block is not None else -1,
                {None: -1, "none": 0, "cg": 1, "ca": 2}[options.cache_mode],
                options.generate_line_info,
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

    @property
    def jit_options(self) -> PtxJitOptions:
        return self._jit_options

    @property
    def function_attributes(self) -> dict[str, int]:
        self._ensure_loaded()
        return self._ext.function_attributes(self._function)

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
                layout = typ.physical_layout or {}
                params.append((
                    tensors[self._arg_to_tensor[typ.arg_idx]],
                    list(typ.tile_sizes),
                    typ.swizzle or "",
                    int(layout.get("block", 0)),
                    int(layout.get("padding", 0)),
                    str(layout.get("order", "row_major")),
                    typ.l2_promotion or "",
                ))
        return params

    @staticmethod
    def _assert_params(params: list) -> None:
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                assert p.is_cuda, f"param {i}: expected CUDA tensor, got {p.device}"
            elif isinstance(p, tuple):
                assert len(p) == 7, f"param {i}: expected 7-tuple, got {len(p)}"
                t, tile, swz, inner, padding, order, promotion = p
                assert isinstance(t, torch.Tensor) and t.is_cuda, (
                    f"param {i}: first element must be a CUDA tensor"
                )
                assert isinstance(tile, list) and all(isinstance(x, int) for x in tile), (
                    f"param {i}: second element must be list[int]"
                )
                assert isinstance(swz, str), f"param {i}: third element must be str"
                assert isinstance(inner, int), f"param {i}: fourth element must be int"
                assert isinstance(padding, int), f"param {i}: fifth element must be int"
                assert isinstance(order, str), f"param {i}: sixth element must be str"
                assert isinstance(promotion, str), (
                    f"param {i}: seventh element must be str"
                )
            else:
                raise TypeError(f"param {i}: expected Tensor or tuple, got {type(p)}")

    def __del__(self):
        try:
            self._ext.module_unload(self._module)
        except Exception:
            pass


def make_ptx_launcher(
    kernel: PtxKernel,
    *,
    jit_options: PtxJitOptions | None = None,
) -> PtxLauncher:
    return PtxLauncher(kernel, jit_options=jit_options)
