// Same as cwise.cu but the kernel body is implemented in raw PTX, loaded
// at runtime via the CUDA Driver API. Computes c[i] = a[i] + b[i].
//
// Build:  nvcc -O3 -std=c++17 cwise_ptx.cu -o cwise_ptx -lcuda
// Run:    ./cwise_ptx

#include <cstdio>
#include <cuda.h>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,    \
                   __LINE__, cudaGetErrorString(_err));                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CU_CHECK(expr)                                                         \
  do {                                                                         \
    CUresult _err = (expr);                                                    \
    if (_err != CUDA_SUCCESS) {                                                \
      const char *msg = nullptr;                                               \
      cuGetErrorString(_err, &msg);                                            \
      std::fprintf(stderr, "CU error %s at %s:%d: %s\n", #expr, __FILE__,      \
                   __LINE__, msg ? msg : "unknown");                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Hand-written PTX kernel: cwise_add(const float* a, const float* b, float* c,
// int n). Equivalent to the body of the cwise_add kernel in cwise.cu.
static const char *kCwiseAddPtx = R"PTX(
.version 9.0
.target sm_80
.address_size 64

.visible .entry cwise_add(
    .param .u64 cwise_add_param_0,
    .param .u64 cwise_add_param_1,
    .param .u64 cwise_add_param_2,
    .param .u32 cwise_add_param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<11>;

    ld.param.u32    %r2,  [cwise_add_param_3];
    ld.param.u64    %rd3, [cwise_add_param_2];
    ld.param.u64    %rd1, [cwise_add_param_0];
    ld.param.u64    %rd2, [cwise_add_param_1];
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %ntid.x;
    mov.u32         %r5, %tid.x;
    mad.lo.s32      %r1, %r3, %r4, %r5;
    setp.ge.s32     %p1, %r1, %r2;
    @%p1 bra        $L__BB0_2;

    cvta.to.global.u64  %rd4, %rd1;
    mul.wide.s32        %rd5, %r1, 4;
    add.s64             %rd6, %rd4, %rd5;
    cvta.to.global.u64  %rd7, %rd2;
    add.s64             %rd8, %rd7, %rd5;
    ld.global.nc.f32    %f1, [%rd8];
    ld.global.nc.f32    %f2, [%rd6];
    add.f32             %f3, %f2, %f1;
    cvta.to.global.u64  %rd9, %rd3;
    add.s64             %rd10, %rd9, %rd5;
    st.global.f32       [%rd10], %f3;

$L__BB0_2:
    ret;
}
)PTX";

int main() {

  // Initialize host buffers.
  constexpr int n = 1 << 20;
  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = 2.0f * static_cast<float>(i);
  }

  // Allocate and copy inputs to device buffers.
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  size_t bytes = sizeof(float) * static_cast<size_t>(n);
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dc, bytes));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

  // Load the PTX module and look up the kernel via the CUDA Driver API.
  // The runtime API has already initialized the primary context for us.
  CUmodule module;
  CUfunction kernel;
  CU_CHECK(cuModuleLoadData(&module, kCwiseAddPtx));
  CU_CHECK(cuModuleGetFunction(&kernel, module, "cwise_add"));

  // Launch kernel.
  constexpr int kBlockSize = 256;
  int grid = (n + kBlockSize - 1) / kBlockSize;
  void *args[] = {&da, &db, &dc, const_cast<int *>(&n)};
  CU_CHECK(cuLaunchKernel(kernel, grid, 1, 1, kBlockSize, 1, 1,
                          /*sharedMemBytes=*/0, /*stream=*/nullptr, args,
                          /*extra=*/nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy outputs to host buffer.
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

  // Check results.
  int errors = 0;
  for (int i = 0; i < n; ++i) {
    float expected = ha[i] + hb[i];
    if (hc[i] != expected) {
      if (errors < 5) {
        std::fprintf(stderr, "mismatch at %d: got %f, want %f\n", i, hc[i],
                     expected);
      }
      ++errors;
    }
  }

  // Cleanup.
  CU_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));

  // Report.
  if (errors == 0) {
    std::printf("OK: n=%d\n", n);
    return 0;
  }
  std::printf("FAIL: %d mismatches out of %d\n", errors, n);
  return 1;
}
