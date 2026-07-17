// Componentwise (elementwise) CUDA kernel + a minimal program that
// exercises it. Computes c[i] = a[i] + b[i] over a 1D contiguous array.
//
// Build:  nvcc -O3 -std=c++17 cwise.cu -o cwise
// Run:    ./cwise

#include <cstdio>
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

__global__ void cwise_add(const float *__restrict__ a,
                          const float *__restrict__ b, float *__restrict__ c,
                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

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

  // Launch kernel.
  constexpr int kBlockSize = 256;
  int grid = (n + kBlockSize - 1) / kBlockSize;
  cwise_add<<<grid, kBlockSize>>>(da, db, dc, n);
  CUDA_CHECK(cudaGetLastError());
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
