// TMA (Tensor Memory Accelerator) example for Hopper (sm_90+).
//
// Loads a 2D tile from global memory into shared memory using
// cp.async.bulk.tensor.2d (TMA load), adds 1.0f in shared memory, then
// stores the tile back to global memory using cp.async.bulk.tensor.2d
// (TMA store). The kernel uses a CUtensorMap descriptor encoded on the
// host with cuTensorMapEncodeTiled.
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_90a tma_load.cu -lcuda -o tma_load
// Run:    ./tma_load

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
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
                   __LINE__, msg ? msg : "?");                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Tile shape (BM rows x BN cols of float).
constexpr int kBM = 64;
constexpr int kBN = 64;

// Full tensor shape (must be a multiple of the tile shape for this demo).
constexpr int kM = 256;
constexpr int kN = 256;

// PTX wrappers for the TMA instructions. The CUtensorMap descriptor is
// passed as a generic pointer in const memory (the kernel parameter ABI
// already places it where the instruction expects it).

// Initialize the mbarrier in shared memory for arrival count = 1.
__device__ static inline void mbarrier_init(uint64_t *bar, uint32_t count) {
  uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar_smem),
               "r"(count));
}

// Issue a 2D TMA load: dst_smem <- desc[crd_x, crd_y]. Bytes-expected is
// posted on the mbarrier so the wait can complete on tx-count.
__device__ static inline void
tma_load_2d(void *dst_smem, const CUtensorMap *desc, int crd_x, int crd_y,
            uint64_t *bar) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(desc), "r"(crd_x), "r"(crd_y), "r"(bar_smem)
      : "memory");
}

// Arrive on the mbarrier with the number of bytes the TMA load will
// deliver, then wait until the load completes.
__device__ static inline void mbarrier_arrive_expect_tx(uint64_t *bar,
                                                        uint32_t bytes) {
  uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_smem),
      "r"(bytes));
}

__device__ static inline void mbarrier_wait(uint64_t *bar, uint32_t phase) {
  uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred p;\n"
               "waitLoop:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@p bra done;\n"
               "bra waitLoop;\n"
               "done:\n"
               "}" ::"r"(bar_smem),
               "r"(phase));
}

// Issue a 2D TMA store: desc[crd_x, crd_y] <- src_smem.
__device__ static inline void tma_store_2d(const CUtensorMap *desc, int crd_x,
                                           int crd_y, void *src_smem) {
  uint32_t src = static_cast<uint32_t>(__cvta_generic_to_shared(src_smem));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], "
      "[%3];" ::"l"(desc),
      "r"(crd_x), "r"(crd_y), "r"(src)
      : "memory");
}

// Commit the TMA store to the bulk async-group and wait for it.
__device__ static inline void tma_store_commit_and_wait() {
  asm volatile("cp.async.bulk.commit_group;");
  asm volatile("cp.async.bulk.wait_group 0;");
}

// Each block loads one BM x BN tile via TMA, adds 1.0f, and stores it back.
// The CUtensorMap descriptors must be passed by const-ref (grid_constant).
__global__ void tma_add_one_kernel(const __grid_constant__ CUtensorMap in_desc,
                                   const __grid_constant__ CUtensorMap
                                       out_desc) {
  // Shared memory tile must be 128-byte aligned for TMA.
  __shared__ alignas(128) float tile[kBM * kBN];
  __shared__ alignas(8) uint64_t bar;

  int tile_x = blockIdx.x * kBN; // column coordinate (in elements)
  int tile_y = blockIdx.y * kBM; // row coordinate (in elements)

  // Single thread issues the TMA load and posts the expected bytes.
  if (threadIdx.x == 0) {
    mbarrier_init(&bar, 1);
    asm volatile("fence.proxy.async.shared::cta;");
    tma_load_2d(tile, &in_desc, tile_x, tile_y, &bar);
    mbarrier_arrive_expect_tx(&bar, sizeof(tile));
  }
  __syncthreads();
  mbarrier_wait(&bar, 0);

  // Operate on the tile in shared memory.
  for (int i = threadIdx.x; i < kBM * kBN; i += blockDim.x) {
    tile[i] += 1.0f;
  }
  __syncthreads();

  // Store the tile back via TMA.
  if (threadIdx.x == 0) {
    asm volatile("fence.proxy.async.shared::cta;");
    tma_store_2d(&out_desc, tile_x, tile_y, tile);
    tma_store_commit_and_wait();
  }
}

// Encode a 2D row-major float tensor of shape (M rows, N cols) with a
// (kBM, kBN) tile into a CUtensorMap descriptor.
static CUtensorMap make_tensor_map_2d(void *gmem, int M, int N) {
  CUtensorMap desc{};
  // TMA uses (fastest, ..., slowest) ordering. For row-major (M, N), the
  // fastest-changing dim is N (columns).
  cuuint64_t size[2] = {static_cast<cuuint64_t>(N),
                        static_cast<cuuint64_t>(M)};
  // Stride of dim[1] in bytes (skip from one row to the next). Stride for
  // dim[0] is implicit (= element size).
  cuuint64_t stride[1] = {static_cast<cuuint64_t>(N) * sizeof(float)};
  cuuint32_t box[2] = {static_cast<cuuint32_t>(kBN),
                       static_cast<cuuint32_t>(kBM)};
  cuuint32_t elem_stride[2] = {1, 1};

  CU_CHECK(cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*tensorRank=*/2, gmem, size,
      stride, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return desc;
}

int main() {
  // Verify Hopper-class device.
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  if (prop.major < 9) {
    std::fprintf(stderr,
                 "TMA requires sm_90+, got sm_%d%d (%s). Skipping.\n",
                 prop.major, prop.minor, prop.name);
    return 0;
  }

  static_assert(kM % kBM == 0 && kN % kBN == 0,
                "tensor must be tileable by the tile shape");

  // Initialize host buffers.
  size_t numel = static_cast<size_t>(kM) * kN;
  size_t bytes = numel * sizeof(float);
  std::vector<float> h_in(numel), h_out(numel);
  for (size_t i = 0; i < numel; ++i) {
    h_in[i] = static_cast<float>(i);
  }

  // Allocate and copy inputs to device buffers.
  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Build TMA descriptors on the host.
  CUtensorMap in_desc = make_tensor_map_2d(d_in, kM, kN);
  CUtensorMap out_desc = make_tensor_map_2d(d_out, kM, kN);

  // Launch one block per tile, 128 threads per block.
  dim3 grid(kN / kBN, kM / kBM);
  dim3 block(128);
  tma_add_one_kernel<<<grid, block>>>(in_desc, out_desc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy outputs to host buffer.
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // Check results.
  int errors = 0;
  for (size_t i = 0; i < numel; ++i) {
    float expected = h_in[i] + 1.0f;
    if (h_out[i] != expected) {
      if (errors < 5) {
        std::fprintf(stderr, "mismatch at %zu: got %f, want %f\n", i, h_out[i],
                     expected);
      }
      ++errors;
    }
  }

  // Cleanup.
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  // Report.
  if (errors == 0) {
    std::printf("OK: %dx%d tile %dx%d\n", kM, kN, kBM, kBN);
    return 0;
  }
  std::printf("FAIL: %d mismatches out of %zu\n", errors, numel);
  return 1;
}
