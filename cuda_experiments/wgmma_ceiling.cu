// Pure-WGMMA ceiling probe: the exact tensor work of the 4096^3 BF16 GEMM
// (128 CTAs x 2 warpgroups x 256 k64 stages x 4 wgmma.m64n256k16) with *no*
// TMA, no mbarriers and no barriers.  Operands are read from a shared-memory
// buffer that is never refilled, so results are meaningless -- the point is the
// issue rate.  SMEM is padded to the real kernel's 221336 bytes so occupancy is
// the same 1 CTA/SM, and the accumulators are written out so nothing is DCE'd.
//
//   nvcc -arch=sm_90a -O3 -o wgmma_ceiling wgmma_ceiling.cu && ./wgmma_ceiling
#include <cstdio>
#include <cuda_runtime.h>

#define SMEM_BYTES 221336
#define STAGES 3
#define LHS_BYTES 16384
#define RHS_BYTES 32768
#define STAGE_BYTES (LHS_BYTES + RHS_BYTES)
#define ITERS 256          // 4 output tiles x 64 k64 stages
#define DESC_CONST 0x4000004000010000ULL

__device__ __forceinline__ unsigned long long desc_at(unsigned smem_addr) {
  return ((unsigned long long)(smem_addr >> 4)) | DESC_CONST;
}

template <int WAIT, bool BARRIER, int NITERS = ITERS>
__global__ __launch_bounds__(256) void wgmma_ceiling(float *out) {
  extern __shared__ char smem[];
  for (int i = threadIdx.x * 4; i < SMEM_BYTES - 3; i += blockDim.x * 4)
    *(int *)(smem + i) = 0;
  __syncthreads();

  unsigned base = (unsigned)__cvta_generic_to_shared(smem);
  float d[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) d[i] = 0.f;

  for (int it = 0; it < NITERS; ++it) {
    unsigned stage = base + (it % STAGES) * STAGE_BYTES;
    unsigned long long da = desc_at(stage);
    unsigned long long db = desc_at(stage + LHS_BYTES);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]),
      "+f"(d[1]),
      "+f"(d[2]),
      "+f"(d[3]),
      "+f"(d[4]),
      "+f"(d[5]),
      "+f"(d[6]),
      "+f"(d[7]),
      "+f"(d[8]),
      "+f"(d[9]),
      "+f"(d[10]),
      "+f"(d[11]),
      "+f"(d[12]),
      "+f"(d[13]),
      "+f"(d[14]),
      "+f"(d[15]),
      "+f"(d[16]),
      "+f"(d[17]),
      "+f"(d[18]),
      "+f"(d[19]),
      "+f"(d[20]),
      "+f"(d[21]),
      "+f"(d[22]),
      "+f"(d[23]),
      "+f"(d[24]),
      "+f"(d[25]),
      "+f"(d[26]),
      "+f"(d[27]),
      "+f"(d[28]),
      "+f"(d[29]),
      "+f"(d[30]),
      "+f"(d[31]),
      "+f"(d[32]),
      "+f"(d[33]),
      "+f"(d[34]),
      "+f"(d[35]),
      "+f"(d[36]),
      "+f"(d[37]),
      "+f"(d[38]),
      "+f"(d[39]),
      "+f"(d[40]),
      "+f"(d[41]),
      "+f"(d[42]),
      "+f"(d[43]),
      "+f"(d[44]),
      "+f"(d[45]),
      "+f"(d[46]),
      "+f"(d[47]),
      "+f"(d[48]),
      "+f"(d[49]),
      "+f"(d[50]),
      "+f"(d[51]),
      "+f"(d[52]),
      "+f"(d[53]),
      "+f"(d[54]),
      "+f"(d[55]),
      "+f"(d[56]),
      "+f"(d[57]),
      "+f"(d[58]),
      "+f"(d[59]),
      "+f"(d[60]),
      "+f"(d[61]),
      "+f"(d[62]),
      "+f"(d[63]),
      "+f"(d[64]),
      "+f"(d[65]),
      "+f"(d[66]),
      "+f"(d[67]),
      "+f"(d[68]),
      "+f"(d[69]),
      "+f"(d[70]),
      "+f"(d[71]),
      "+f"(d[72]),
      "+f"(d[73]),
      "+f"(d[74]),
      "+f"(d[75]),
      "+f"(d[76]),
      "+f"(d[77]),
      "+f"(d[78]),
      "+f"(d[79]),
      "+f"(d[80]),
      "+f"(d[81]),
      "+f"(d[82]),
      "+f"(d[83]),
      "+f"(d[84]),
      "+f"(d[85]),
      "+f"(d[86]),
      "+f"(d[87]),
      "+f"(d[88]),
      "+f"(d[89]),
      "+f"(d[90]),
      "+f"(d[91]),
      "+f"(d[92]),
      "+f"(d[93]),
      "+f"(d[94]),
      "+f"(d[95]),
      "+f"(d[96]),
      "+f"(d[97]),
      "+f"(d[98]),
      "+f"(d[99]),
      "+f"(d[100]),
      "+f"(d[101]),
      "+f"(d[102]),
      "+f"(d[103]),
      "+f"(d[104]),
      "+f"(d[105]),
      "+f"(d[106]),
      "+f"(d[107]),
      "+f"(d[108]),
      "+f"(d[109]),
      "+f"(d[110]),
      "+f"(d[111]),
      "+f"(d[112]),
      "+f"(d[113]),
      "+f"(d[114]),
      "+f"(d[115]),
      "+f"(d[116]),
      "+f"(d[117]),
      "+f"(d[118]),
      "+f"(d[119]),
      "+f"(d[120]),
      "+f"(d[121]),
      "+f"(d[122]),
      "+f"(d[123]),
      "+f"(d[124]),
      "+f"(d[125]),
      "+f"(d[126]),
      "+f"(d[127])
        : "l"(da), "l"(db));
      da += 2; db += 2;
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (WAIT == 0)      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    else if (WAIT == 1) asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    else                asm volatile("wgmma.wait_group.sync.aligned 4;\n" ::: "memory");
    if (BARRIER) {
      // Same split rendezvous the real mainloop pays once per stage.
      if (threadIdx.x < 32) asm volatile("bar.sync 1, 256;\n" ::: "memory");
      else                  asm volatile("bar.arrive 1, 256;\n" ::: "memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 128; ++i) s += d[i];
  if (s == 12345.678f) out[blockIdx.x] = s;
}

static float bench(void (*k)(float *), float *out, int shmem) {
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
  for (int i = 0; i < 20; ++i) k<<<128, 256, shmem>>>(out);
  cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float best = 1e30f;
  for (int r = 0; r < 20; ++r) {
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) k<<<128, 256, shmem>>>(out);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    if (ms / 20 < best) best = ms / 20;
  }
  return best * 1e3f;  // us
}

int main() {
  float *out; cudaMalloc(&out, 1024 * sizeof(float));
  double flop = 2.0 * 4096 * 4096 * 4096;
  int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  struct { const char *name; void (*k)(float *); } vs[] = {
    {"wgmma only, wait_group 0",            wgmma_ceiling<0, false>},
    {"wgmma only, wait_group 1",            wgmma_ceiling<1, false>},
    {"wgmma only, wait_group 4",            wgmma_ceiling<4, false>},
    {"wgmma + leader barrier, wait_group 0", wgmma_ceiling<0, true>},
    {"fill+writeback only (overhead)",       wgmma_ceiling<0, false, 0>},
  };
  printf("%-40s %9s %10s\n", "variant", "us", "TFLOP/s");
  for (auto &v : vs) {
    float us = bench(v.k, out, SMEM_BYTES);
    printf("%-40s %9.2f %10.1f\n", v.name, us, flop / (us * 1e-6) / 1e12);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
  return 0;
}
