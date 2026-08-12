import torch
import torch.nn.functional as F
import time

def test_tile_fused_dual_gemm(M=4096, K1=4096, N1=128, N2=4096, device='cuda'):
    print(f"--- 1. Tile-Fused Dual GEMM Baseline ---")
    print(f"Shapes: A[{M}, {K1}], B1[{K1}, {N1}], B2[{N1}, {N2}]")

    # Initialize tensors
    A = torch.randn(M, K1, dtype=torch.bfloat16, device=device)
    B1 = torch.randn(K1, N1, dtype=torch.bfloat16, device=device)
    B2 = torch.randn(N1, N2, dtype=torch.bfloat16, device=device)

    # Mathematical baseline
    # Note: Evaluated sequentially to match the intra-kernel numerical behavior
    intermediate = torch.matmul(A, B1)
    out = torch.matmul(intermediate, B2)

    print(f"Intermediate Output dtype: {intermediate.dtype}, shape: {intermediate.shape}")
    print(f"Final Output dtype: {out.dtype}, shape: {out.shape}\n")
    return out

def test_stable_softplus_epilogue(M=4096, K=4096, N=4096, tau=1.0, device='cuda'):
    print(f"--- 2. Stable Softplus Epilogue Baseline ---")
    print(f"Shapes: A[{M}, {K}], B[{K}, {N}], tau={tau}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    # Perform GEMM (accumulation is generally fp32 internally)
    C = torch.matmul(A, B).to(torch.float32)

    # Stable softplus: PyTorch's F.softplus is stable by default.
    # We explicitly define the beta (1/tau) and threshold (20)
    # to match the specification limits logic.
    beta = 1.0 / tau
    C_softplus = F.softplus(C, beta=beta, threshold=20.0)

    # Downcast back to bfloat16
    out = C_softplus.to(torch.bfloat16)

    print(f"Output dtype: {out.dtype}, shape: {out.shape}\n")
    return out

def test_dynamic_quant_epilogue(M=4096, K=4096, N=4096, device='cuda'):
    print(f"--- 3. Dynamic Quantization Epilogue Baseline ---")
    print(f"Shapes: A[{M}, {K}], B[{K}, {N}]")

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    # Compute base GEMM
    C = torch.matmul(A, B).to(torch.float32)

    # Find absolute maximum per row (channel-wise)
    # C is [M, N], max_vals is [M, 1]
    max_vals = torch.amax(torch.abs(C), dim=1, keepdim=True)

    # Prevent division by zero
    max_vals = torch.clamp(max_vals, min=1e-7)

    # Calculate FP32 scaling factors
    scales = max_vals / 127.0

    # Quantize and cast to INT8
    # Using torch.round and clamping to valid int8 range to simulate hardware casting
    Y = torch.round(C / scales)
    Y = torch.clamp(Y, min=-128, max=127).to(torch.int8)

    print(f"Quantized Y dtype: {Y.dtype}, shape: {Y.shape}")
    print(f"Scales dtype: {scales.dtype}, shape: {scales.shape}\n")
    return Y, scales

if __name__ == "__main__":
    # Fallback to CPU if CUDA is not available in the testing environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running baselines on: {device.upper()}\n")

    # Seed for reproducibility
    torch.manual_seed(42)

    # Run the tests
    test_tile_fused_dual_gemm(device=device)
    test_stable_softplus_epilogue(device=device)
    test_dynamic_quant_epilogue(device=device)

    print("All baseline forward passes completed successfully.")
