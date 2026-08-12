"""Helpers for timing a replacement cubin through CuTe's original host ABI."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.export  # Registers the CuTe external-module load provider.
from cutlass.cute.runtime import from_dlpack
import torch


SOURCE = Path(
    "/home/arseniivanov/extrapolation/agents/cutlass/examples/python/"
    "CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py"
)
CUDA_ELF_MAGIC = b"\x7fELF\x02\x01\x01A"
FATBIN_MAGIC = b"\x50\xed\x55\xba\x01\x00\x10\x00"


def _load_kernel_class():
    # The module name participates in CuTe's generated kernel symbol.  Match
    # the reference wrapper exactly so the retained PTX/cubin remains ABI- and
    # symbol-compatible with this host object.
    spec = importlib.util.spec_from_file_location("cutlass_hopper_persistent", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HopperWgmmaGemmPersistentKernel


def _tensor(torch_tensor):
    value = from_dlpack(torch_tensor.unsqueeze(-1), assumed_align=16)
    value.element_type = cutlass.BFloat16
    return value.mark_layout_dynamic(leading_dim=1)


def prepare_patched(inputs, cubin_path: Path, prefix: str):
    """Compile CuTe's host wrapper, replace only its embedded cubin, and load it."""
    a, b = inputs["a"], inputs["b"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    m_a, m_b, m_c = _tensor(a), _tensor(b), _tensor(output)
    gemm = _load_kernel_class()(cutlass.Float32, (128, 256), (1, 1), 1, True)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, m_a, m_b, m_c, max_clusters, stream)

    replacement = cubin_path.read_bytes()
    host_object = bytearray(compiled.dump_to_object(prefix))
    if replacement[:8] != CUDA_ELF_MAGIC:
        raise RuntimeError(f"not an sm_90a CUDA cubin: {cubin_path}")
    # These comparison cubins are assembled from the PTX dumped by this exact
    # configuration and intentionally retain the original 24,048-byte image
    # size, so no host ELF offsets or fatbin metadata need to change.
    if len(replacement) != 24048:
        raise RuntimeError(f"unexpected replacement cubin size: {len(replacement)}")

    # Repackage instead of overwriting only the inner ELF.  CUDA validates
    # fatbin image metadata when cudaLibraryLoadData is used by CuTe's host
    # wrapper; a raw byte replacement produces cudaErrorSymbolNotFound even
    # though cuModuleLoad accepts the same cubin.
    with tempfile.TemporaryDirectory() as temporary:
        fatbin_path = Path(temporary) / "kernels.fatbin"
        subprocess.run([
            "/usr/local/cuda-13.0/bin/fatbinary",
            f"--create={fatbin_path}",
            "--64",
            "--ident=kernels",
            f"--image3=kind=elf,sm=90,file={cubin_path}",
        ], check=True)
        fatbin = fatbin_path.read_bytes()
    fatbin_offset = host_object.find(FATBIN_MAGIC)
    if fatbin_offset < 0:
        raise RuntimeError("CuTe export object does not contain a fatbin image")
    if len(fatbin) != 24136:
        raise RuntimeError(f"unexpected replacement fatbin size: {len(fatbin)}")
    host_object[fatbin_offset:fatbin_offset + len(fatbin)] = fatbin

    object_path = cubin_path.with_name(f"{prefix}_host.o")
    object_path.write_bytes(host_object)
    external_module = cutlass.runtime.load_module(str(object_path))
    patched = getattr(external_module, prefix)

    def run():
        patched(m_a, m_b, m_c, stream)
        return output

    # Keep both the loaded module and original compiler engine alive.
    run._owners = (external_module, compiled)
    return run
