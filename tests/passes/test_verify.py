import textwrap

import pytest

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.passes.verify_gpu_dialect import (
    VerificationStage,
    verify_gpu_dialect,
)
from ayla.ptx_codegen import make_ptx_kernel


def _dynamic_matmul_module() -> ModuleOp:
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @dynamic(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>) block_dim=[128, None, None] {
            matmul(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>)
          }
        }"""))
    assert isinstance(module, ModuleOp)
    return module


def test_default_verification_allows_dynamic_shapes():
    module = _dynamic_matmul_module()

    assert verify_gpu_dialect(module) is module


def test_default_verification_rejects_inconsistent_matmul_shapes():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%a: gmem<7x11xbf16>, %b: gmem<13x12xbf16>, %c: gmem<7x13xbf16>) {
            matmul(%a: gmem<7x11xbf16>, %b: gmem<13x12xbf16>, %c: gmem<7x13xbf16>)
          }
        }"""))

    with pytest.raises(ValueError, match="inconsistent matmul shapes"):
        verify_gpu_dialect(module)


def test_ptx_lowering_verification_rejects_dynamic_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        verify_gpu_dialect(
            _dynamic_matmul_module(), VerificationStage.PTX_LOWERING,
        )


def test_ptx_codegen_verifies_static_shapes():
    with pytest.raises(ValueError, match="unresolved dynamic shape"):
        make_ptx_kernel(_dynamic_matmul_module())


def test_verification_rejects_parsed_grid_barrier_without_sync():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            barrier() attrs={scope: "grid"}
          }
        }"""))

    with pytest.raises(ValueError, match="grid scope requires one sync operand"):
        verify_gpu_dialect(module)


def test_verification_rejects_undersized_grid_barrier_sync():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @invalid(%sync: gmem<1xf32>) {
            barrier(%sync: gmem<1xf32>) attrs={scope: "grid"}
          }
        }"""))

    with pytest.raises(ValueError, match="at least 8 bytes"):
        verify_gpu_dialect(module)
