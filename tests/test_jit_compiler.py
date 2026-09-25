from typing import cast

import pytest
import torch

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.jit_compiler import ShapeSpecializingJITCompiler
from ayla.ptx_runtime import PtxLauncher


def _template(name="dispatch_test") -> ModuleOp:
    module = gpu_dialect.parse(f"""
        module {{
          kernel @{name}(%a: gmem<?x8xbf16>, %b: gmem<16x8xbf16>, %c: gmem<?x16xbf16>) {{
            matmul(%a: gmem<?x8xbf16>, %b: gmem<16x8xbf16>, %c: gmem<?x16xbf16>) collective=xyz.grid
          }}
        }}
    """)
    assert isinstance(module, ModuleOp)
    return module


def _shapes(m=4):
    return [(m, 8), (16, 8), (m, 16)]


class _Launcher:
    def __init__(self):
        self.launches = 0

    def __call__(self, *args):
        self.launches += 1


class _Tensor:
    device = torch.device("cuda")


def _patch_metadata(monkeypatch, compiler, m=4):
    key = ("test", m)
    monkeypatch.setattr(
        compiler, "_metadata", lambda tensors: (key, _shapes(m)),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_jit_compiler_caches_by_shape(monkeypatch):
    compiled: list[ModuleOp] = []
    template = _template()
    launcher = _Launcher()

    def compile_fn(module: ModuleOp) -> PtxLauncher:
        compiled.append(module)
        return cast(PtxLauncher, launcher)

    compiler = ShapeSpecializingJITCompiler(
        template,
        compile_fn,
    )
    _patch_metadata(monkeypatch, compiler)

    tensors = cast(
        tuple[torch.Tensor, ...], (_Tensor(), _Tensor(), _Tensor())
    )
    compiler(*tensors)
    compiler(*tensors)
    assert len(compiled) == 1
    assert compiler.cache_size == 1
    assert launcher.launches == 2
    assert "?" not in repr(compiled[0])
    assert "?" in repr(template)


def test_jit_compiler_rejects_capture_time_cache_miss(monkeypatch):
    compiler = ShapeSpecializingJITCompiler(
        _template("dispatch_capture_miss"),
        lambda module: cast(PtxLauncher, _Launcher()),
    )
    _patch_metadata(monkeypatch, compiler)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before capture"):
        compiler(*cast(tuple[torch.Tensor, ...], (_Tensor(), _Tensor(), _Tensor())))


def test_jit_compiler_rejects_noncontiguous_tensors():
    class Tensor:
        is_cuda = True
        device = torch.device("cuda")

        def is_contiguous(self):
            return False

    compiler = ShapeSpecializingJITCompiler(
        _template(), lambda module: cast(PtxLauncher, _Launcher()),
    )
    tensor = cast(torch.Tensor, Tensor())
    with pytest.raises(ValueError, match="contiguous"):
        compiler(tensor, tensor, tensor)
