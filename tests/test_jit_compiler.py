import pytest
import torch

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.jit_compiler import ShapeSpecializingJITCompiler


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
    signature = tuple(
        (shape, tuple(reversed(range(1, len(shape) + 1))), torch.bfloat16)
        for shape in _shapes(m)
    )
    monkeypatch.setattr(
        compiler,
        "_metadata",
        lambda tensors: (torch.device("cuda"), signature),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_jit_compiler_caches_by_shape(monkeypatch):
    compiled = []
    template = _template()

    def compile_fn(module):
        compiled.append(module)
        return _Launcher()

    compiler = ShapeSpecializingJITCompiler(
        template,
        compile_fn,
    )
    _patch_metadata(monkeypatch, compiler)

    tensors = (_Tensor(), _Tensor(), _Tensor())
    compiler(*tensors)
    compiler(*tensors)
    assert len(compiled) == 1
    compiled_launcher = compiler._launcher(tensors)
    assert compiled_launcher.launches == 2
    assert "?" not in repr(compiled[0])
    assert "?" in repr(template)


def test_jit_compiler_rejects_capture_time_cache_miss(monkeypatch):
    compiler = ShapeSpecializingJITCompiler(
        _template("dispatch_capture_miss"),
        lambda module: _Launcher(),
    )
    _patch_metadata(monkeypatch, compiler)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before capture"):
        compiler(_Tensor(), _Tensor(), _Tensor())


def test_jit_compiler_rejects_noncontiguous_tensors():
    class Tensor:
        is_cuda = True
        device = torch.device("cuda")

        def is_contiguous(self):
            return False

    compiler = ShapeSpecializingJITCompiler(
        _template(), lambda module: _Launcher(),
    )
    with pytest.raises(ValueError, match="contiguous"):
        compiler(Tensor(), Tensor(), Tensor())
