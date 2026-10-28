from contextlib import nullcontext

import pytest
import torch

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.jit_compiler import JITCompiler, ShapeSpecializingJITCompiler


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
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())


def test_jit_compiler_caches_by_shape(monkeypatch):
    compiled = []
    template = _template()

    def compile_fn(module):
        compiled.append(module)
        return _Launcher()

    compiler = ShapeSpecializingJITCompiler(
        template,
        lambda module: module,
    )
    monkeypatch.setattr(compiler, "_compile", compile_fn)
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
        lambda module: module,
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
        _template(),
        lambda module: module,
    )
    with pytest.raises(ValueError, match="contiguous"):
        compiler._metadata((Tensor(), Tensor(), Tensor()))


def _static_module(name="static_dispatch_test") -> ModuleOp:
    return gpu_dialect.parse_module(f"""
        module {{
          kernel @{name}(%src: gmem<4x8xbf16>, %dst: gmem<4x8xbf16>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {{
          }}
        }}
    """)


def test_static_jit_compiler_is_lazy_and_preserves_the_module(monkeypatch):
    from ayla import jit_compiler

    module = _static_module()
    original = repr(module)
    compiled = []
    launcher = _Launcher()
    loaded = []
    launcher.ensure_loaded = lambda: loaded.append(True)

    def lower(cloned):
        compiled.append(cloned)
        cloned.body.ops.clear()
        return object()

    monkeypatch.setattr(jit_compiler, "make_ptx_kernel", lower)
    monkeypatch.setattr(jit_compiler, "make_ptx_launcher", lambda kernel: launcher)
    compiler = JITCompiler(module)
    _patch_metadata(monkeypatch, compiler)
    assert compiler.name == "static_dispatch_test"
    assert len(compiler.signature) == 2
    assert compiler.cache_size == 0
    assert compiled == []

    compiler(_Tensor(), _Tensor())
    compiler(_Tensor(), _Tensor())
    assert len(compiled) == 1
    assert loaded == [True]
    assert launcher.launches == 2
    assert compiler.cache_size == 1
    assert repr(module) == original


def test_static_jit_compiler_caches_per_device(monkeypatch):
    compiler = JITCompiler(_static_module("device_cache_test"))
    _patch_metadata(monkeypatch, compiler)
    compiled = []

    def compile_fn(module):
        launcher = _Launcher()
        compiled.append(launcher)
        return launcher

    monkeypatch.setattr(compiler, "_compile", compile_fn)
    for device in (0, 1, 0):
        monkeypatch.setattr(
            compiler,
            "_metadata",
            lambda tensors: (
                torch.device(f"cuda:{device}"),
                (((4, 8), (8, 1), torch.bfloat16),) * 2,
            ),
        )
        compiler(_Tensor(), _Tensor())
    assert len(compiled) == 2
    assert [launcher.launches for launcher in compiled] == [2, 1]


def test_static_jit_compiler_requires_warmup_before_capture(monkeypatch):
    compiler = JITCompiler(_static_module("static_capture_test"))
    _patch_metadata(monkeypatch, compiler)
    compiled = []
    launcher = _Launcher()

    def compile_fn(module):
        compiled.append(module)
        return launcher

    monkeypatch.setattr(compiler, "_compile", compile_fn)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        compiler(_Tensor(), _Tensor())
    assert compiled == []
    assert compiler.cache_size == 0

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiler(_Tensor(), _Tensor())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    compiler(_Tensor(), _Tensor())
    assert len(compiled) == 1
    assert launcher.launches == 2


def test_failed_compilation_is_not_cached(monkeypatch):
    compiler = JITCompiler(_static_module("failed_compile_test"))
    _patch_metadata(monkeypatch, compiler)

    def compile_fn(module):
        raise RuntimeError("compilation failed")

    monkeypatch.setattr(compiler, "_compile", compile_fn)
    with pytest.raises(RuntimeError, match="compilation failed"):
        compiler(_Tensor(), _Tensor())
    assert compiler.cache_size == 0


def test_jit_signature_excludes_launcher_tma_descriptors():
    from ayla.base.ir import Value
    from ayla.gpu_dialect import KernelOp, GmemTensorType, TmaDescType

    module = _static_module("tma_signature_test")
    kernel = module.body.ops[0]
    assert isinstance(kernel, KernelOp)
    gmem_type = kernel.signature[0]
    assert isinstance(gmem_type, GmemTensorType)
    kernel.body.append_argument(
        Value(
            name="desc",
            type=TmaDescType.build(
                arg_idx=0,
                gmem_type=gmem_type,
                tile_shape=[4, 8],
            ),
        )
    )
    compiler = JITCompiler(module)
    assert len(compiler.signature) == 2
    assert len(kernel.signature) == 3
