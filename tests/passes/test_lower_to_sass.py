"""Machine lowering must preserve the actual scheduled GPU operation graph."""

from pathlib import Path
import os
import re

import pytest
import torch

from ayla import gpu_dialect as g
from ayla.base.ir import Block, Value, find_ops_of_type
from ayla.gemm_jit import (
    _module,
    build_gemm_module,
    gemm_jit_compiler,
    select_gemm_schedule,
)
from ayla.gemm_collection.gemm import schedule_matmul
from ayla.passes.lower_to_sass import lower_to_sass, make_sass_launcher
from ayla.sass_codegen import RegisterAllocator

REFERENCE = (
    Path(
        os.environ.get(
            "AYLA_SASS_DOCUMENT",
            Path(__file__).resolve().parents[2] / "nvidia-sass-document",
        )
    )
    / "sm90.json"
)
requires_assembler = pytest.mark.skipif(
    not REFERENCE.exists(), reason="generate the Hopper assembler DB first"
)
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires a Hopper GPU",
)
COLLECTIVE = {
    g.CollectiveAxis.X: g.CollectiveLevel.WG,
    g.CollectiveAxis.Y: g.CollectiveLevel.BLOCK,
    g.CollectiveAxis.Z: g.CollectiveLevel.GRID,
}
SHAPES = [
    ("mm", 1, 128, 128, 64),
    ("addmm", 1, 64, 1000, 128),
    ("bmm", 12, 64, 64, 448),
    ("mm", 1, 512, 768, 768),
    ("addmm", 1, 512, 768, 768),
    ("mm", 1, 512, 768, 3072),
    ("addmm", 1, 512, 768, 3072),
    ("mm", 1, 768, 768, 8192),
    ("mm", 1, 2048, 512, 128),
    ("mm", 1, 512, 3072, 768),
    ("addmm", 1, 3136, 3072, 1024),
    ("mm", 1, 30000, 128, 4096),
]


def scheduled(op="mm", dtype="float16", m=128, n=128, k=64, batch=1):
    # Hold the FP16 schedule fixed when testing BF16 conversion semantics.
    seed = select_gemm_schedule(
        op, "float16", m, n, k, *([batch] if op == "bmm" else [])
    )
    return build_gemm_module(op, dtype, batch, m, n, k, seed), seed


def run_and_check(module, op, dtype, m, n, k, batch=1, reference_fn=None):
    kernel = lower_to_sass(module)
    launcher = make_sass_launcher(kernel)
    dims = (batch,) if op == "bmm" else ()
    dt = getattr(torch, dtype)
    for seed in (7, 29, 20261103):
        torch.manual_seed(seed)
        a = torch.randn(*dims, m, k, device="cuda", dtype=dt)
        b = torch.randn(*dims, n, k, device="cuda", dtype=dt)
        bias = torch.randn(n, device="cuda", dtype=dt)
        out = torch.empty(*dims, m, n, device="cuda", dtype=dt)
        reference = (
            torch.addmm(bias, a, b.T) if op == "addmm" else a @ b.transpose(-1, -2)
        )
        if reference_fn is not None:
            # An independent CPU FP32 reference also avoids depending on
            # cuBLAS mixed-format state in the custom-IR semantic tests.
            reference = reference_fn(a.cpu(), b.cpu(), bias.cpu()).to(out.device)
        for _ in range(3):
            launcher(*([bias] if op == "addmm" else []), a, b, out)
            torch.testing.assert_close(
                out,
                reference,
                atol=0.25 if dtype == "bfloat16" else 0.125,
                rtol=0.01 if dtype == "bfloat16" else 0.002,
            )
    return kernel


@requires_assembler
def test_actual_ir_and_no_ptx_or_reconstruction(monkeypatch):
    module, seed = scheduled()
    before = str(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("PTX lowering or canonical reconstruction was invoked")

    import ayla.gemm_jit as jit
    import ayla.ptx_codegen as ptx
    import ayla.passes.lower_to_ptx_template as templates

    monkeypatch.setattr(jit, "_module", forbidden)
    monkeypatch.setattr(ptx, "make_ptx_kernel", forbidden)
    monkeypatch.setattr(templates.LowerToPtxTemplate, "run", forbidden)
    kernel = lower_to_sass(module)
    assert kernel.image.startswith(b"\x7fELF")
    assert kernel.dynamic_shared_mem_bytes == 0
    assert "matmul(" in kernel.lowered_ir and "ptx_template" not in kernel.lowered_ir
    assert str(module) == before
    assert lower_to_sass(module).image == kernel.image
    assert kernel.grid_dim == tuple(d or 1 for d in module.body.ops[0].grid_dim)


@requires_assembler
def test_modified_ir_is_accepted_and_changes_code():
    module, _ = scheduled()
    before = lower_to_sass(module).source
    mm = find_ops_of_type(module, g.MatmulOp)[0]
    mm.attrs["accumulate"] = True
    root = g.root_buffer(mm.dst)
    constant = g.ConstantOp.build(1.0, g.RmemTensorType.build([], g.F32Type()))
    kernel = module.body.ops[0]
    position = kernel.body.ops.index(root.definition) + 1
    kernel.body.ops[position:position] = [
        constant,
        g.ElementwiseOp.build(root, [constant.result()], "fill"),
    ]
    after = lower_to_sass(module).source
    assert before != after
    assert "1065353216" in after


@requires_assembler
def test_unsupported_operation_is_diagnosed():
    module, _ = scheduled()
    op = find_ops_of_type(module, g.ElementwiseOp)[-1]
    op.attrs["fn"] = "not_a_function"
    with pytest.raises(ValueError, match="unknown elementwise function"):
        lower_to_sass(module)


@requires_assembler
def test_encoding_cache_matches_reference():
    from ayla.passes.lower_to_sass import _assembler, _assemble_cached, _ASSEMBLER_LOCK

    module = schedule_matmul(
        _module("mm", "float16", 1, 64, 64, 64, False),
        output_m_tile=64,
        output_n_tile=64,
        warpgroup_n_tile=64,
        operand_stages=1,
        output_tma_inner_size=64,
    )
    kernel = lower_to_sass(module)
    assembler = _assembler()
    with _ASSEMBLER_LOCK, assembler._using_arch("sm90"):
        matcher, encoder = assembler._make_matcher_encoder()
        full = assembler._resolve_labels_and_encode(
            assembler.parse_sass(kernel.source), matcher, encoder
        )
        assert _assemble_cached(assembler, kernel.source) == full


def test_register_allocator_reuses_disjoint_groups_and_preserves_loop_values():
    ra = RegisterAllocator()
    carried = ra.new("R", 4, 4, "carried")
    a = ra.new("R", 4, 4, "first scratch")
    b = ra.new("R", 4, 4, "second scratch")
    lines = [
        f"MOV {carried[0]}, RZ",
        f"MOV {a[0]}, {carried[0]}",
        f"MOV {a[1]}, {a[0]}",
        "NOP",
        "NOP",
        f"MOV {b[0]}, RZ",
        f"MOV {b[1]}, {b[0]}",
        "BRA loop",
    ]
    ra.loops = [(1, 7)]
    ra.allocate(lines)
    assert carried.last >= 8
    assert a.offset == b.offset
    assert a.offset != carried.offset
    assert all(x.offset % 4 == 0 for x in (carried, a, b))


def test_register_pressure_is_diagnosed():
    ra = RegisterAllocator()
    blocks = [ra.new("UR", 32, 4) for _ in range(2)]
    with pytest.raises(ValueError, match="register pressure"):
        ra.allocate([f"MOV {blocks[0][0]}, RZ", f"MOV {blocks[1][0]}, {blocks[0][0]}"])


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("op,batch,m,n,k", SHAPES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_oss_scheduled_ir(op, batch, m, n, k, dtype):
    module, _ = scheduled(op, dtype, m, n, k, batch)
    run_and_check(module, op, dtype, m, n, k, batch)


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("op", ["mm", "addmm", "bmm"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "nt,kt,stages,store",
    [
        (48, 64, 1, "scalar"),
        (48, 64, 9, "stmatrix"),
        (96, 64, 3, "stmatrix"),
        (128, 128, 2, "stmatrix"),
    ],
)
def test_tails_phase_reuse_and_layouts(op, dtype, nt, kt, stages, store):
    m, n, k, batch = 81, 256, (1152 if kt == 128 else 1088), 3
    module = schedule_matmul(
        _module(op, dtype, batch, m, n, k, False),
        output_m_tile=64,
        output_n_tile=nt,
        warpgroup_n_tile=nt,
        operand_k_tile=kt,
        operand_stages=stages,
        output_store=store,
        output_tma_inner_size=min(nt, 64),
    )
    run_and_check(module, op, dtype, m, n, k, batch)


@requires_assembler
@requires_hopper
def test_nonzero_initializer_changes_execution():
    module, _ = scheduled()
    mm = find_ops_of_type(module, g.MatmulOp)[0]
    root = g.root_buffer(mm.dst)
    mm.attrs["accumulate"] = True
    c = g.ConstantOp.build(1.0, g.RmemTensorType.build([], g.F32Type()))
    body = module.body.ops[0].body
    position = body.ops.index(root.definition) + 1
    body.ops[position:position] = [c, g.ElementwiseOp.build(root, [c.result()], "fill")]
    run_and_check(
        module,
        "mm",
        "float16",
        128,
        128,
        64,
        reference_fn=lambda a, b, bias: (a.float() @ b.float().T + 1).half(),
    )


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("fn", ["mul", "sigmoid", "silu"])
def test_scheduled_epilogue_is_lowered(fn):
    module, _ = scheduled()
    conversion = find_ops_of_type(module, g.ElementwiseOp)[-1]
    src = conversion.src(0)
    body = conversion.parent_block
    position = body.ops.index(conversion)
    if fn == "mul":
        c = g.ConstantOp.build(0.5, g.RmemTensorType.build([], g.F32Type()))
        temp = g.AllocOp.build_with_fresh_id(src.type)
        fill = g.ElementwiseOp.build(temp.result(), [c.result()], "fill")
        operation = g.ElementwiseOp.build(src, [src, temp.result()], "mul")
        body.ops[position:position] = [c, temp, fill, operation]
    else:
        body.ops.insert(position, g.ElementwiseOp.build(src, [src], fn))

    def reference(a, b, bias):
        x = a.float() @ b.float().T
        return (
            x * 0.5
            if fn == "mul"
            else torch.sigmoid(x)
            if fn == "sigmoid"
            else torch.nn.functional.silu(x)
        ).half()

    run_and_check(module, "mm", "float16", 128, 128, 64, reference_fn=reference)


@requires_assembler
@requires_hopper
def test_store_conversion_preserves_snapshot_before_source_overwrite():
    module, _ = scheduled()
    conversion = find_ops_of_type(module, g.ElementwiseOp)[-1]
    body = conversion.parent_block
    zero = g.ConstantOp.build(0.0, g.RmemTensorType.build([], g.F32Type()))
    position = body.ops.index(conversion) + 1
    body.ops[position:position] = [
        zero,
        g.ElementwiseOp.build(conversion.src(0), [zero.result()], "fill"),
    ]
    kernel = run_and_check(module, "mm", "float16", 128, 128, 64)
    assert kernel.optimization_stats["store_conversions_deferred"] == 0


@pytest.mark.parametrize("boundary", ["refill", "accumulator", "control_flow"])
def test_prefetched_matrix_waits_drain_before_observers_and_at_group_limit(boundary):
    from ayla.sass_codegen import relax_prefetched_matrix_waits

    lhs, rhs = [
        g.AllocOp.build_with_fresh_id(g.SmemTensorType.build([64, 16], g.F16Type()))
        for _ in range(2)
    ]
    acc = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 64], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    locks = [
        g.AllocOp.build_with_fresh_id(g.LockType.build("wgmma")) for _ in range(10)
    ]
    groups, waits = [], []
    for lock in locks:
        groups.append(
            g.AsyncOp.build(
                lock.result(),
                COLLECTIVE,
                Block(
                    ops=[
                        g.MatmulOp.build(
                            lhs.result(),
                            rhs.result(),
                            acc.result(),
                            COLLECTIVE,
                            {"accumulate": True},
                        )
                    ]
                ),
            )
        )
        waits.append(g.AwaitOp.build(lock.result()))
    refill_lock = g.AllocOp.build_with_fresh_id(g.LockType.build("mbar"))
    zero = g.ConstantOp.build(0.0, g.RmemTensorType.build([], g.F32Type()))
    stop = {
        "refill": g.AsyncOp.build(refill_lock.result(), COLLECTIVE, Block()),
        "accumulator": g.ElementwiseOp.build(acc.result(), [zero.result()], "fill"),
        "control_flow": g.ForOp.build(0, 2),
    }[boundary]
    operations = [lhs, rhs, acc, *locks, refill_lock, zero]
    for group, wait in zip(groups[:9], waits[:9]):
        operations.extend((group, wait))
    operations.extend((stop, groups[9], waits[9]))
    kernel = standalone(Block(ops=operations)).body.ops[0]
    assert relax_prefetched_matrix_waits(kernel) == 7
    assert [wait.pending_groups for wait in waits[:8]] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert waits[8].is_full_wait and waits[9].is_full_wait


@requires_assembler
@requires_hopper
def test_jit_geometry_and_graph_capture():
    compiler = gemm_jit_compiler("mm", "float16", backend="sass")
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty(128, 128, device="cuda", dtype=a.dtype)
    compiler(a, b, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiler(a, b, out)
    graph.replay()
    torch.testing.assert_close(out, a @ b.T, atol=0.125, rtol=0.002)


def standalone(body, name="standalone", grid=(1, 1, 1)):
    kernel = g.KernelOp.build(
        name, body, grid_dim=grid, block_dim=(128, 1, 1), num_warp_groups=1
    )
    return g.ModuleOp.build(Block(ops=[kernel]))


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_broadcast_widening_special_values(dtype):
    half = g.F16Type() if dtype == "float16" else g.BF16Type()
    src = Value("src", g.GmemTensorType.build([64, 48], half, strides=[0, 1]))
    out = Value("out", g.GmemTensorType.build([64, 48], g.F32Type()))
    packed = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build([64, 48], half, warps=4, layout="wgmma_layout_0")
    )
    wide = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    kernel = lower_to_sass(
        standalone(
            Block(
                arguments=[src, out],
                ops=[
                    packed,
                    wide,
                    g.MoveOp.build(packed.result(), src, collective=COLLECTIVE),
                    g.ElementwiseOp.build(wide.result(), [packed.result()], "convert"),
                    g.MoveOp.build(out, wide.result(), collective=COLLECTIVE),
                ],
            )
        )
    )
    # Include both signs of zero and the smallest subnormal, plus infinities
    # and signaling/quiet NaNs. Compare finite values bit for bit.
    bits = [
        0,
        0x8000,
        1,
        0x8001,
        0x3C00,
        0xBC00,
        0x7C00,
        0xFC00,
        0x7C01,
        0x7E00,
        0xFC01,
        0xFE00,
    ]
    if dtype == "bfloat16":
        bits = [
            0,
            0x8000,
            1,
            0x8001,
            0x3F80,
            0xBF80,
            0x7F80,
            0xFF80,
            0x7F81,
            0x7FC0,
            0xFF81,
            0xFFC0,
        ]
    row = torch.tensor(bits * 4, dtype=torch.uint16).view(getattr(torch, dtype))
    input_tensor = row.to("cuda").expand(64, 48)
    result = torch.empty(64, 48, dtype=torch.float32, device="cuda")
    make_sass_launcher(kernel)(input_tensor, result)
    reference = row.float().expand(64, 48)
    actual = result.cpu()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(reference)
    assert torch.equal(
        actual.view(torch.int32)[finite],
        reference.contiguous().view(torch.int32)[finite],
    )


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_broadcast_widening_all_16bit_patterns(dtype):
    columns = ((65536 + 47) // 48) * 48
    half = g.F16Type() if dtype == "float16" else g.BF16Type()
    src = Value("src", g.GmemTensorType.build([64, columns], half, strides=[0, 1]))
    out = Value("out", g.GmemTensorType.build([64, columns], g.F32Type()))
    bx = g.BlockIdOp.build(g.CollectiveAxis.X)
    src_tile = g.TileOp.build(
        src, [0, bx.result()], g.GmemTensorType.build([64, 48], half, strides=[0, 1])
    )
    out_tile = g.TileOp.build(
        out,
        [0, bx.result()],
        g.GmemTensorType.build([64, 48], g.F32Type(), strides=[columns, 1]),
    )
    packed = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build([64, 48], half, warps=4, layout="wgmma_layout_0")
    )
    wide = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    body = Block(
        arguments=[src, out],
        ops=[
            bx,
            src_tile,
            out_tile,
            packed,
            wide,
            g.MoveOp.build(packed.result(), src_tile.result(), collective=COLLECTIVE),
            g.ElementwiseOp.build(wide.result(), [packed.result()], "convert"),
            g.MoveOp.build(out_tile.result(), wide.result(), collective=COLLECTIVE),
        ],
    )
    kernel = lower_to_sass(standalone(body, grid=(columns // 48, 1, 1)))
    bits = torch.arange(columns, dtype=torch.int32).to(torch.uint16)
    row = bits.view(getattr(torch, dtype))
    result = torch.empty(64, columns, device="cuda", dtype=torch.float32)
    make_sass_launcher(kernel)(row.cuda().expand(64, columns), result)
    reference = row.float().expand(64, columns).contiguous()
    actual = result.cpu()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(reference)
    assert torch.equal(
        actual.view(torch.int32)[finite], reference.view(torch.int32)[finite]
    )


@requires_assembler
@requires_hopper
def test_non_gemm_loop_kernel():
    out = Value("out", g.GmemTensorType.build([64, 48], g.F32Type()))
    fragment = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    factor = g.AllocOp.build_with_fresh_id(fragment.result().type)
    three = g.ConstantOp.build(3.0, g.RmemTensorType.build([], g.F32Type()))
    two = g.ConstantOp.build(2.0, g.RmemTensorType.build([], g.F32Type()))
    loop = g.ForOp.build(0, 5)
    loop.body.ops.append(
        g.ElementwiseOp.build(
            fragment.result(), [fragment.result(), factor.result()], "mul"
        )
    )
    body = Block(
        arguments=[out],
        ops=[
            fragment,
            factor,
            three,
            two,
            g.ElementwiseOp.build(fragment.result(), [three.result()], "fill"),
            g.ElementwiseOp.build(factor.result(), [two.result()], "fill"),
            loop,
            g.MoveOp.build(out, fragment.result(), collective=COLLECTIVE),
        ],
    )
    kernel = lower_to_sass(standalone(body))
    result = torch.empty(64, 48, device="cuda", dtype=torch.float32)
    make_sass_launcher(kernel)(result)
    torch.testing.assert_close(result, torch.full_like(result, 96.0))
    assert "HGMMA" not in kernel.source and "FMUL" in kernel.source


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("fn", ["sigmoid", "silu"])
@pytest.mark.parametrize("in_place", [False, True])
def test_function_overlap_preserves_independent_values(fn, in_place):
    typ = g.WgrmemTensorType.build(
        [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
    )
    src = Value("src", g.GmemTensorType.build([64, 48], g.F32Type()))
    out = Value("out", src.type)
    loaded = g.AllocOp.build_with_fresh_id(typ)
    target = loaded if in_place else g.AllocOp.build_with_fresh_id(typ)
    ops = [loaded] if in_place else [loaded, target]
    ops += [
        g.MoveOp.build(loaded.result(), src, collective=COLLECTIVE),
        g.ElementwiseOp.build(target.result(), [loaded.result()], fn),
        g.MoveOp.build(out, target.result(), collective=COLLECTIVE),
    ]
    kernel = lower_to_sass(standalone(Block(arguments=[src, out], ops=ops)))
    values = torch.linspace(-90, 90, 48).repeat(64, 1)
    # Include zeros, infinities and NaNs among otherwise distinct inputs.
    values[:, :6] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0]
    )
    result = torch.empty_like(values, device="cuda")
    make_sass_launcher(kernel)(values.cuda(), result)
    expected = (
        torch.sigmoid(values) if fn == "sigmoid" else torch.nn.functional.silu(values)
    )
    if fn == "silu":
        # Existing EX2/RCP/MUL lowering evaluates +inf * 1 as +inf.
        expected[:, 2] = float("inf")
    torch.testing.assert_close(
        result.cpu(), expected, atol=2e-6, rtol=2e-5, equal_nan=True
    )


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("divisor", [3, 7, 10, 31, 37, 257, 2**31 - 1, 2**32 - 1])
@pytest.mark.parametrize("backend", ["ptx", "sass"])
@pytest.mark.parametrize("operation", ["div", "mod"])
def test_index_division_and_conditionals(divisor, operation, backend):
    blocks = 17
    out = Value("out", g.GmemTensorType.build([blocks * 64, 48], g.F32Type()))
    fragment = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    bx = g.BlockIdOp.build(g.CollectiveAxis.X)
    multiplier = g.IndexConstantOp.build(0x9E3779B9)
    numerator = g.IndexMulOp.build(bx.result(), multiplier.result())
    d = g.IndexConstantOp.build(divisor)
    arithmetic = g.IndexDivOp if operation == "div" else g.IndexModOp
    q = arithmetic.build(numerator.result(), d.result())
    threshold = (
        max(1, 2**32 // (2 * divisor)) if operation == "div" else max(1, divisor // 2)
    )
    limit = g.IndexConstantOp.build(threshold)
    predicate = g.IndexLtOp.build(q.result(), limit.result())
    zero = g.ConstantOp.build(0.0, g.RmemTensorType.build([], g.F32Type()))
    one = g.ConstantOp.build(1.0, g.RmemTensorType.build([], g.F32Type()))
    condition = g.IfOp.build(
        predicate.result(),
        Block(ops=[g.ElementwiseOp.build(fragment.result(), [one.result()], "fill")]),
    )
    tile = g.TileOp.build(
        out, [bx.result(), 0], g.GmemTensorType.build([64, 48], g.F32Type())
    )
    body = Block(
        arguments=[out],
        ops=[
            fragment,
            bx,
            multiplier,
            numerator,
            d,
            q,
            limit,
            predicate,
            zero,
            one,
            g.ElementwiseOp.build(fragment.result(), [zero.result()], "fill"),
            condition,
            tile,
            g.MoveOp.build(tile.result(), fragment.result(), collective=COLLECTIVE),
        ],
    )
    module = standalone(body, grid=(blocks, 1, 1))
    if backend == "sass":
        launcher = make_sass_launcher(lower_to_sass(module))
    else:
        from ayla.ptx_codegen import make_ptx_kernel
        from ayla.ptx_runtime import assemble_ptx, make_ptx_launcher

        kernel = make_ptx_kernel(module)
        launcher = make_ptx_launcher(
            kernel, assemble_ptx(kernel, "/usr/local/cuda-13/bin/ptxas")
        )
    result = torch.empty(blocks * 64, 48, device="cuda", dtype=torch.float32)
    launcher(result)
    reference = torch.empty_like(result)
    for i in range(blocks):
        numerator_value = (i * 0x9E3779B9) & 0xFFFFFFFF
        value = (
            numerator_value // divisor
            if operation == "div"
            else numerator_value % divisor
        )
        reference[i * 64 : (i + 1) * 64] = float(value < threshold)
    torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("pending", [0, 1])
def test_shared_storage_waits_for_async_readers(pending):
    from ayla.sass_codegen import allocate_shared

    a, b, rhs, scratch = [
        g.AllocOp.build_with_fresh_id(g.SmemTensorType.build([64, 16], g.F16Type()))
        for _ in range(4)
    ]
    acc = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 64], g.F32Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    source = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 16], g.F16Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    lock = g.AllocOp.build_with_fresh_id(g.LockType.build("wgmma"))
    groups = [
        g.AsyncOp.build(
            lock.result(),
            COLLECTIVE,
            Block(
                ops=[
                    g.MatmulOp.build(
                        lhs.result(), rhs.result(), acc.result(), COLLECTIVE
                    )
                ]
            ),
        )
        for lhs in (a, b)
    ]
    body = Block(
        ops=[
            a,
            b,
            rhs,
            scratch,
            acc,
            source,
            lock,
            *groups,
            g.AwaitOp.build(lock.result(), {"pending": pending}),
            g.MoveOp.build(scratch.result(), source.result(), collective=COLLECTIVE),
            g.BarrierOp.build(),
            g.AwaitOp.build(lock.result()),
        ]
    )
    # The planner must account for storage
    # written after a partial wait while a newer matrix group still reads.
    kernel = standalone(body).body.ops[0]
    locations, _, lives = allocate_shared(kernel)

    def overlap(x, y):
        return (
            locations[x] < locations[y] + lives[y].size
            and locations[y] < locations[x] + lives[x].size
        )

    assert locations[scratch.result()] == locations[a.result()]
    if pending:
        assert not overlap(scratch.result(), b.result())
        assert not overlap(scratch.result(), rhs.result())
    else:
        assert lives[b.result()].last < lives[scratch.result()].first


@requires_assembler
def test_store_preparation_runs_before_matrix_completion(monkeypatch):
    from ayla.sass_codegen import SassCodegen

    module, _ = scheduled("mm", m=512, n=768, k=3072)
    optimized = lower_to_sass(module).source
    monkeypatch.setattr(SassCodegen, "redundant_refill_barrier", lambda *args: False)
    conservative = lower_to_sass(module).source
    assert optimized.count("BAR.SYNC") < conservative.count("BAR.SYNC")
    assert optimized.count("HGMMA.") == conservative.count("HGMMA.")
    # Invariant lane/address work happens while the first GMMA is outstanding.
    store_module, _ = scheduled()
    store_source = lower_to_sass(store_module).source
    first_matrix = store_source.index("HGMMA.")
    first_wait = store_source.index("WARPGROUP.DEPBAR")
    assert "SHF.R.U32.HI" in store_source[first_matrix:first_wait]


@requires_assembler
@requires_hopper
@pytest.mark.parametrize("value", [1.01171875, -1.01171875])
def test_bfloat16_fill_rounds_ties_to_even(value):
    typ = g.GmemTensorType.build([64, 48], g.BF16Type())
    out = Value("out", typ)
    fragment = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 48], g.BF16Type(), warps=4, layout="wgmma_layout_0"
        )
    )
    constant = g.ConstantOp.build(value, g.RmemTensorType.build([], g.BF16Type()))
    body = Block(
        arguments=[out],
        ops=[
            fragment,
            constant,
            g.ElementwiseOp.build(fragment.result(), [constant.result()], "fill"),
            g.MoveOp.build(out, fragment.result(), collective=COLLECTIVE),
        ],
    )
    result = torch.empty(64, 48, device="cuda", dtype=torch.bfloat16)
    make_sass_launcher(lower_to_sass(standalone(body)))(result)
    torch.testing.assert_close(result, torch.full_like(result, value), rtol=0, atol=0)


def test_machine_liveness_preserves_loop_carried_values():
    from ayla.sass_codegen import eliminate_dead_code

    lines = [
        "MOV32I R1000, 0;[7:7:{}:1:0]",
        "#def_label(loop)",
        "IADD3 R1000, R1000, 1, RZ;[7:7:{}:1:0]",
        "MOV32I R1001, 42;[7:7:{}:1:0]",
        "ISETP.LT.U32.AND P1002, PT, R1000, 4, PT;[7:7:{}:1:0]",
        "@P1002 BRA #label(loop);[7:7:{}:1:0]",
        "STG.E desc[{UR0,UR1}][{R2,R3}], R1000;[7:1:{}:2:0]",
        "EXIT;[7:7:{1}:8:0]",
    ]
    result, _ = eliminate_dead_code(lines)
    assert lines[0] in result and lines[2] in result
    assert lines[3] not in result


def test_machine_liveness_keeps_predicated_bypass_value_and_implicit_operands():
    from ayla.sass_codegen import eliminate_dead_code

    lines = [
        "UMOV UR1000, 0;[7:7:{}:1:0]",
        "@P0 UMOV UR1000, 64;[7:7:{}:1:0]",
        "UMOV UR1001, 16;[7:7:{}:1:0]",
        "UMOV UR1002, 128;[7:7:{}:1:0]",
        "UMOV UR1003, 192;[7:7:{}:1:0]",
        "UMOV UR1004, 256;[7:7:{}:1:0]",
        "UMOV UR1005, 0;[7:7:{}:1:0]",
        "UTMALDG.2D [UR1000], [UR1004];[7:1:{}:2:0]",
        "EXIT;[7:7:{1}:8:0]",
    ]
    result, _ = eliminate_dead_code(lines)
    assert result == lines


@requires_assembler
@requires_hopper
def test_loop_index_reuse_updates_each_iteration():
    out = Value("out", g.GmemTensorType.build([64, 48], g.F32Type()))
    typ = g.WgrmemTensorType.build(
        [64, 48], g.F32Type(), warps=4, layout="wgmma_layout_0"
    )
    fragment = g.AllocOp.build_with_fresh_id(typ)
    factor = g.AllocOp.build_with_fresh_id(typ)
    one = g.ConstantOp.build(1.0, g.RmemTensorType.build([], g.F32Type()))
    two = g.ConstantOp.build(2.0, g.RmemTensorType.build([], g.F32Type()))
    scale = g.IndexConstantOp.build(64)
    cutoff = g.IndexConstantOp.build(128)
    loop = g.ForOp.build(0, 3)
    index = g.IndexMulOp.build(loop.iv, scale.result())
    predicate = g.IndexLtOp.build(index.result(), cutoff.result())
    conditional = g.IfOp.build(
        predicate.result(),
        Block(
            ops=[
                g.ElementwiseOp.build(
                    fragment.result(), [fragment.result(), factor.result()], "mul"
                )
            ]
        ),
    )
    loop.body.ops = [index, predicate, conditional]
    body = Block(
        arguments=[out],
        ops=[
            fragment,
            factor,
            one,
            two,
            scale,
            cutoff,
            g.ElementwiseOp.build(fragment.result(), [one.result()], "fill"),
            g.ElementwiseOp.build(factor.result(), [two.result()], "fill"),
            loop,
            g.MoveOp.build(out, fragment.result(), collective=COLLECTIVE),
        ],
    )
    result = torch.empty(64, 48, device="cuda", dtype=torch.float32)
    make_sass_launcher(lower_to_sass(standalone(body)))(result)
    torch.testing.assert_close(result, torch.full_like(result, 4.0), rtol=0, atol=0)





def stream_fragment(rows, columns, dtype, fn, fill_value=0.0):
    dt = {"float32": g.F32Type, "float16": g.F16Type, "bfloat16": g.BF16Type}[dtype]()
    typ = g.GmemTensorType.build([rows, columns], dt)
    src, out = Value("src", typ), Value("out", typ)
    bx = g.BlockIdOp.build(g.CollectiveAxis.X)
    dst = g.TileOp.build(
        out, [bx.result(), 0], g.GmemTensorType.build([64, columns], dt)
    )
    fragment = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build([64, columns], dt, warps=4, layout="wgmma_layout_0")
    )
    ops = [bx, dst, fragment]
    if fn == "fill":
        value = g.ConstantOp.build(fill_value, g.RmemTensorType.build([], dt))
        ops += [
            value,
            g.ElementwiseOp.build(fragment.result(), [value.result()], "fill"),
        ]
    else:
        view = g.TileOp.build(
            src, [bx.result(), 0], g.GmemTensorType.build([64, columns], dt)
        )
        ops += [
            view,
            g.MoveOp.build(fragment.result(), view.result(), collective=COLLECTIVE),
        ]
        if fn != "copy":
            ops += [g.ElementwiseOp.build(fragment.result(), [fragment.result()], fn)]
    ops += [g.MoveOp.build(dst.result(), fragment.result(), collective=COLLECTIVE)]
    return standalone(
        Block(arguments=[out] if fn == "fill" else [src, out], ops=ops),
        grid=((rows + 63) // 64, 1, 1),
    )


@requires_hopper
@requires_assembler
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("columns", [48, 128])
@pytest.mark.parametrize(
    "fn,value",
    [("copy", 0.0), ("fill", 0.0), ("fill", -0.0), ("fill", 1.0), ("fill", -1.0)],
)
def test_streaming_tails_preserve_bits_and_output_canaries(dtype, columns, fn, value):
    rows = 520
    kernel = lower_to_sass(stream_fragment(rows, columns, dtype, fn, value))
    dt = getattr(torch, dtype)
    src = (
        torch.linspace(-2, 2, rows * columns, device="cuda")
        .reshape(rows, columns)
        .to(dt)
    )
    src.flatten()[:6] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), torch.finfo(dt).tiny],
        device="cuda",
        dtype=dt,
    )
    storage = torch.full((rows + 64, columns), 7.5, device="cuda", dtype=dt)
    expected = src if fn == "copy" else torch.full_like(src, value)
    launcher = make_sass_launcher(kernel)
    for _ in range(3):
        launcher(*([src] if fn == "copy" else []), storage[:rows])
        torch.cuda.synchronize()
        bits = torch.int32 if dtype == "float32" else torch.int16
        assert torch.equal(storage[:rows].view(bits), expected.view(bits))
        assert torch.all(storage[rows:] == 7.5)
    if fn == "fill":
        assert bool(re.search(r"STG[^;]*, RZ;", kernel.source)) == (
            value == 0.0 and not torch.signbit(torch.tensor(value))
        )


@requires_hopper
@requires_assembler
@pytest.mark.parametrize("fn", ["sigmoid", "silu"])
def test_immediate_activation_constants_keep_special_values(fn):
    rows, columns = 520, 48
    kernel = lower_to_sass(stream_fragment(rows, columns, "float32", fn))
    values = torch.tensor(
        [0.0, -0.0, -100.0, 100.0, -float("inf"), float("inf"), float("nan")]
    )
    src = (
        values.repeat((rows * columns + len(values) - 1) // len(values))[
            : rows * columns
        ]
        .reshape(rows, columns)
        .cuda()
    )
    storage = torch.full((rows + 64, columns), 7.5, device="cuda")
    reference = torch.sigmoid(src.cpu())
    if fn == "silu":
        reference *= src.cpu()
    make_sass_launcher(kernel)(src, storage[:rows])
    torch.cuda.synchronize()
    torch.testing.assert_close(
        storage[:rows].cpu(), reference, rtol=2e-5, atol=2e-6, equal_nan=True
    )
    assert torch.all(storage[rows:] == 7.5)


@requires_assembler
def test_large_memory_displacement_retains_address_fallback():
    module = stream_fragment(64, 48, "float32", "copy")
    for value in module.body.ops[0].body.arguments:
        value.type = g.GmemTensorType.build([64, 48], g.F32Type(), strides=[1 << 22, 1])
    for tile in find_ops_of_type(module, g.TileOp):
        tile.result().type = g.GmemTensorType.build(
            [64, 48], g.F32Type(), strides=[1 << 22, 1]
        )
    kernel = lower_to_sass(module)
    displacements = re.findall(r"\]\[\{R\d+,R\d+\}\+(-?\d+)\]", kernel.source)
    assert displacements and all(
        -(1 << 23) <= int(n) < (1 << 23) for n in displacements
    )
    assert kernel.source.count("IMAD.WIDE.U32") > 2


def test_nested_shared_tiles_use_root_physical_strides():
    from ayla.sass_codegen import Memory, SassCodegen

    attrs = {"physical_layout": {"block_width": 64, "order": "block_major"}}
    root = g.AllocOp.build_with_fresh_id(
        g.SmemTensorType.build([128, 128], g.BF16Type(), attrs=attrs)
    )
    rows = g.TileOp.build(
        root.result(), [1, 0],
        g.SmemTensorType.build([64, 128], g.BF16Type(), strides=[128, 1], attrs=attrs),
    )
    inner = g.TileOp.build(
        rows.result(), [0, 5],
        g.SmemTensorType.build([64, 16], g.BF16Type(), strides=[128, 1], attrs=attrs),
    )
    kernel = standalone(Block(ops=[root, rows, inner])).body.ops[0]
    codegen = SassCodegen(kernel)
    codegen.values[root.result()] = Memory(root.result())
    codegen.tile(rows)
    codegen.tile(inner)
    # Crossing a column block advances over all 128 root rows, including
    # those outside the 64-row view.
    assert codegen.values[inner.result()].offset == 64 * 64 * 2 + 128 * 64 * 2 + 16 * 2
    assert codegen.values[inner.result()].origin == (64, 80)


def test_contiguous_placement_preserves_alignment_reservations_and_boundary_uses():
    from ayla.sass_codegen import LiveRange, place_live_ranges

    first = LiveRange(size=4, alignment=4, first=0, last=2)
    boundary = LiveRange(size=2, alignment=2, first=2, last=3)
    reused = LiveRange(size=4, alignment=4, first=3, last=4)
    unused = LiveRange(size=128, alignment=128)
    assert place_live_ranges([first, boundary, reused, unused], reserved=((1, 1),)) == 8
    assert (first.offset, boundary.offset, reused.offset) == (4, 2, 4)
    # Registers are identity-bearing allocations even with matching attributes.
    assert len({first, LiveRange(size=4, alignment=4, first=0, last=2)}) == 2
