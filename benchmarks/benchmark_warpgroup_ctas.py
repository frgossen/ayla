"""Compare generic warpgroup pairing against measured GEMM and cuBLAS schedules."""

import argparse
import ctypes
import itertools
import json
import math
from pathlib import Path
import statistics

import torch

from ayla.gemm_collection.gemm import auto_matmul_4096_wgmma256_wg_tma_store_module
from ayla.gemm_jit import _seed, build_gemm_module, select_gemm_schedule
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from benchmarks.common import _time_graph


def _driver_resources(run) -> dict[str, int]:
    run.ensure_loaded()
    assert run._function is not None
    cuda = ctypes.CDLL("libcuda.so.1")
    query = cuda.cuFuncGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    query.restype = ctypes.c_int
    resources = {}
    for name, attribute in (
        ("max_threads", 0),
        ("shared_bytes", 1),
        ("local_bytes", 3),
        ("registers", 4),
    ):
        value = ctypes.c_int()
        result = query(ctypes.byref(value), attribute, ctypes.c_void_p(run._function))
        if result:
            raise RuntimeError(f"cuFuncGetAttribute({attribute}) failed: {result}")
        resources[name] = value.value
    return resources


def capture(fn, launches):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(launches):
            fn()
    return graph, stream


def paired_config(m, n, stages, copies=2):
    columns = 168 if n == 11008 else 128
    tasks = math.ceil(m / 128) * math.ceil(n / columns)
    grid = max(g for g in range(2, min(264, tasks) + 1, 2) if tasks % g == 0)
    return dict(
        output_m_tile=128,
        warpgroup_m_tile=128,
        output_n_tile=columns,
        warpgroup_n_tile=columns,
        operand_k_tile=64,
        operand_stages=stages,
        cyclic_operand_pipeline=True,
        warpgroup_ctas=copies,
        output_grid_size=grid,
        output_grid_fast_axis="m",
        output_fragment_cols=64,
        output_fragment_swizzle="128B",
        output_store_buffers=1,
        output_tma_inner_size=None,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    parser.add_argument(
        "--shapes", nargs="+", choices=["wide", "square"], default=["wide", "square"]
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--launches", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Frozen preceding schedules")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["library", "unpaired4", "paired4", "paired5", "selected", "cublas"],
        default=["library", "unpaired4", "paired4", "paired5", "cublas"],
    )
    args = parser.parse_args()
    if not {"library", "cublas"} <= set(args.variants):
        parser.error("--variants must include both library and cublas controls")
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20261122)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    records = []
    for shape in args.shapes:
        m, n, k = (2048, 11008, 4096) if shape == "wide" else (4096, 4096, 4096)
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(n, k, device="cuda", dtype=dtype)
        reference = a.float() @ b.float().T
        seed = select_gemm_schedule("mm", args.dtype, m, n, k)
        selected_seed = seed
        if args.baseline:
            frozen = json.loads(args.baseline.read_text())[args.dtype][shape]
            source = tuple(frozen["source_shape"])
            seed = _seed(
                dict(frozen["config"]),
                "mm",
                m,
                n,
                k,
                (1, *source),
                exact=source == (m, n, k),
            )
        variants = []
        for name in args.variants:
            config = None
            if name in ("library", "selected"):
                chosen = seed if name == "library" else selected_seed
                module = build_gemm_module("mm", args.dtype, 1, m, n, k, chosen)
                config = chosen.captured_config
                out = torch.empty(m, n, device="cuda", dtype=dtype)
            else:
                out = torch.empty(m, n, device="cuda", dtype=dtype)
            if name == "cublas":

                def fn(out=out):
                    return torch.mm(a, b.T, out=out)

                resources = None
            else:
                if name not in ("library", "selected"):
                    config = paired_config(
                        m, n, int(name[-1]), 1 if name.startswith("unpaired") else 2
                    )
                    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
                        m=m,
                        n=n,
                        k=k,
                        dtype="f16" if args.dtype == "float16" else "bf16",
                        **config,
                    )
                kernel = make_ptx_kernel(module)
                (args.output / f"{shape}_{name}.ptx").write_text(kernel.ptx)
                run = make_ptx_launcher(kernel)
                resources = _driver_resources(run)
                resources.update(block_dim=kernel.block_dim, grid_dim=kernel.grid_dim)

                def fn(run=run, out=out):
                    return run(a, b, out)

            fn()
            torch.cuda.synchronize()
            relative_l2 = ((out.float() - reference).norm() / reference.norm()).item()
            assert torch.isfinite(out).all() and relative_l2 < (
                0.001 if dtype == torch.float16 else 0.005
            ), (name, relative_l2)
            variants.append(
                dict(
                    name=name,
                    graph=capture(fn, args.launches),
                    keep=fn,
                    record=dict(
                        name=name,
                        config=config,
                        resources=resources,
                        relative_l2=relative_l2,
                        samples=[],
                    ),
                )
            )
            print(args.dtype, shape, name, resources, relative_l2, flush=True)
        graphs = {v["name"]: v["graph"] for v in variants}
        for _ in range(20):
            for graph, _stream in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        orders = list(itertools.permutations(["candidate", "library", "cublas"]))
        for v in variants:
            v["record"].update(library_samples=[], cublas_samples=[])
        for round in range(args.rounds):
            for v in variants if round % 2 == 0 else variants[::-1]:
                for role in orders[round % len(orders)]:
                    name = v["name"] if role == "candidate" else role
                    graph, stream = graphs[name]
                    ms, _ = _time_graph(graph, stream, graph_launches=args.launches)
                    v["record"][
                        "samples" if role == "candidate" else role + "_samples"
                    ].append(ms * 1000)
        for v in variants:
            record = v["record"]
            record["median_us"] = statistics.median(record["samples"])
            for control in ["library", "cublas"]:
                record[control + "_over_variant"] = statistics.median(
                    c / t
                    for c, t in zip(record[control + "_samples"], record["samples"])
                )
            print(
                args.dtype,
                shape,
                record["name"],
                record["median_us"],
                record["library_over_variant"],
                record["cublas_over_variant"],
                flush=True,
            )
        records.append(dict(shape=[m, n, k], results=[v["record"] for v in variants]))
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                dtype=args.dtype,
                device=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                rounds=args.rounds,
                launches=args.launches,
                clocks="unlocked",
                workloads=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
