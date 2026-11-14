"""Optional SM probes for resident Ayla kernels; no workload-specific imports."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import cast

import numpy as np
import torch

import ayla.ptx_codegen as codegen
from ayla.base.ir import Op, Value, clone_op, find_ops_of_type
from ayla.gpu_dialect import (
    BarrierOp, F32Type, GmemTensorType, IfOp, ForOp, BlockIdOp,
    MatmulOp, PTXTemplateOp, KernelOp, ModuleOp, ShuffleOp,
)


def _phase_kind(ops):
    matmuls = [m for op in ops for m in find_ops_of_type(op, MatmulOp)]
    if matmuls:
        labels = [str(m.attrs.get("label", "")) for m in matmuls]
        return "swiglu" if (any(s.startswith("gate") for s in labels)
                             and any(s.startswith("up") for s in labels)) else "gemm"
    return "norm" if any(find_ops_of_type(op, ShuffleOp) for op in ops) else "map"


CAPACITY = 2048


READINESS_SNAPSHOT = 1 << 62


WORK_KINDS = {"down", "projection", "other GEMM", "pointwise / norm", "other work", "coefficient / row setup"}


KINDS = {
    1: "projection",
    3: "down",
    5: "grid / pool sync",
    7: "dependency wait",
    9: "other work",
    17: "counter publication",
    19: "coefficient / row setup",
    21: "scheduler metadata / dispatch",
}


COLORS = {
    "projection": "#3578bb",
    "down": "#c95460",
    "grid / pool sync": "#b6bec9",
    "dependency wait": "#e7a13b",
    "other work": "#65a69a",
    "other GEMM": "#7961a5",
    "pointwise / norm": "#65a69a",
    "counter publication": "#96a2bc",
    "readiness acquire / fence": "#d4d9df",
    "coefficient / row setup": "#a4d4c5",
    "scheduler metadata / dispatch": "#aeb5c0",
}


def _groups(body):
    groups, current = [], []
    for op in body.ops:
        current.append(op)
        if isinstance(op, BarrierOp) and op.scope == "grid":
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _tag_roots(source):
    kernel = source.body.ops[0]
    roots = sorted(
        (op for op in find_ops_of_type(kernel, IfOp) if "tile_root" in op.attrs),
        key=lambda op: cast(int, op.attrs["tile_root"]),
    )
    kinds = [_phase_kind(op.body.ops) for op in roots]
    swiglu = [cast(int, op.attrs["tile_root"]) for op, kind in zip(roots, kinds, strict=True) if kind == "swiglu"]
    down = max(swiglu) + 1 if swiglu else -1
    for op, kind in zip(roots, kinds, strict=True):
        op.attrs["sm_trace_kind"] = (
            "projection" if kind == "swiglu" else "down" if op.attrs["tile_root"] == down else "other work"
        )
        op.attrs["sm_trace_root"] = op.attrs["tile_root"]
    return list({op.attrs["tile_root"]: dict(root=op.attrs["tile_root"], kind=kind)
                 for op, kind in zip(roots, kinds, strict=True)}.values())


def _tag_resident_tasks(source):
    """Trace baseline output tiles/rows with the same granularity as static."""
    kernel = source.body.ops[0]
    if kernel.attrs.get("tile_scheduler") is not None:
        return []
    roots, phase_roots = [], []

    def block_dependent(value, seen=None):
        seen = set() if seen is None else seen
        if value in seen:
            return False
        seen.add(value)
        op = value.definition
        return isinstance(op, BlockIdOp) or (
            isinstance(op, Op) and any(block_dependent(v, seen) for v in op.operands)
        )

    for phase in _groups(kernel.body):
        phase_roots.append(len(roots))
        loops = [
            loop
            for op in phase
            for loop in find_ops_of_type(op, ForOp)
            if loop.attrs.get("tile_output_loop")
        ]
        regions = []
        if loops:
            for loop in loops:
                matmuls = set(find_ops_of_type(loop, MatmulOp))
                guards = [
                    op
                    for op in loop.body.ops
                    if isinstance(op, IfOp)
                    and (
                        set(find_ops_of_type(op, MatmulOp)) == matmuls
                        if matmuls
                        else block_dependent(op.condition)
                    )
                ]
                regions.append(
                    (
                        guards[0] if guards else loop,
                        _phase_kind(loop.body.ops),
                        loop.attrs.get("tile_task_count"),
                    )
                )
        else:
            matmuls = {m for op in phase for m in find_ops_of_type(op, MatmulOp)}
            guards = [
                guard
                for op in phase
                for guard in find_ops_of_type(op, IfOp)
                if matmuls
                and set(find_ops_of_type(guard, MatmulOp)) == matmuls
                and block_dependent(guard.condition)
            ]
            regions.append((guards[-1] if guards else None, _phase_kind(phase), None))
        root = len(roots)
        kind = _phase_kind(phase)
        for guard, _, _ in regions:
            if guard is not None:
                guard.attrs.update(
                    sm_trace_root=root,
                    sm_trace_kind="projection" if kind == "swiglu" else "other work",
                )
        counts = [count for _, _, count in regions if count is not None]
        roots.append(
            dict(root=root, kind=kind, task_count=max(counts) if counts else None)
        )
    swiglu = [r["root"] for r in roots if r["kind"] == "swiglu"]
    if swiglu:
        down = max(swiglu) + 1
        for op in [*find_ops_of_type(kernel, IfOp), *find_ops_of_type(kernel, ForOp)]:
            if op.attrs.get("sm_trace_root") == down:
                op.attrs["sm_trace_kind"] = "down"
    kernel.attrs["sm_trace_phase_roots"] = phase_roots
    kernel.attrs["sm_trace_granularity"] = "tile"
    return roots


def _instrument(source: ModuleOp, cached_lowered: ModuleOp | None = None, *, capacity=CAPACITY):
    """Insert probes after synchronization/layout passes; avoid rescheduling work."""
    source = cast(ModuleOp, clone_op(source, {}))
    source_kernel = cast(KernelOp, source.body.ops[0])
    kinds = [_phase_kind(group) for group in _groups(source_kernel.body)]
    swiglu = [i for i, kind in enumerate(kinds) if kind == "swiglu"]
    down_phase = max(swiglu) + 1 if swiglu else -1
    kinds = [
        "projection" if kind == "swiglu" else "down" if i == down_phase else "other"
        for i, kind in enumerate(kinds)
    ]
    root_kinds = {row["root"]: row["kind"] for row in _tag_roots(source)}
    root_kinds.update({row["root"]: row["kind"] for row in _tag_resident_tasks(source)})
    whole_graph = source_kernel.attrs.get("tile_order") in ("source", "memory")
    base_pipeline = codegen.PtxLoweringPipeline

    class Observer(base_pipeline):
        def run(self, module):
            lowered = module if cached_lowered is not None else super().run(module)
            kernel = cast(KernelOp, lowered.body.ops[0])
            trace = Value(
                "sm_trace",
                GmemTensorType.build([kernel.grid_dim[0], capacity + 1, 4], F32Type()),
            )
            kernel.body.append_argument(trace)

            def stamp(event):
                return f"""
@!%sm_trace_leader bra ${{label:skip}};
setp.ge.u32 %sm_trace_full, %sm_trace_count, {capacity};
@%sm_trace_full bra ${{label:skip}};
mov.u64 %sm_trace_clock, %globaltimer;
st.global.u64 [%sm_trace_cursor], %sm_trace_clock;
st.global.u64 [%sm_trace_cursor+8], {event};
add.u64 %sm_trace_cursor, %sm_trace_cursor, 16;
add.u32 %sm_trace_count, %sm_trace_count, 1;
st.global.u32 [%sm_trace_base+8], %sm_trace_count;
${{label:skip}}:
"""

            def marker(event):
                return PTXTemplateOp.build(
                    [], [], stamp(event), description=f"SM trace {event}"
                )

            def visit(block, within_tile=False):
                for op in list(block.ops):
                    if isinstance(op, (IfOp, ForOp)) and "sm_trace_kind" in op.attrs:
                        visit(op.body, True)
                        event = {"projection": 1, "down": 3}.get(
                            cast(str, op.attrs["sm_trace_kind"]), 9
                        )
                        if "sm_trace_root" in op.attrs:
                            event += 1000 * (cast(int, op.attrs["sm_trace_root"]) + 1)
                        if isinstance(op, IfOp) and op.else_body is not None:
                            visit(op.else_body)
                        publications = [
                            child
                            for child in op.body.ops
                            if isinstance(child, PTXTemplateOp)
                            and child.attrs.get("description") == "tile publish"
                        ]
                        if (
                            whole_graph
                            and op.attrs.get("tile_completion_scope") == "cta_root"
                        ):
                            waits = [
                                index
                                for index, child in enumerate(op.body.ops)
                                if isinstance(child, PTXTemplateOp)
                                and child.attrs.get("description") == "tile wait"
                            ]
                            op.body.ops.insert(
                                waits[-1] + 2 if waits else 0, marker(event)
                            )
                            op.body.ops.append(marker(event + 1))
                        elif whole_graph:
                            if (
                                len(publications) != 1
                                or publications[0] is not op.body.ops[-1]
                            ):
                                raise ValueError(
                                    "task must end with exactly one publication"
                                )
                            publication = publications[0]
                            pieces = publication.template.split("bar.sync 0;", 1)
                            if len(pieces) != 2:
                                raise ValueError(
                                    "publication must drain stores and join the CTA"
                                )
                            # Work ends after full store completion and the
                            # existing CTA join, before counter bookkeeping.
                            publication.attrs["template"] = (
                                pieces[0]
                                + "bar.sync 0;"
                                + _raw_stamp(event + 1)
                                + (
                                    _raw_stamp(17)
                                    if "red.release.gpu.global.add.u32" in pieces[1]
                                    else ""
                                )
                                + pieces[1]
                                + (
                                    _raw_stamp(18)
                                    if "red.release.gpu.global.add.u32" in pieces[1]
                                    else ""
                                )
                            )
                            # Native static roots acquire inside the tile
                            # guard. Work starts after readiness, including
                            # the wait-end probe inserted by visit above.
                            waits = [
                                index
                                for index, child in enumerate(op.body.ops)
                                if isinstance(child, PTXTemplateOp)
                                and child.attrs.get("description") == "tile wait"
                            ]
                            begin = waits[-1] + 2 if waits else 0
                            op.body.ops.insert(begin, marker(event))
                        else:
                            op.body.ops = [
                                marker(event),
                                *op.body.ops,
                                marker(event + 1),
                            ]
                    elif (
                        isinstance(op, PTXTemplateOp)
                        and op.attrs.get("description") in {
                            "load static readiness", "compute static readiness",
                            "load frontier publications", "load root publications",
                            "load static row bounds",
                            "load deferred lookahead readiness", "load static worker lane",
                        }
                        and whole_graph
                    ):
                        op.attrs["template"] = _raw_stamp(21) + op.template + _raw_stamp(22)
                    elif (
                        isinstance(op, PTXTemplateOp)
                        and op.attrs.get("description") == "tile publish"
                        and whole_graph
                        and not within_tile
                    ):
                        # GEMM cohorts finish their async stores and publish
                        # once after the root loop, outside individual tiles.
                        op.attrs["template"] = (
                            _raw_stamp(17) + op.template + _raw_stamp(18)
                        )
                    elif isinstance(op, PTXTemplateOp) and op.attrs.get(
                        "description"
                    ) in ("tile wait", "tile wait_pair"):
                        event = (
                            5
                            if op.attrs["description"] == "tile wait"
                            and not whole_graph
                            else 7
                        )
                        if "readiness_root" in op.attrs:
                            event += 1000 * (cast(int, op.attrs["readiness_root"]) + 1)
                        # Capture the first acquired counter value, so ready
                        # checks can be distinguished from actual input stalls.
                        for name in (
                            ["event"]
                            if op.attrs["description"] == "tile wait"
                            else ["first", "last"]
                        ):
                            seen_load = "ld.acquire.gpu.global.u32 ${b32:seen}, [${b64:counter}];"
                            if seen_load not in op.template:
                                break
                            expected = (
                                "${b32:operand.expected.val}"
                                if name == "event"
                                else "1"
                            )
                            root = cast(int, op.attrs.get("readiness_root", 255))
                            sample = (
                                f"""
                            @${{pred:sampled_{name}}} bra ${{label:sample_done_{name}}};
                            cvt.u64.u32 ${{b64:sample_event_{name}}}, ${{b32:operand.{name}.val}};
                            shl.b64 ${{b64:sample_packet_{name}}}, ${{b64:sample_event_{name}}}, 32;
                            cvt.u64.u32 ${{b64:sample_expected_{name}}}, {expected};
                            shl.b64 ${{b64:sample_expected_{name}}}, ${{b64:sample_expected_{name}}}, 16;
                            or.b64 ${{b64:sample_packet_{name}}}, ${{b64:sample_packet_{name}}}, ${{b64:sample_expected_{name}}};
                            cvt.u64.u32 ${{b64:sample_seen_{name}}}, ${{b32:seen}};
                            or.b64 ${{b64:sample_packet_{name}}}, ${{b64:sample_packet_{name}}}, ${{b64:sample_seen_{name}}};
                            or.b64 ${{b64:sample_packet_{name}}}, ${{b64:sample_packet_{name}}}, {READINESS_SNAPSHOT | (root << 48)};
                            """
                                + _raw_stamp(f"${{b64:sample_packet_{name}}}")
                                + f"""
                            mov.pred ${{pred:sampled_{name}}}, 1;
                            ${{label:sample_done_{name}}}:
                            """
                            )
                            template = op.template
                            # wait_pair has two load sites; instrument each once.
                            if name == "last":
                                marker_pos = template.index("${label:wait_last}:")
                                prefix, suffix = (
                                    template[:marker_pos],
                                    template[marker_pos:],
                                )
                                template = prefix + suffix.replace(
                                    seen_load, seen_load + sample, 1
                                )
                            else:
                                template = template.replace(
                                    seen_load, seen_load + sample, 1
                                )
                            op.attrs["template"] = (
                                f"mov.pred ${{pred:sampled_{name}}}, 0;\n" + template
                            )

                        def wait_marker(event):
                            if not whole_graph:
                                return marker(event)
                            # Do not paint a dependency wait for a statically
                            # eliminated acquire (event zero).
                            pair = op.attrs["description"] == "tile wait_pair"
                            return PTXTemplateOp.build(
                                list(op.operands[1:3]) if pair else [op.operands[1]],
                                [],
                                (
                                    "or.b32 ${b32:flags}, ${b32:operand.first.val}, ${b32:operand.last.val};\n"
                                    "setp.eq.u32 ${pred:empty}, ${b32:flags}, 0;\n"
                                    if pair
                                    else "setp.eq.u32 ${pred:empty}, ${b32:operand.event.val}, 0;\n"
                                )
                                + "@${pred:empty} bra ${label:no_acquire};\n"
                                + stamp(event)
                                + "\n${label:no_acquire}:\n",
                                operand_names=["first", "last"] if pair else ["event"],
                                description=f"SM trace {event}",
                            )

                        # Each stamp has a separate placeholder scope.
                        pos = block.ops.index(op)
                        block.ops[pos : pos + 1] = [
                            wait_marker(event),
                            op,
                            wait_marker(event + 1),
                        ]
                    elif (
                        isinstance(op, PTXTemplateOp)
                        and op.attrs.get("description") == "load tile task"
                    ):
                        original_return = "@${pred:done} ret;"
                        if original_return not in op.template:
                            raise ValueError("missing uniform task exit")
                        op.attrs["template"] = op.template.replace(
                            original_return,
                            "@!${pred:done} bra ${label:trace_continue};\n"
                            + _raw_stamp(12)
                            + "ret;\n${label:trace_continue}:",
                        )
                    else:
                        for child in op.blocks:
                            visit(child, within_tile)

            # Observe scalar preparation at its actual phase. Previously the
            # plot omitted coefficient/index work before row loops, including
            # the work that idle GEMM lanes can execute immediately.
            if kernel.attrs.get("tile_static_lowering") == "root_loops" or source_kernel.attrs.get("sm_trace_granularity") == "tile":
                last_publication = -1
                setup_boundaries = []
                for index, op in enumerate(kernel.body.ops):
                    if isinstance(op, PTXTemplateOp) and (
                        op.attrs.get("description") == "tile publish"
                        or "atom.acq_rel.gpu.global.inc.u32" in op.template
                    ):
                        last_publication = index
                    if isinstance(op, ForOp) or (isinstance(op, IfOp) and find_ops_of_type(op, ForOp)):
                        tagged = [nested for nested in [op, *find_ops_of_type(op, IfOp)]
                                  if "sm_trace_root" in nested.attrs]
                        if tagged and root_kinds.get(tagged[0].attrs["sm_trace_root"]) == "norm":
                            setup_boundaries.append((last_publication + 1, index,
                                                     cast(int, tagged[0].attrs["sm_trace_root"])))
                for start, end, root in reversed(setup_boundaries):
                    kernel.body.ops.insert(end, marker(1000 * (root + 1) + 20))
                    kernel.body.ops.insert(start, marker(1000 * (root + 1) + 19))
            visit(kernel.body)
            groups, current = [], []
            for op in kernel.body.ops:
                current.append(op)
                if (
                    isinstance(op, PTXTemplateOp)
                    and "atom.acq_rel.gpu.global.inc.u32" in op.template
                ):
                    groups.append(current)
                    current = []
            if current:
                groups.append(current)
            if len(groups) != len(kinds):
                raise ValueError(
                    f"phase structure changed: {len(groups)} != {len(kinds)}"
                )
            ops = []
            for phase_index, (kind, group) in enumerate(
                zip(kinds, groups, strict=True)
            ):
                static_pair = any(
                    "sm_trace_kind" in child.attrs
                    for top in group
                    for child in [
                        *find_ops_of_type(top, IfOp),
                        *find_ops_of_type(top, ForOp),
                    ]
                )
                event = {"projection": 1, "down": 3}.get(kind, 9)
                if source_kernel.attrs.get("sm_trace_granularity") == "tile":
                    event += 1000 * (
                        cast(list[int], source_kernel.attrs["sm_trace_phase_roots"])[
                            phase_index
                        ]
                        + 1
                    )
                if not static_pair:
                    ops.append(marker(event))
                barrier = group[-1]
                if (
                    isinstance(barrier, PTXTemplateOp)
                    and "atom.acq_rel.gpu.global.inc.u32" in barrier.template
                ):
                    # Existing first CTA join precedes work-end; second join
                    # terminates the grid wait. Do not add a new CTA barrier.
                    pieces = barrier.template.split("bar.sync 0;", 1)
                    if len(pieces) != 2:
                        raise ValueError("unrecognized grid barrier")
                    ops.extend(group[:-1])
                    # Preserve the original template's temporary registers
                    # and labels; split templates would allocate new mappings.
                    suffix = "\n" if static_pair else "\n" + _raw_stamp(event + 1)
                    barrier.attrs["template"] = (
                        (_raw_stamp(17) if static_pair else "")
                        + pieces[0]
                        + "bar.sync 0;"
                        + suffix
                        + (_raw_stamp(18) if static_pair else "")
                        + _raw_stamp(5)
                        + pieces[1]
                        + _raw_stamp(6)
                    )
                    ops.append(barrier)
                else:
                    ops.extend(group)
                    if not static_pair:
                        ops.append(marker(event + 1))
            setup = PTXTemplateOp.build(
                [trace],
                [],
                f"""
.reg .b32 %sm_trace_tid, %sm_trace_block, %sm_trace_smid, %sm_trace_count;
.reg .b64 %sm_trace_offset, %sm_trace_base, %sm_trace_cursor, %sm_trace_clock;
.reg .pred %sm_trace_leader, %sm_trace_full;
mov.u32 %sm_trace_tid, %tid.x;
setp.eq.u32 %sm_trace_leader, %sm_trace_tid, 0;
mov.u32 %sm_trace_block, %ctaid.x;
mov.u32 %sm_trace_smid, %smid;
mul.wide.u32 %sm_trace_offset, %sm_trace_block, {(capacity + 1) * 16};
add.u64 %sm_trace_base, ${{b64:operand.trace.ptr}}, %sm_trace_offset;
add.u64 %sm_trace_cursor, %sm_trace_base, 16;
mov.u32 %sm_trace_count, 0;
@%sm_trace_leader st.global.u32 [%sm_trace_base], %sm_trace_smid;
@%sm_trace_leader st.global.u32 [%sm_trace_base+4], 0;
@%sm_trace_leader st.global.u32 [%sm_trace_base+8], 0;
@%sm_trace_leader st.global.u32 [%sm_trace_base+12], 0;
""",
                operand_names=["trace"],
                description="SM trace setup",
            )
            kernel.body.ops = [setup, marker(11), *ops, marker(12)]
            return lowered

    # Injecting raw stamps into a single original barrier preserves its
    # temporary register/label mappings. Unique literal labels avoid collisions.
    raw_index = 0

    def _raw_stamp(event):
        nonlocal raw_index
        raw_index += 1
        label = f"sm_trace_skip_{raw_index}"
        return f"""
@!%sm_trace_leader bra {label};
setp.ge.u32 %sm_trace_full, %sm_trace_count, {capacity};
@%sm_trace_full bra {label};
mov.u64 %sm_trace_clock, %globaltimer;
st.global.u64 [%sm_trace_cursor], %sm_trace_clock;
st.global.u64 [%sm_trace_cursor+8], {event};
add.u64 %sm_trace_cursor, %sm_trace_cursor, 16;
add.u32 %sm_trace_count, %sm_trace_count, 1;
st.global.u32 [%sm_trace_base+8], %sm_trace_count;
{label}:
"""

    observed = Observer().run(
        cast(ModuleOp, clone_op(cached_lowered, {}))
        if cached_lowered is not None else source
    )
    return codegen.make_ptx_kernel(observed, lowered=True)


def _capture_batch(run, replays=100):
    """Capture raw launches, keeping Python scheduling outside the timed batch."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(replays):
            run()
    return graph


def _bench_batch(graph, replays=100):
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def _profile(run, path):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        with torch.profiler.record_function("measured_cuda_graph_replay"):
            run()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = sorted(
        [event for event in trace["traceEvents"] if event.get("cat") == "kernel"],
        key=lambda e: e["ts"],
    )
    if not kernels:
        raise RuntimeError("PyTorch Profiler recorded no CUDA kernels")
    origin = kernels[0]["ts"]
    return [
        dict(
            name=e["name"],
            start_ms=(e["ts"] - origin) / 1000,
            duration_ms=e["dur"] / 1000,
            args=e.get("args", {}),
        )
        for e in kernels
    ]


def _launch_activity_summary(path):
    """Count device operations in one invocation, including scratch resets."""
    events = json.loads(path.read_text())["traceEvents"]
    categories = Counter(event.get("cat") for event in events)
    return dict(
        compute_launches=categories["kernel"],
        memset_operations=categories["gpu_memset"],
        memcpy_operations=categories["gpu_memcpy"],
    )


def _snapshot(event):
    if not event & READINESS_SNAPSHOT:
        return None
    return dict(
        root=(event >> 48) & 255,
        event=(event >> 32) & 65535,
        expected=(event >> 16) & 65535,
        observed=event & 65535,
    )


def _readiness_snapshots(raw):
    rows = raw.cpu().numpy()
    origin = int(rows[:, 1, 0].min())
    return [
        dict(
            sm=int(row[0, 0]), cta=cta, time_ms=(int(ticks) - origin) / 1e6, **snapshot
        )
        for cta, row in enumerate(rows)
        for ticks, event in row[1 : int(row[0, 1]) + 1]
        if (snapshot := _snapshot(int(event))) is not None
    ]


def _decode(raw):
    rows = raw.cpu().numpy()
    # Only the first valid entry of each CTA participates. Unused capacity
    # in torch.empty buffers can contain arbitrary positive epoch-sized data.
    origin = int(rows[:, 1, 0].min())
    if origin <= 0:
        raise RuntimeError("missing CTA entry timestamps")
    intervals, lifetime, smids = [], [], []
    for block, row in enumerate(rows):
        smid, count = map(int, row[0])
        if count > raw.shape[1] - 1 or count < 2:
            raise RuntimeError(f"missing/overflowed SM trace for CTA {block}: {count}")
        smids.append(smid)
        stack = {}
        snapshots = []
        events = row[1 : count + 1]
        if events[0, 1] != 11 or events[-1, 1] != 12:
            raise RuntimeError("missing CTA entry/exit markers")
        if np.any(np.diff(events[:, 0]) < 0):
            raise RuntimeError("nonmonotonic SM timestamps")
        for ticks, event in events:
            t = (int(ticks) - origin) / 1e6
            event = int(event)
            snapshot = _snapshot(event)
            if snapshot is not None:
                snapshots.append(dict(time_ms=t, **snapshot))
                continue
            base = event % 1000
            if base in KINDS:
                if event in stack:
                    raise RuntimeError("nested unmatched work markers")
                stack[event] = t
            elif base - 1 in KINDS:
                start = stack.pop(event - 1)
                intervals.append(
                    dict(
                        sm=smid,
                        cta=block,
                        kind=KINDS[base - 1],
                        start_ms=start,
                        end_ms=t,
                        root=event // 1000 - 1,
                    )
                )
        if stack:
            raise RuntimeError("unmatched SM trace markers")
        for interval in intervals:
            if interval["cta"] != block or interval["kind"] != "dependency wait":
                continue
            samples = [
                sample
                for sample in snapshots
                if interval["start_ms"] <= sample["time_ms"] <= interval["end_ms"]
            ]
            if samples and all(
                sample["observed"] >= sample["expected"] for sample in samples
            ):
                interval["kind"] = "readiness acquire / fence"
        lifetime.append(
            dict(
                sm=smid,
                cta=block,
                start_ms=(int(events[0, 0]) - origin) / 1e6,
                end_ms=(int(events[-1, 0]) - origin) / 1e6,
            )
        )
    if len(set(smids)) != len(smids):
        raise RuntimeError("expected one resident CTA per SM")
    return intervals, lifetime


def _work_segments(intervals):
    """Exclude nested readiness polling from every scheduled-work region."""
    waits = defaultdict(list)
    for item in intervals:
        if item["kind"] in ("dependency wait", "readiness acquire / fence", "scheduler metadata / dispatch"):
            waits[item["sm"]].append((item["start_ms"], item["end_ms"]))
    result = []
    for item in intervals:
        if item["kind"] not in WORK_KINDS:
            result.append(item)
            continue
        pieces = [(item["start_ms"], item["end_ms"])]
        for lo, hi in waits[item["sm"]]:
            pieces = [(a, min(b, lo)) for a, b in pieces if a < lo] + [
                (max(a, hi), b) for a, b in pieces if b > hi
            ]
        result.extend(item | dict(start_ms=a, end_ms=b) for a, b in pieces if b > a)
    return result
