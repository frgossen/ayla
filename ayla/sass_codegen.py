"""Scheduled GPU IR to Hopper machine instructions, without PTX templates.

The frontend lowering passes are shared with the PTX backend. This emitter
consumes their actual operations, values, buffer views and synchronization.
Virtual register blocks are allocated after emission, preserving contiguous
hardware operands and loop-carried lifetimes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import struct

from .base.ir import Value, find_ops_of_type
from .base.pipeline import Pipeline
from . import gpu_dialect as g
from .passes.insert_syncs import InsertSyncs
from .passes.lower_to_async import LowerToAsync
from .passes.lower_to_tma_move import LowerToTmaMove
from .passes.pack_smem_allocs import ReuseSmemAllocs
from .passes.verify_gpu_dialect import VerificationStage, VerifyGpuDialect
from .pipelines.canonicalize import Canonicalize


REGISTERS = re.compile(r"\b(?:UR\d+|R\d+|UP\d+|P\d+)\b")
FIXED = frozenset(
    (
        "MOV",
        "MOV32I",
        "UMOV",
        "IADD3",
        "UIADD3",
        "IMAD",
        "UIMAD",
        "LOP3",
        "ULOP3",
        "SHF",
        "USHF",
        "ISETP",
        "UISETP",
        "FADD",
        "FMUL",
        "FMUL32I",
        "CS2R",
        "F2FP",
        "HADD2",
        "R2UR",
    )
)
NO_DESTINATION = frozenset(
    (
        "STS",
        "STSM",
        "STG",
        "UTMALDG",
        "UTMASTG",
        "UTMACMDFLUSH",
        "UTMACCTL",
        "BAR",
        "WARPGROUP",
        "DEPBAR",
        "FENCE",
        "BRA",
        "EXIT",
        "NOP",
    )
)


def split_operands(text):
    parts, start, depth = [], 0, 0
    for index, character in enumerate(text):
        if character in "{[(":
            depth += 1
        elif character in "}])":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


@dataclass
class Instruction:
    opcode: str
    reads: set[str]
    writes: set[str]
    guarded: bool = False
    removable: bool = False
    target: str | None = None


def register_effects(line):
    if line.startswith("#"):
        return Instruction("LABEL", set(), set())
    text, bracket = line.split(";", 1)
    unguarded = re.sub(r"^@!?U?P\d+\s+", "", text)
    guard = text[: len(text) - len(unguarded)]
    mnemonic, operands = (
        unguarded.split(" ", 1) if " " in unguarded else (unguarded, "")
    )
    opcode = mnemonic.split(".")[0]
    parts = split_operands(operands)
    destinations = 0 if opcode in NO_DESTINATION else 1
    if opcode in ("ISETP", "UISETP", "SHFL") or (
        opcode == "UIADD3" and len(parts) > 1 and parts[1].startswith("UP")
    ):
        destinations = 2
    writes = set(REGISTERS.findall(",".join(parts[:destinations])))
    reads = set(REGISTERS.findall(",".join(parts[destinations:]) + guard))
    if opcode == "HGMMA":
        base = int(re.search(r"gdesc\[UR(\d+)\]", text)[1])
        reads.update(f"UR{base + i}" for i in range(4))
    elif opcode in ("UTMALDG", "UTMASTG"):
        rank = int(mnemonic.split(".")[1][:-1])
        base = int(re.search(r"UR(\d+)", parts[0])[1])
        reads.update(
            f"UR{base + i}" for i in range(rank + (2 if opcode == "UTMALDG" else 1))
        )
        pointer = int(re.search(r"UR(\d+)", parts[1])[1])
        reads.update((f"UR{pointer}", f"UR{pointer + 1}"))
    elif opcode == "UTMACCTL" and parts:
        pointer = int(re.search(r"UR(\d+)", parts[0])[1])
        reads.update((f"UR{pointer}", f"UR{pointer + 1}"))
    elif opcode == "SYNCS" and ".EXCH.64" in mnemonic:
        base = int(re.search(r"UR(\d+)", parts[-1])[1])
        reads.add(f"UR{base + 1}")
    control = bracket.strip("[]").split(":")
    target = re.search(r"#label\(([^)]+)\)", text)
    return Instruction(
        opcode,
        reads,
        writes,
        bool(guard),
        bool(writes) and opcode in FIXED and control[:2] == ["7", "7"],
        target[1] if target else None,
    )


def eliminate_dead_code(lines):
    """Remove unused fixed-result calculations before physical allocation.

    Solve liveness over branches and backedges. Predicated writes preserve the
    bypass value. Memory operations, fences, waits and decoupled producers are
    retained. Implicit GMMA/TMA register spans participate in liveness.
    Return a prefix map so asynchronous and loop live ranges can be remapped.
    """
    instructions = [register_effects(line) for line in lines]
    labels = {
        line[len("#def_label(") : -1]: i
        for i, line in enumerate(lines)
        if line.startswith("#def_label(")
    }
    successors, predecessors = [], [[] for _ in lines]
    for i, instruction in enumerate(instructions):
        targets = []
        if instruction.opcode != "EXIT" and (
            instruction.opcode != "BRA" or instruction.guarded
        ):
            if i + 1 < len(lines):
                targets.append(i + 1)
        if instruction.target is not None:
            targets.append(labels[instruction.target])
        successors.append(targets)
        for target in targets:
            predecessors[target].append(i)
    live = [set() for _ in lines]
    kept = [True] * len(lines)
    work = deque(reversed(range(len(lines))))
    queued = set(work)
    while work:
        i = work.popleft()
        queued.remove(i)
        instruction = instructions[i]
        out = set().union(*(live[s] for s in successors[i]))
        kept[i] = not instruction.removable or bool(instruction.writes & out)
        incoming = out
        if kept[i]:
            incoming = (
                out if instruction.guarded else out - instruction.writes
            ) | instruction.reads
        if incoming != live[i]:
            live[i] = incoming
            for p in predecessors[i]:
                if p not in queued:
                    work.append(p)
                    queued.add(p)
    prefix = [0]
    result = []
    for line, keep in zip(lines, kept, strict=True):
        if keep:
            result.append(line)
        prefix.append(len(result))
    return result, prefix


@dataclass(eq=False, kw_only=True)
class LiveRange:
    size: int
    alignment: int
    first: int = 10**9
    last: int = -1
    offset: int = 0

    def extend_loop(self, start, end, completion=0):
        if self.first < start <= self.last <= end:
            self.last = end + completion
        elif start <= self.first <= end < self.last:
            self.first = start


def place_live_ranges(intervals, reserved=()):
    """First-fit placement of aligned, contiguous ranges with inclusive lives.

    The same placement serves shared bytes and register slots. Each caller
    supplies its own completion-aware lifetimes and hardware reservations.
    """
    active, peak = [], 0
    for interval in sorted(intervals, key=lambda item: item.first):
        if interval.last < 0:
            continue
        active = [old for old in active if old.last >= interval.first]
        occupied = [*reserved, *((old.offset, old.size) for old in active)]
        offset = 0
        while True:
            offset = (
                (offset + interval.alignment - 1)
                // interval.alignment
                * interval.alignment
            )
            overlap = next(
                (
                    (base, size)
                    for base, size in occupied
                    if offset < base + size and base < offset + interval.size
                ),
                None,
            )
            if overlap is None:
                break
            offset = sum(overlap)
        interval.offset = offset
        active.append(interval)
        peak = max(peak, offset + interval.size)
    return peak


def allocate_shared(kernel):
    """Overlay tensor storage only after every async reader/writer completes.

    Locks remain live throughout execution. Views alias their original root;
    structured loops extend live ranges across backedges. An outstanding
    transfer without a matching wait keeps storage live to kernel exit.
    """
    allocations = find_ops_of_type(kernel, g.AllocOp)
    lifetimes = {}
    for alloc in allocations:
        typ = alloc.result().type
        if isinstance(typ, g.SmemTensorType):
            alignment = alloc.attrs.get(
                "allocation_alignment", 1024 if typ.swizzle else 128
            )
            lifetimes[alloc.result()] = LiveRange(
                size=typ.num_elements * typ.dtype.byte_size, alignment=alignment
            )
    events = []
    loops = []

    def walk(block):
        for op in block.ops:
            start = len(events)
            events.append(op)
            for child in op.blocks:
                walk(child)
            if isinstance(op, g.ForOp):
                events.append(None)
                loops.append((start, len(events) - 1))

    walk(kernel.body)

    def roots(op):
        return {
            g.root_buffer(value)
            for value in op.operands
            if isinstance(value.type, g.SmemTensorType)
        } & lifetimes.keys()

    def touch(values, index):
        for value in values:
            life = lifetimes[value]
            life.first = min(life.first, index)
            life.last = max(life.last, index)

    matrix = []
    pending = {}
    for index, op in enumerate(events):
        if isinstance(op, (g.MoveOp, g.TmaMoveOp, g.MatmulOp)):
            touch(roots(op), index)
        elif isinstance(op, g.AsyncOp):
            values = set().union(*(roots(child) for child in op.body.ops))
            touch(values, index)
            if op.lock.type.kind == "wgmma":
                matrix.append(values)
            else:
                pending.setdefault(op.lock, set()).update(values)
        elif isinstance(op, g.AwaitOp):
            if op.lock.type.kind == "wgmma":
                count = max(0, len(matrix) - op.pending_groups)
                for values in matrix[:count]:
                    touch(values, index)
                matrix = matrix[count:]
            else:
                touch(pending.pop(op.lock, set()), index)
    # Ordinary shared stores complete at the next CTA rendezvous. Keep their
    # storage until then even when the IR never reads the stored value again.
    for index, op in enumerate(events):
        if isinstance(op, g.MoveOp) and isinstance(op.dst.type, g.SmemTensorType):
            completion = next(
                (
                    j
                    for j in range(index + 1, len(events))
                    if isinstance(events[j], g.BarrierOp)
                    and events[j].scope in ("block", "block_leader", "warpgroup")
                ),
                len(events),
            )
            touch(roots(op), completion)
    for values in [*matrix, *pending.values()]:
        touch(values, len(events))
    for life in lifetimes.values():
        for start, end in reversed(loops):
            life.extend_loop(start, end)
    peak = place_live_ranges(lifetimes.values())
    locations = {value: life.offset for value, life in lifetimes.items()}
    peak = (peak + 7) // 8 * 8
    for alloc in allocations:
        value = alloc.result()
        if isinstance(value.type, g.LockType) and value.type.kind == "mbar":
            locations[value] = peak
            peak += 8
    return locations, peak, lifetimes


def relax_prefetched_matrix_waits(kernel):
    """Batch groups when no intervening operation observes their completion.

    Restrict to the kernel's straight-line region. A TMA issue, shared write,
    accumulator access or control-flow boundary stops the proof. Retain full
    drains before epilogues/refills and never exceed eight hardware groups.
    The modified IR also drives shared-memory lifetime allocation.
    """
    pure = (
        g.AllocOp,
        g.TileOp,
        g.ReinterpretOp,
        g.ReshapeOp,
        g.IndexConstantOp,
        g.IndexAddOp,
        g.IndexMulOp,
        g.IndexDivOp,
        g.IndexModOp,
        g.IndexLtOp,
        g.BlockIdOp,
        g.ThreadIdOp,
        g.ConstantOp,
        g.ToPtrOp,
        g.PtrAddOp,
    )
    queue = 0
    relaxed = 0
    ops = kernel.body.ops
    for index, op in enumerate(ops):
        if isinstance(op, (g.ForOp, g.IfOp)):
            queue = 8  # Unknown carried state; the next full drain resolves it.
        elif isinstance(op, g.AsyncOp) and op.lock.type.kind == "wgmma":
            queue = min(8, queue + 1)
        elif isinstance(op, g.AwaitOp) and op.lock.type.kind == "wgmma":
            if op.is_full_wait and 0 < queue < 8:
                for candidate in ops[index + 1 :]:
                    if isinstance(candidate, pure) or (
                        isinstance(candidate, g.AwaitOp)
                        and candidate.lock.type.kind == "mbar"
                    ):
                        continue
                    if (
                        isinstance(candidate, g.AsyncOp)
                        and candidate.lock.type.kind == "wgmma"
                        and candidate.body.ops
                        and all(
                            isinstance(mm, g.MatmulOp)
                            and mm.attrs.get("accumulate", True)
                            for mm in candidate.body.ops
                        )
                    ):
                        op.attrs["pending"] = queue
                        op.attrs["sass_wait_elided"] = True
                        relaxed += 1
                    break
            queue = min(queue, op.pending_groups)
    return relaxed


class _Emitter:
    """Conservative issue schedule, with explicit variable-latency barriers."""

    def __init__(self):
        self.lines = []
        self.pending: set[int] = set()

    def emit(self, instruction, *, wr=7, rd=7, stall=8):
        # sm_90_latencies.txt: UMOV needs at most five cycles to any
        # consumer; packed F2FP is an FXU operation with six-cycle RAW.
        if instruction.startswith("UMOV "):
            stall = 5
        elif instruction.startswith("F2FP."):
            stall = 6
        if any(m in instruction for m in ("IMAD", "FADD")):
            stall = 6
        if instruction.startswith("F2F.") and wr == 7 and rd == 7:
            wr, rd = 0, 1
        if instruction.startswith("FENCE.") and wr == 7:
            wr = 0  # Proxy publication is a decoupled completion, not a delay.
        if "ISETP" in instruction:
            stall = 13
        req = ",".join(map(str, sorted(self.pending)))
        self.lines.append(
            f"{instruction};[{wr}:{rd}:{{{req}}}:{stall}:{int(stall >= 12)}]"
        )
        self.pending = {s for s in (wr, rd) if s != 7}

    def label(self, name):
        # Branches drain outstanding general scoreboards before jumping.
        assert not self.pending
        self.lines.append(f"#def_label({name})")

    def mov(self, dst, value):
        if dst.startswith("UR"):
            self.emit(f"UMOV {dst}, {value}")
        else:
            self.emit(
                f"MOV {dst}, {value}"
                if str(value).startswith(("R", "UR"))
                else f"MOV32I {dst}, {value}"
            )


def _schedule_instructions(lines):
    """Schedule RAW, WAR, and WAW dependencies without moving instructions.

    Keep variable-latency barriers and GMMA completion waits intact. Track
    fixed results across memory instructions, including the implicit register
    spans of tensor-map argument blocks and composite GMMA descriptors.
    Register and predicate latencies follow sm_90_latencies.txt.
    """
    fixed = FIXED
    writes, reads, result = {}, {}, []
    previous_control = None
    previous_issue, minimum = 0, 0

    def latency(register, producer, consumer=None):
        if register.startswith("UP"):
            return 4 if consumer in fixed and consumer.startswith("U") else 13
        if register.startswith("P"):
            return 13
        if register.startswith("UR"):
            if producer == "R2UR":
                return 15
            if producer in ("UMOV", "ULDC"):
                return (
                    2
                    if consumer in fixed
                    and consumer.startswith("U")
                    or consumer in ("UTMALDG", "UTMASTG")
                    else 5
                )
            if consumer == "UMOV":
                return 7
            if consumer in ("UTMALDG", "UTMASTG"):
                return 9
            return 4 if consumer in fixed and consumer.startswith("U") else 12
        # Slow MIO operations consume both addresses and data through the
        # eight-cycle path (TABLE_TRUE(GPR), sm_90_latencies.txt). FADD/FMUL
        # are FMAI operations on Hopper; six cycles is a conservative bound.
        slow_mio = {
            "LDG",
            "STG",
            "LDS",
            "STS",
            "STSM",
            "LDSM",
            "SHFL",
            "HGMMA",
            "BAR",
            "S2R",
        }
        return (
            8 if consumer in slow_mio or consumer in (None, "BRA", "WARPGROUP") else 6
        )

    for line in lines:
        if line.startswith("#"):
            if previous_control is not None:
                drain = max(
                    [previous_issue + minimum]
                    + [
                        at + latency(register, producer)
                        for register, (at, producer) in writes.items()
                    ]
                )
                minimum = drain - previous_issue
                previous_control[3] = str(minimum)
            writes.clear()
            reads.clear()
            result.append((line, None))
            continue
        instruction, bracket = line.split(";", 1)
        control = bracket.strip("[]").split(":")
        effects = register_effects(line)
        opcode, dst, src = effects.opcode, effects.writes, effects.reads
        if opcode == "WARPGROUP" and ".ARRIVE" in instruction:
            src.update(register for register in writes if register.startswith("R"))

        issue = previous_issue + minimum if previous_control is not None else 0
        for register in src:
            if register in writes:
                at, producer = writes[register]
                issue = max(issue, at + latency(register, producer, opcode))
        for register in dst:
            if register in reads:
                at, reader = reads[register]
                anti = 3 if register.startswith("UR") else (1 if reader in fixed else 2)
                issue = max(issue, at + anti)
            if register in writes:
                at, producer = writes[register]
                if register.startswith("UR"):
                    output = (
                        7
                        if producer != "UMOV" and opcode == "UMOV"
                        else (1 if producer == "UMOV" and opcode == "UMOV" else 4)
                    )
                else:
                    output = 1 if opcode in fixed else 2
                issue = max(issue, at + output)
        if previous_control is not None:
            previous_control[3] = str(issue - previous_issue)
        for register in src:
            reads[register] = (issue, opcode)
        for register in dst:
            if opcode in fixed or opcode == "ULDC":
                writes[register] = (issue, opcode)
            else:
                # Decoupled writes use the unchanged hardware scoreboards;
                # asynchronous GMMA writes use the existing group waits.
                writes.pop(register, None)
        previous_control = control
        previous_issue = issue
        minimum = (
            1 if opcode in fixed and control[:2] == ["7", "7"] else int(control[3])
        )
        result.append((instruction, control))
    if previous_control is not None:
        previous_control[3] = str(minimum)
    scheduled = []
    for instruction, control in result:
        if control is None:
            scheduled.append(instruction)
            continue
        control[4] = str(int(int(control[3]) >= 12))
        scheduled.append(instruction + ";[" + ":".join(control) + "]")
    return scheduled


def _parameter_layout(signature):
    # CUDA parameter alignment is absolute in constant bank 0. TMA maps need
    # 128-byte alignment; pointers need 8. The reference parser's natural
    # alignment caps at 8, so emit the KPARAM records explicitly.
    offset = 0x210
    params = []
    for i, typ in enumerate(signature):
        size = 128 if isinstance(typ, g.TmaDescType) else 8
        offset = (offset + size - 1) & -size
        params.append((i, offset - 0x210, size))
        offset += size
    return params


class SassLoweringPipeline(Pipeline):
    """The backend-independent preparation preceding machine lowering."""

    def __init__(self):
        super().__init__()
        self.add_instrumentation_pass(VerifyGpuDialect(VerificationStage.PTX_LOWERING))
        for step in (
            LowerToTmaMove(),
            LowerToAsync(),
            Canonicalize(),
            InsertSyncs(),
            ReuseSmemAllocs(),
        ):
            self.add_pass(step)


@dataclass(eq=False)
class RegisterBlock(LiveRange):
    family: str
    virtual_base: int
    name: str

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        return f"{self.family}{self.virtual_base + index}"

    def group(self, start=0, size=None):
        return (
            "{"
            + ",".join(self[i] for i in range(start, start + (size or self.size)))
            + "}"
        )


class RegisterAllocator:
    """Linear scan of contiguous blocks, with structured loop live ranges.

    Register pressure beyond the hardware budget is diagnosed, never silently
    truncated. Spilling is not yet implemented. R1 is reserved for the ABI;
    UR63/P7/UP7 are hardware zero/true sentinels.
    """

    pattern = re.compile(r"\b(UR|UP|R|P)(\d+)\b")

    def __init__(self):
        self.blocks = []
        self.lookup = {}
        self.next = 1000
        self.loops = []

    def new(self, family, size=1, alignment=1, name="temporary"):
        block = RegisterBlock(
            family=family,
            virtual_base=self.next,
            size=size,
            alignment=alignment,
            name=name,
        )
        self.next += size
        self.blocks.append(block)
        for i in range(size):
            self.lookup[block[i]] = block
        return block

    def touch(self, block, index):
        block.first = min(block.first, index)
        # Keep a decoupled operand through the following scoreboard wait.
        block.last = max(block.last, index + 1)

    def allocate(self, lines):
        for index, line in enumerate(lines):
            for match in self.pattern.finditer(line):
                block = self.lookup.get(match[0])
                if block is not None:
                    self.touch(block, index)
        for block in self.blocks:
            for start, end in reversed(self.loops):
                block.extend_loop(start, end, completion=1)
        counts = {}
        for family, limit in {"R": 254, "UR": 63, "P": 7, "UP": 7}.items():
            blocks = [b for b in self.blocks if b.family == family and b.last >= 0]
            counts[family] = place_live_ranges(
                blocks, ((1, 1),) if family == "R" else ()
            )
            if counts[family] > limit:
                block = next(b for b in blocks if b.offset + b.size > limit)
                raise ValueError(
                    f"SASS {family} register pressure exceeds {limit}: {block.name} ({block.size} registers); spilling is unsupported"
                )

        def replace(match):
            block = self.lookup.get(match[0])
            if block is None:
                return match[0]
            offset = int(match[2]) - block.virtual_base
            return f"{block.family}{block.offset + offset}"

        return [self.pattern.sub(replace, line) for line in lines], counts


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple


def expression(op, a, b):
    if isinstance(a, int) and isinstance(b, int):
        return {
            "add": lambda: a + b,
            "mul": lambda: a * b,
            "div": lambda: a // b,
            "mod": lambda: a % b,
        }[op]()
    if op == "add" and a == 0:
        return b
    if op == "add" and b == 0:
        return a
    if op == "mul" and (a == 0 or b == 0):
        return 0
    if op == "mul" and a == 1:
        return b
    if op == "mul" and b == 1:
        return a
    if op == "div" and b == 1:
        return a
    if op == "mod" and b == 1:
        return 0
    return Expr(op, (a, b))


@dataclass
class Memory:
    root: Value
    offset: object = 0
    origin: tuple = ()


class SassCodegen:
    """Operation dispatch and virtual machine emission for Hopper."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.descriptor_batch = 4
        self.e = _Emitter()
        self.ra = RegisterAllocator()
        self.values = {}
        self.fragments = {}
        self.shared = {}
        self.phases = {}
        self.params = _parameter_layout(kernel.signature)
        self.parameter_pointers = {}
        self.parameter_base = None
        self.shared_bytes = 0
        self.label_index = 0
        self.in_async = False
        self.matrix_groups = []
        self.matrix_read_groups = []
        self.completed_matrix_reads = set()
        self.fence_needed = True
        self.memory_desc = None
        self.cache_policies = {}
        self.descriptor_pool = []
        self.store_bases = {}
        self.materializations = {}
        self.deferred_conversions = {}
        self.zero_stores = set()
        self.prefetch_branches_elided = 0
        self.tid = self.reg("R", name="thread ID")
        self.owner = self.reg("P", name="TMA owner warp")
        self.leader = self.reg("P", name="TMA issuing lane")
        self.uniform_leader = self.reg("UP", name="TMA issuing warp")

    def reg(self, family, size=1, alignment=1, name="temporary"):
        return self.ra.new(family, size, alignment, name)

    def emit(self, instructions, **control):
        # A block shares control defaults; each instruction still updates the
        # scoreboard independently. Async issues keep explicit per-call controls.
        for instruction in instructions.strip().splitlines():
            instruction = instruction.strip()
            self.e.emit(instruction, **control)
            if "BRA " in instruction:
                self.materializations.clear()

    def mark_label(self, name):
        self.materializations.clear()
        self.e.label(name)

    def label(self, prefix):
        label = f"{prefix}_{self.label_index}"
        self.label_index += 1
        return label

    def materialize(self, value, family="UR", destination=None):
        # Scalar index values are immutable within a basic block. Branches,
        # nested regions and loop latches end reuse, preserving dominance.
        if destination is not None:
            return self._materialize(value, family, destination)
        if not isinstance(value, (int, Expr, RegisterBlock)):
            raise ValueError(f"unsupported SASS expression {value}")
        key = (family, value)
        if key not in self.materializations:
            self.materializations[key] = self._materialize(value, family)
        return self.materializations[key]

    def _materialize(self, value, family="UR", destination=None):
        if isinstance(value, RegisterBlock):
            if value.family == family and destination is None:
                return value[0]
            dst = destination or self.reg(family)[0]
            if family == "UR" and value.family != family:
                raise ValueError("nonuniform SASS value cannot become a TMA coordinate")
            self.e.mov(dst, value[0])
            return dst
        if value == 0 and destination is None:
            return "URZ" if family == "UR" else "RZ"
        dst = destination or self.reg(family)[0]
        if isinstance(value, int):
            self.e.mov(dst, value & 0xFFFFFFFF)
            return dst
        if not isinstance(value, Expr):
            raise ValueError(f"unsupported SASS expression {value}")
        op, (a, b) = value.op, value.args
        a = self.materialize(a, family)
        if op in ("div", "mod"):
            if not isinstance(b, int) or b <= 0 or b >= 2**32:
                raise ValueError(
                    "SASS index division requires a positive constant divisor"
                )
            if b & (b - 1):
                self.constant_division(dst, a, b, family, op == "mod")
                return dst
            if op == "div":
                text = (
                    f"USHF.R.U32 {dst}, {a}, {b.bit_length() - 1}, URZ"
                    if family == "UR"
                    else f"SHF.R.U32.HI {dst}, RZ, {b.bit_length() - 1}, {a}"
                )
            else:
                text = f"{'U' if family == 'UR' else ''}LOP3.LUT {dst}, {a}, {b - 1}, {'URZ' if family == 'UR' else 'RZ'}, 0xc0, {'!UPT' if family == 'UR' else 'PT'}"
        else:
            operand = (
                b
                if isinstance(b, int) and -(2**19) <= b < 2**19
                else self.materialize(b, family)
            )
            prefix = "U" if family == "UR" else ""
            text = (
                f"{prefix}IADD3 {dst}, {a}, {operand}, {prefix}RZ"
                if op == "add"
                else f"{prefix}IMAD {dst}, {a}, {operand}, {prefix}RZ"
            )
        self.emit(text)
        return dst

    @staticmethod
    def uniform(value):
        if isinstance(value, int):
            return True
        if isinstance(value, RegisterBlock):
            return value.family == "UR"
        return isinstance(value, Expr) and all(
            SassCodegen.uniform(arg) for arg in value.args
        )

    def constant_division(self, dst, source, divisor, family, remainder):
        magic, shift, add = g.unsigned_division_magic(divisor)
        x = self.reg("R")
        q = self.reg("R")
        self.emit(f"""MOV {x[0]}, {source}
            IMAD.HI.U32 {q[0]}, {x[0]}, {magic}, RZ""")
        if add:
            t = self.reg("R")
            self.emit(f"""IADD3 {t[0]}, {x[0]}, -{q[0]}, RZ
                SHF.R.U32.HI {t[0]}, RZ, 1, {t[0]}
                IADD3 {q[0]}, {t[0]}, {q[0]}, RZ""")
        self.emit(f"SHF.R.U32.HI {q[0]}, RZ, {shift}, {q[0]}")
        if remainder:
            self.emit(f"IMAD {q[0]}, {q[0]}, {(-divisor) & 0xFFFFFFFF}, {x[0]}")
        self.emit(f"R2UR {dst}, {q[0]}") if family == "UR" else self.e.mov(dst, q[0])

    def pointer(self, value):
        if value not in self.parameter_pointers:
            index = self.kernel.body.arguments.index(value)
            off = self.params[index][1]
            dst = self.reg("UR", 2, 2, "parameter pointer")
            if isinstance(value.type, g.TmaDescType):
                if self.parameter_base is None:
                    self.parameter_base = self.reg("UR", 2, 2, "parameter bank base")
                    self.emit(f"ULDC.64 {self.parameter_base.group()}, c[0x0][0x198]")
                base = self.parameter_base
                carry = self.reg("UP")
                self.emit(f"""UIADD3 {dst[0]}, {carry[0]}, {base[0]}, {off}, URZ
                    UIADD3.X {dst[1]}, URZ, {base[1]}, URZ, {carry[0]}, !UPT""")
            else:
                self.emit(f"ULDC.64 {dst.group()}, c[0x0][{0x210 + off:#x}]")
            self.parameter_pointers[value] = dst
        return self.parameter_pointers[value]

    def memory_descriptor(self):
        if self.memory_desc is None:
            self.memory_desc = self.reg("UR", 2, 2, "global memory descriptor")
            self.emit(f"ULDC.64 {self.memory_desc.group()}, c[0x0][0x208]")
        return self.memory_desc

    def root_fragment(self, value):
        root = g.root_buffer(value)
        if root not in self.fragments:
            typ = root.type
            if not isinstance(typ, g.WgrmemTensorType) or typ.dtype.name not in (
                "f32",
                "f16",
                "bf16",
            ):
                raise ValueError(f"unsupported SASS fragment {typ}")
            count = typ.num_elements_per_thread // (1 if typ.dtype.name == "f32" else 2)
            # Keep FP32 GMMA destinations 32-aligned: smaller alignments
            # failed memcheck or numerical validation. Packed halves need
            # only the four-register alignment of STSM operands.
            self.fragments[root] = self.reg(
                "R", count, 32 if typ.dtype.name == "f32" else 4, "fragment"
            )
        return self.fragments[root]

    def fragment_regs(self, value):
        block = self.root_fragment(value)
        indices = (
            [2 * p + i for p, _, _ in g.wgmma_view_pairs(value) for i in (0, 1)]
            if value.type.dtype.name == "f32"
            else [p for p, _, _ in g.wgmma_view_pairs(value)]
        )
        return [block[i] for i in indices]

    def shared_address(self, value):
        mem = self.values[value]
        if not isinstance(mem, Memory):
            raise ValueError("expected a SASS shared-memory view")
        return expression("add", self.shared[mem.root], mem.offset)

    def init(self):
        if (
            self.kernel.is_clustered
            or self.kernel.num_threads != 128
            or self.kernel.resolved_num_warp_groups != 1
        ):
            raise ValueError(
                "SASS machine lowering currently requires one nonclustered 128-thread warpgroup"
            )
        self.emit(f"S2R {self.tid[0]}, SR_TID.X", wr=0)
        self.emit(f"""ISETP.LT.U32.AND {self.owner[0]}, PT, {self.tid[0]}, 32, PT
            ISETP.EQ.U32.AND {self.leader[0]}, PT, {self.tid[0]}, RZ, PT
            VOTEU.ANY {self.uniform_leader[0]}, {self.leader[0]}""")
        self.shared, self.shared_bytes, self.shared_lifetimes = allocate_shared(
            self.kernel
        )
        for alloc in find_ops_of_type(self.kernel, g.AllocOp):
            value = alloc.result()
            typ = value.type
            if isinstance(typ, g.SmemTensorType):
                self.values[value] = Memory(value)
            elif isinstance(typ, g.LockType) and typ.kind == "mbar":
                waits = [
                    wait
                    for wait in find_ops_of_type(self.kernel, g.AwaitOp)
                    if wait.lock is value
                ]
                once = len(waits) == 1 and waits[0].parent_block is self.kernel.body
                self.phases[value] = (
                    None if once else self.reg("R", name="mbarrier phase")
                )
                if not once:
                    self.e.mov(self.phases[value][0], 0)
            elif not isinstance(
                typ, (g.LockType, g.WgrmemTensorType, g.RmemTensorType)
            ):
                raise ValueError(f"unsupported SASS allocation {typ}")
        done = self.label("init_done")
        self.emit(f"@!{self.owner[0]} BRA #label({done})", stall=1)
        state = self.reg("UR", 2, 2, "mbarrier initial state")
        self.e.mov(state[0], 0x1FFFFE)
        self.e.mov(state[1], 0x7FFFF800)
        locks = list(self.phases)
        # Independent lock addresses share an immutable state pair. Completion
        # waits protect that pair through publication; six write scoreboards
        # bound the batch, avoiding a wait after every initialization store.
        for start in range(0, len(locks), 6):
            chunk = locks[start : start + 6]
            for slot, lock in enumerate(chunk):
                self.emit(
                    f"@{self.uniform_leader[0]} SYNCS.EXCH.64 URZ, [URZ+{self.shared[lock]}], {state[0]}",
                    wr=slot,
                    stall=2,
                )
                self.e.pending.clear()
            self.e.pending.update(range(len(chunk)))
            self.emit("NOP", stall=1)
            self.ra.touch(state, len(self.e.lines) - 1)
        if not locks:
            self.emit("NOP", stall=1)
        self.mark_label(done)
        for value in self.kernel.body.arguments:
            if isinstance(value.type, g.GmemTensorType):
                self.values[value] = Memory(value, origin=(0,) * value.type.rank)
            elif isinstance(value.type, g.TmaDescType):
                self.pointer(value)
        # Definitions dominate loops and conditionals. First-use caching inside
        # a conditional would leave a later reader uninitialized on its bypass.
        for move in find_ops_of_type(self.kernel, g.MoveOp):
            for value in (move.src, move.dst):
                if isinstance(value.type, g.GmemTensorType):
                    self.pointer(g.root_buffer(value))
                    self.memory_descriptor()
        for transfer in find_ops_of_type(self.kernel, g.TmaMoveOp):
            if transfer.l2_cache_hint is not None:
                self.cache_policy(transfer.l2_cache_hint)

        matrices = find_ops_of_type(self.kernel, g.MatmulOp)
        highs = {
            tuple(self.descriptor_bits(v, 0) >> 32 for v in (mm.lhs, mm.rhs))
            for mm in matrices
        }
        if len(highs) == 1:
            hi_lhs, hi_rhs = next(iter(highs))
            for _ in range(self.descriptor_batch):
                descriptor = self.reg("UR", 4, 4, "invariant GMMA descriptor")
                self.e.mov(descriptor[1], hi_lhs)
                self.e.mov(descriptor[3], hi_rhs)
                self.descriptor_pool.append(descriptor)

    def cache_policy(self, hint):
        if hint not in self.cache_policies:
            cache = self.reg("UR", 2, 2, "TMA cache policy")
            self.e.mov(cache[0], 0)
            self.e.mov(
                cache[1],
                ({"evict_first": 0x12, "evict_last": 0x14}[hint] << 24) | 0xF00000,
            )
            self.cache_policies[hint] = cache
        return self.cache_policies[hint]

    def run(self):
        self.init()
        self.block(self.kernel.body)
        if self.matrix_groups:
            raise ValueError("SASS IR leaves GMMA groups outstanding")
        self.emit("EXIT")
        lines, prefix = eliminate_dead_code(self.e.lines)
        old_length = len(self.e.lines)
        self.optimization_stats = {
            "dead_instructions_removed": old_length - len(lines),
            "store_conversions_deferred": len(self.deferred_conversions),
            "prefetch_owner_branches_elided": self.prefetch_branches_elided,
        }
        for block in self.ra.blocks:
            if block.last >= 0:
                block.first = prefix[min(block.first, old_length)]
                block.last = prefix[min(block.last + 1, old_length)]
        self.ra.loops = [(prefix[a], prefix[b + 1] - 1) for a, b in self.ra.loops]
        lines, counts = self.ra.allocate(lines)
        # Uniform BRA uses an operand predicate rather than an @UP guard.
        lines = [re.sub(r"^@(!?UP\d+) BRA ", r"BRA.U \1, ", line) for line in lines]
        source = "\n".join(_schedule_instructions(lines)) + "\n"
        return source.replace("URZ", "UR63"), self.params, counts

    def block(self, block):
        self.materializations.clear()
        index = 0
        while index < len(block.ops):
            op = block.ops[index]
            following = block.ops[index + 1] if index + 1 < len(block.ops) else None
            if isinstance(op, g.AsyncOp) and op.lock.type.kind == "mbar":
                # Adjacent loads have the same issuing warp. Constants emit
                # no instructions, so they do not interrupt this region.
                # Every load still drains its argument read scoreboard;
                # no await, collective or other operation may be crossed.
                end = index + 1
                loads = 1
                while end < len(block.ops):
                    candidate = block.ops[end]
                    if isinstance(candidate, g.IndexConstantOp):
                        end += 1
                    elif (
                        isinstance(candidate, g.AsyncOp)
                        and candidate.lock.type.kind == "mbar"
                    ):
                        loads += 1
                        end += 1
                    else:
                        break
                if loads > 1:
                    self.prefetch_branches_elided += loads - 1
                    done = self.label("tma_done")
                    self.emit(f"@!{self.owner[0]} BRA #label({done})", stall=1)
                    for candidate in block.ops[index:end]:
                        if isinstance(candidate, g.AsyncOp):
                            self.async_op(candidate, owner_selected=True)
                        else:
                            self.lower(candidate)
                    self.mark_label(done)
                    index = end
                    continue
            # Only an adjacent, exclusively consumed positive-zero fill can
            # disappear. Views, other readers and negative zero retain storage.
            if (
                isinstance(op, g.ElementwiseOp)
                and op.fn == "fill"
                and isinstance(following, g.MoveOp)
                and following.src is op.dst
                and isinstance(following.dst.type, g.GmemTensorType)
                and op.dst.type.dtype.name in ("f32", "f16", "bf16")
                and self.only_reader(op.dst, following, op)
                and isinstance(self.values.get(op.src(0)), (int, float))
                and struct.pack("<f", self.values[op.src(0)]) == b"\x00" * 4
            ):
                self.zero_stores.add(following)
                index += 1
                continue
            if isinstance(op, g.ElementwiseOp) and self.can_defer_conversion(op):
                self.deferred_conversions[op.dst] = op.src(0)
                index += 1
                continue
            if (
                isinstance(op, g.MoveOp)
                and isinstance(op.dst.type, g.WgrmemTensorType)
                and isinstance(op.src.type, g.GmemTensorType)
                and op.src.type.rank == 2
                and tuple(op.src.type.strides) == (0, 1)
                and op.src.type.dtype.name in ("f16", "bf16")
                and isinstance(following, g.ElementwiseOp)
                and following.fn == "convert"
                and following.src(0) is op.dst
                and following.dst.type.dtype.name == "f32"
                and self.only_reader(op.dst, following, op)
            ):
                self.broadcast_convert(op, following)
                index += 2
                continue
            if isinstance(op, g.BarrierOp) and self.redundant_refill_barrier(
                block, index
            ):
                index += 1
                continue
            self.lower(op)
            index += 1
        self.materializations.clear()

    def can_defer_conversion(self, op):
        if (
            op.fn != "convert"
            or op.src(0).type.dtype.name != "f32"
            or op.dst.type.dtype.name not in ("f16", "bf16")
        ):
            return False
        readers = set(op.dst.users()) - {op}
        if not readers or not all(
            isinstance(r, g.MoveOp)
            and r.src is op.dst
            and self.can_stmatrix(r.dst)
            and r.parent_block is op.parent_block
            for r in readers
        ):
            return False
        block = op.parent_block
        first = block.ops.index(op)
        last = max(block.ops.index(r) for r in readers)
        if any(block.ops.index(r) <= first for r in readers):
            return False
        root = g.root_buffer(op.src(0))
        for candidate in block.ops[first + 1 : last + 1]:
            if isinstance(candidate, (g.ForOp, g.IfOp)):
                return False
            for typ in (g.ElementwiseOp, g.MatmulOp, g.MoveOp):
                if any(
                    isinstance(writer.dst.type, g.WgrmemTensorType)
                    and g.root_buffer(writer.dst) is root
                    for writer in find_ops_of_type(candidate, typ)
                ):
                    return False
        return True

    def redundant_refill_barrier(self, block, index):
        """GMMA completion already releases a single-warpgroup input stage.

        Retain barriers for ordinary stores, initialization, control flow and
        pending readers. Compare physical intervals because roots can alias.
        """
        op = block.ops[index]
        if op.scope not in ("block", "block_leader", "warpgroup"):
            return False
        pure = (
            g.TileOp,
            g.IndexConstantOp,
            g.IndexAddOp,
            g.IndexMulOp,
            g.IndexDivOp,
            g.IndexModOp,
            g.ReinterpretOp,
        )
        previous = index - 1
        while previous >= 0 and isinstance(block.ops[previous], pure):
            previous -= 1
        if (
            previous < 0
            or not isinstance(block.ops[previous], g.AwaitOp)
            or block.ops[previous].lock.type.kind != "wgmma"
        ):
            return False
        if index + 1 >= len(block.ops):
            return False
        refill = block.ops[index + 1]
        if not isinstance(refill, g.AsyncOp) or refill.lock.type.kind != "mbar":
            return False
        transfers = list(refill.body.ops)
        if not transfers or not all(isinstance(t, g.TmaMoveOp) for t in transfers):
            return False
        destinations = {g.root_buffer(t.dst) for t in transfers}
        if not destinations <= self.completed_matrix_reads:
            return False
        pending = (
            set().union(*self.matrix_read_groups) if self.matrix_read_groups else set()
        )

        def interval(root):
            return self.shared[root], self.shared[
                root
            ] + root.type.num_elements * root.type.dtype.byte_size

        return all(
            not (interval(a)[0] < interval(b)[1] and interval(b)[0] < interval(a)[1])
            for a in destinations
            for b in pending
        )

    @staticmethod
    def only_reader(value, reader, writer):
        # Views and additional readers prevent fusion: materialize the value
        # whenever another operation can observe the intermediate result.
        return set(value.users()) <= {reader, writer}

    def broadcast_convert(self, move, conversion):
        """Lower a broadcast fragment load followed by conversion as one unit."""
        memory = self.values[move.src]
        pointer = self.pointer(memory.root)
        desc = self.memory_descriptor()
        block = self.root_fragment(conversion.dst)
        pairs = g.wgmma_view_pairs(conversion.dst)
        if conversion.dst.type.sizes != move.dst.type.sizes:
            raise ValueError("SASS broadcast conversion shapes must match")
        columns = move.dst.type.sizes[-1]
        origin = self.materialize(memory.origin[-1], "R")
        fmt = move.src.type.dtype.name.upper()
        for stripe in range(0, columns, 64):
            lane = self.reg("R")
            column = self.reg("R")
            word = self.reg("R")
            low = self.reg("R")
            high = self.reg("R")
            address = self.reg("R", 2, 2, "broadcast address")
            predicate = self.reg("P")
            self.emit(f"""LOP3.LUT {lane[0]}, {self.tid[0]}, 31, RZ, 0xc0, PT
                SHF.L.U32 {column[0]}, {lane[0]}, 1, RZ
                IADD3 {column[0]}, {column[0]}, {origin}, RZ
                IADD3 {column[0]}, {column[0]}, {stripe}, RZ
                ISETP.LT.U32.AND {predicate[0]}, PT, {column[0]}, {memory.root.type.sizes[-1]}, PT
                MOV {address[0]}, {pointer[0]}
                MOV {address[1]}, {pointer[1]}
                IMAD.WIDE.U32 {address.group()}, {column[0]}, 2, {address.group()}""")
            self.e.mov(word[0], 0)
            self.emit(
                f"@{predicate[0]} LDG.E {word[0]}, desc[{desc.group()}][{address.group()}]",
                wr=5,
                rd=1,
            )
            groups = sorted({col for _, _, col in pairs if stripe <= col < stripe + 64})
            batch_width = 3
            if fmt == "F16":
                self.emit(f"""HADD2.F32 {low[0]}, -RZ, {word[0]}.H0_H0
                    HADD2.F32 {high[0]}, -RZ, {word[0]}.H1_H1""")
            else:
                # BF16 is the upper half of the corresponding F32 encoding.
                # Preserve finite values, signed zero and NaN classification.
                self.emit(f"""SHF.L.U32 {low[0]}, {word[0]}, 16, RZ
                    LOP3.LUT {high[0]}, {word[0]}, 0xffff0000, RZ, 0xc0, PT""")
            self.emit(f"LOP3.LUT {lane[0]}, {lane[0]}, 3, RZ, 0xc0, PT")
            for start in range(0, len(groups), batch_width):
                chunk = groups[start : start + batch_width]
                indices = []
                for j, col in enumerate(chunk):
                    index = self.reg("R")
                    self.emit(f"IADD3 {index[0]}, {lane[0]}, {(col - stripe) // 2}, RZ")
                    indices.append(index)
                for j, (col, index) in enumerate(zip(chunk, indices)):
                    pair = next(p for p, row, c in pairs if row == 0 and c == col)
                    source_index = index if isinstance(index, int) else index[0]
                    clamp = 31
                    self.emit(
                        f"SHFL.IDX PT, {block[2 * pair]}, {low[0]}, {source_index}, {clamp}",
                        wr=2 * j,
                        stall=1,
                    )
                    self.e.pending.clear()
                    self.emit(
                        f"SHFL.IDX PT, {block[2 * pair + 1]}, {high[0]}, {source_index}, {clamp}",
                        wr=2 * j + 1,
                        stall=1,
                    )
                    self.e.pending.clear()
                for j, col in enumerate(chunk):
                    source_pair = next(
                        p for p, row, c in pairs if row == 0 and c == col
                    )
                    for pair, row, c in pairs:
                        if row != 0 and c == col:
                            self.e.pending.add(2 * j)
                            self.e.mov(block[2 * pair], block[2 * source_pair])
                            self.e.pending.add(2 * j + 1)
                            self.e.mov(block[2 * pair + 1], block[2 * source_pair + 1])
        self.fence_needed = True

    def lower(self, op):
        if isinstance(op, g.AllocOp):
            return
        if isinstance(op, g.IndexConstantOp):
            self.values[op.result()] = op.attrs["value"]
            return
        if isinstance(op, g.ConstantOp):
            self.values[op.result()] = op.attrs["value"]
            return
        if isinstance(op, (g.BlockIdOp, g.ThreadIdOp)):
            if isinstance(op, g.ThreadIdOp) and op.axis == g.CollectiveAxis.X:
                self.values[op.result()] = self.tid
                return
            family = "UR" if isinstance(op, g.BlockIdOp) else "R"
            dst = self.reg(family, name="index")
            self.emit(
                f"{'S2UR' if family == 'UR' else 'S2R'} {dst[0]}, SR_{'CTAID' if family == 'UR' else 'TID'}.{op.axis.name.upper()}",
                wr=0,
            )
            self.values[op.result()] = dst
            return
        if isinstance(op, (g.IndexAddOp, g.IndexMulOp, g.IndexDivOp, g.IndexModOp)):
            fn = {
                g.IndexAddOp: "add",
                g.IndexMulOp: "mul",
                g.IndexDivOp: "div",
                g.IndexModOp: "mod",
            }[type(op)]
            value = expression(fn, *(self.values[v] for v in op.operands))
            # Preserve SSA definition placement: CTA coordinates defined
            # outside a loop are computed once, rather than rematerialized
            # inside every TMA owner branch. Their definition dominates uses.
            if isinstance(value, Expr):
                register = self.materialize(value, "UR" if self.uniform(value) else "R")
                value = self.ra.lookup[register]
            self.values[op.result()] = value
            return
        if isinstance(op, g.IndexLtOp):
            dst = self.reg("P")
            a, b = (self.materialize(self.values[v], "R") for v in op.operands)
            self.emit(f"ISETP.LT.U32.AND {dst[0]}, PT, {a}, {b}, PT")
            self.values[op.result()] = dst
            return
        if isinstance(op, g.TileOp):
            self.tile(op)
            return
        if isinstance(op, g.ToPtrOp):
            self.values[op.result()] = self.values[op.operands[0]]
            return
        if isinstance(op, g.PtrAddOp):
            mem = self.values[op.ptr]
            self.values[op.result()] = Memory(
                mem.root,
                expression("add", mem.offset, self.values[op.byte_offset]),
                mem.origin,
            )
            return
        if isinstance(op, g.ReinterpretOp):
            mem = self.values[op.arg]
            self.values[op.result()] = Memory(
                mem.root, expression("add", mem.offset, op.byte_offset), mem.origin
            )
            return
        if isinstance(op, g.ReshapeOp):
            mem = self.values[op.arg]
            if isinstance(op.result().type, g.GmemTensorType) and mem.offset != 0:
                raise ValueError("SASS reshaping an offset global view is unsupported")
            self.values[op.result()] = Memory(
                mem.root, mem.offset, (0,) * op.result().type.rank
            )
            return
        if isinstance(op, g.ForOp):
            self.loop(op)
            return
        if isinstance(op, g.IfOp):
            predicate = self.values[op.operands[0]]
            end = self.label("if_end")
            self.emit(f"@!{predicate[0]} BRA #label({end})")
            self.block(op.blocks[0])
            self.emit("NOP", stall=1)
            self.mark_label(end)
            return
        if isinstance(op, g.AsyncOp):
            self.async_op(op)
            return
        if isinstance(op, g.AwaitOp):
            self.await_op(op)
            return
        if isinstance(op, g.BarrierOp):
            if op.scope == "thread":
                return
            if op.scope not in ("block", "block_leader", "warpgroup"):
                raise ValueError(f"unsupported SASS barrier scope {op.scope}")
            # Every writing warp must publish generic shared stores to the
            # asynchronous proxy before rendezvous. A fence in the TMA
            # owner branch alone cannot publish the other warps' STSM writes.
            if self.shared_writes_before(op):
                self.emit("FENCE.VIEW.ASYNC.S")
            self.emit("BAR.SYNC 0")
            return
        if isinstance(op, g.ElementwiseOp):
            self.elementwise(op)
            return
        if isinstance(op, g.MoveOp):
            self.move(op)
            return
        if isinstance(op, g.MatmulOp):
            raise ValueError("SASS matmul must be lowered into an async group")
        raise ValueError(f"unsupported SASS operation {op.op_id}: {op}")

    @staticmethod
    def shared_writes_before(op):
        """Generic stores since the preceding CTA rendezvous, including regions."""
        block = op.parent_block
        for preceding in reversed(block.ops[: block.ops.index(op)]):
            if isinstance(preceding, g.BarrierOp) and preceding.scope in (
                "block",
                "block_leader",
                "warpgroup",
            ):
                return False
            if any(
                isinstance(move.dst.type, g.SmemTensorType)
                for move in find_ops_of_type(preceding, g.MoveOp)
            ):
                return True
        return False

    def inputs_already_published(self, op):
        if self.shared_writes_before(op):
            return False
        block = op.parent_block
        for preceding in reversed(block.ops[: block.ops.index(op)]):
            if isinstance(preceding, g.BarrierOp) and preceding.scope in (
                "block",
                "block_leader",
                "warpgroup",
            ):
                return self.shared_writes_before(preceding)
            if preceding.blocks:
                return False
        return False

    def tile(self, op):
        if isinstance(op.result().type, g.WgrmemTensorType):
            return
        parent = self.values[op.arg]
        if not isinstance(parent, Memory):
            raise ValueError("SASS tile requires a memory value")
        typ = op.arg.type
        out = op.result().type
        offset = parent.offset
        origin = list(parent.origin or (0,) * typ.rank)
        for d, index in enumerate(op.indices):
            index = index if isinstance(index, int) else self.values[index]
            coordinate = expression("mul", index, out.sizes[d])
            origin[d] = expression("add", origin[d], coordinate)
        for index, stride in op.byte_offset_terms():
            index = index if isinstance(index, int) else self.values[index]
            offset = expression("add", offset, expression("mul", index, stride))
        self.values[op.result()] = Memory(parent.root, offset, tuple(origin))

    def loop(self, op):
        if op.trip_count == 0:
            return
        iv = self.reg("UR", name="loop induction")
        self.e.mov(iv[0], op.lb)
        self.values[op.iv] = iv
        start = self.label("loop")
        self.emit("NOP", stall=1)
        self.mark_label(start)
        first = len(self.e.lines)
        self.block(op.body)
        self.emit(f"UIADD3 {iv[0]}, {iv[0]}, {op.step}, URZ")
        predicate = self.reg("P")
        current = self.materialize(iv, "R")
        self.emit(f"""ISETP.LT.U32.AND {predicate[0]}, PT, {current}, {op.ub}, PT
            @{predicate[0]} BRA #label({start})""")
        self.ra.loops.append((first, len(self.e.lines) - 1))

    def async_op(self, op, owner_selected=False):
        kind = op.lock.type.kind
        if kind == "wgmma":
            mms = list(op.body.ops)
            if not mms or not all(isinstance(mm, g.MatmulOp) for mm in mms):
                raise ValueError("unsupported SASS matrix async group")
            if self.fence_needed:
                # The ISA table gives six cycles from ARRIVE to GMMA readers.
                # PTXAS also uses shorter waits than our default eight cycles.
                self.emit(
                    "WARPGROUP.ARRIVE",
                    stall=6,
                )
                self.fence_needed = False
            blocks = []
            for start in range(0, len(mms), self.descriptor_batch):
                batch = mms[start : start + self.descriptor_batch]
                descriptors = [
                    self.matrix_descriptor(mm, index) for index, mm in enumerate(batch)
                ]
                for j, (mm, descriptor) in enumerate(zip(batch, descriptors)):
                    block = self.root_fragment(mm.dst)
                    blocks.append(block)
                    regs = self.fragment_regs(mm.dst)
                    if len(regs) != mm.rhs.type.sizes[0] // 2:
                        raise ValueError(
                            "SASS HGMMA currently requires a 64-row fragment"
                        )
                    acc = "{" + ",".join(regs) + "}"
                    rc = acc if mm.attrs.get("accumulate", True) else "RZ, !UPT"
                    fmt = ".BF16" if mm.lhs.type.dtype.name == "bf16" else ""
                    if mm.lhs.type.dtype.name not in ("f16", "bf16"):
                        raise ValueError("unsupported SASS matrix input dtype")
                    gsb = ", gsb0" if start + j == len(mms) - 1 else ""
                    self.emit(
                        f"HGMMA.64x{mm.rhs.type.sizes[0]}x16.F32{fmt} {acc}, gdesc[{descriptor[0]}], {rc}{gsb}",
                        stall=4,
                    )
            self.matrix_groups.append(set(blocks))
            self.matrix_read_groups.append(
                {g.root_buffer(v) for mm in mms for v in (mm.lhs, mm.rhs)}
            )
            if len(self.matrix_groups) > 8:
                raise ValueError("SASS IR exceeds eight outstanding GMMA groups")
            return
        if kind not in ("mbar", "bulk"):
            raise ValueError(f"unsupported SASS async lock {kind}")
        transfers = list(op.body.ops)
        if not transfers or not all(isinstance(t, g.TmaMoveOp) for t in transfers):
            raise ValueError("unsupported SASS TMA async body")
        done = None if owner_selected else self.label("tma_done")
        arrival_operand = None
        if done is not None:
            self.emit(f"@!{self.owner[0]} BRA #label({done})", stall=1)
        if kind == "mbar":
            size = sum(
                t.dst.type.num_elements * t.dst.type.dtype.byte_size for t in transfers
            )
            byte_count = self.materialize(size, "R")
            self.emit(
                f"@{self.leader[0]} SYNCS.ARRIVE.TRANS64.RED RZ, [RZ+URZ+{self.shared[op.lock]}], {byte_count}",
                rd=1,
                stall=2,
            )
            arrival_operand = self.ra.lookup[byte_count]
            self.e.pending.clear()
        else:
            if not self.inputs_already_published(op):
                self.emit("FENCE.VIEW.ASYNC.S")
        prepared = [self.prepare_tma(t, op.lock, kind) for t in transfers]
        for index, (instruction, args) in enumerate(prepared):
            self.emit(instruction, rd=1, stall=2)
            if index + 1 < len(prepared):
                self.e.pending.clear()  # All argument blocks remain stable.  # All argument blocks remain stable.
        if kind == "bulk":
            self.emit(f"@{self.uniform_leader[0]} UTMACMDFLUSH", rd=0, stall=2)
        self.emit("NOP", stall=1)
        if arrival_operand is not None:
            self.ra.touch(arrival_operand, len(self.e.lines) - 1)
        for _, args in prepared:
            self.ra.touch(args, len(self.e.lines) - 1)
        if done is not None:
            self.mark_label(done)

    @staticmethod
    def descriptor_bits(value, address):
        typ = value.type
        if (
            not isinstance(typ, g.SmemTensorType)
            or typ.rank != 2
            or typ.sizes[-1] != 16
            or typ.strides[-1] != 1
        ):
            raise ValueError("SASS HGMMA requires row-major K=16 shared tiles")
        width = g.physical_layout(typ)
        stride = width[0] if width else typ.strides[0]
        swizzle = {None: 0, "128B": 1, "64B": 2, "32B": 3}[typ.swizzle]
        return (
            (address >> 4)
            | (16 >> 4) << 16
            | (stride * typ.dtype.byte_size * 8 >> 4) << 32
            | swizzle << 62
        )

    def matrix_descriptor(self, mm, slot):
        descriptor = (
            self.descriptor_pool[slot]
            if self.descriptor_pool
            else self.reg("UR", 4, 4, "GMMA descriptor")
        )
        for value, start in ((mm.lhs, 0), (mm.rhs, 2)):
            address = self.shared_address(value)
            if not isinstance(address, int):
                raise ValueError(
                    "dynamic SASS GMMA descriptor addresses are unsupported"
                )
            bits = self.descriptor_bits(value, address)
            self.e.mov(descriptor[start], bits & 0xFFFFFFFF)
            if not self.descriptor_pool:
                self.e.mov(descriptor[start + 1], bits >> 32)
        return descriptor

    def prepare_tma(self, op, lock, kind):
        if op.multicast:
            raise ValueError("SASS TMA multicast is unsupported")
        value = op.src if kind == "mbar" else op.dst
        desc = value.type
        memory = op.dst if kind == "mbar" else op.src
        coords = [self.values[i] for i in op.indices]
        layout = g.physical_layout(desc)
        if layout:
            width, order = layout
            row, col = coords[-2:]
            block = expression("div", col, width)
            inner = expression("mod", col, width)
            coords = (
                [inner, row, block, *reversed(coords[:-2])]
                if order == "block_major"
                else [inner, block, row, *reversed(coords[:-2])]
            )
        else:
            coords = list(reversed(coords))
        rank = len(coords)
        span = rank + (2 if kind == "mbar" else 1)
        args = self.reg(
            "UR", span, max(8, 1 << (span - 1).bit_length()), "TMA argument block"
        )
        self.materialize(self.shared_address(memory), destination=args[0])
        start = 1
        if kind == "mbar":
            self.materialize(self.shared[lock], destination=args[1])
            start = 2
        for index, coordinate in enumerate(coords):
            self.materialize(coordinate, destination=args[start + index])
        pointer = self.pointer(value)
        policy = ""
        if op.l2_cache_hint:
            cache = self.cache_policy(op.l2_cache_hint)
            policy = f", desc[{cache.group()}]"
        return (
            f"@{self.uniform_leader[0]} {'UTMALDG' if kind == 'mbar' else 'UTMASTG'}.{rank}D [{args[0]}], [{pointer[0]}]{policy}",
            args,
        )

    def await_op(self, op):
        kind = op.lock.type.kind
        if kind == "mbar":
            phase = self.phases[op.lock]
            phase_register = "RZ" if phase is None else phase[0]
            predicate = self.reg("P")
            label = self.label("mbar_wait")
            if self.e.pending:
                self.emit("NOP", stall=1)
            self.mark_label(label)
            self.emit(
                f"SYNCS.PHASECHK.TRANS64.TRYWAIT {predicate[0]}, [URZ+{self.shared[op.lock]}], {phase_register}",
                wr=0,
                stall=2,
            )
            self.emit(f"@!{predicate[0]} BRA #label({label})", stall=1)
            if phase is not None:
                self.emit(f"LOP3.LUT {phase[0]}, {phase[0]}, 0x80000000, RZ, 0x3c, PT")
        elif kind == "wgmma":
            # Pure lane/address work can run while tensor-core work is pending.
            # Only static views already defined at this point may be hoisted.
            for move in find_ops_of_type(self.kernel, g.MoveOp):
                dst = move.dst
                if (
                    isinstance(dst.type, g.SmemTensorType)
                    and self.can_stmatrix(dst)
                    and isinstance(move.src.type, g.WgrmemTensorType)
                    and op.parent_block is self.kernel.body
                    and dst.definition.parent_block is self.kernel.body
                    and dst in self.values
                    and isinstance(self.shared_address(dst), int)
                ):
                    self.stmatrix_base(dst, cache=True)
            if not op.attrs.get("sass_wait_elided"):
                # Keep the hardware completion wait. Only shorten its explicit
                # issue spacing; PTXAS uses two cycles on the measured subset.
                self.emit(
                    f"WARPGROUP.DEPBAR.LE gsb0, {op.pending_groups}",
                    stall=3,
                )
            completed = max(0, len(self.matrix_groups) - op.pending_groups)
            for group in self.matrix_groups[:completed]:
                for block in group:
                    self.ra.touch(block, len(self.e.lines) - 1)
            self.matrix_groups = self.matrix_groups[completed:]
            self.completed_matrix_reads = (
                set().union(*self.matrix_read_groups[:completed])
                if completed
                else set()
            )
            self.matrix_read_groups = self.matrix_read_groups[completed:]
        elif kind == "bulk":
            done = self.label("bulk_done")
            self.emit(f"@!{self.owner[0]} BRA #label({done})", stall=1)
            self.emit("DEPBAR.LE SB0, 0")
            self.e.pending.clear()
            self.mark_label(done)
        else:
            raise ValueError(f"unsupported SASS await {kind}")

    def elementwise(self, op):
        dst = self.fragment_regs(op.dst)
        if op.fn == "fill":
            value = self.values[op.src(0)]
            if not isinstance(value, (int, float)):
                raise ValueError("SASS fill requires a scalar constant")
            if op.dst.type.dtype.name == "f32":
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
            else:
                if op.dst.type.dtype.name == "f16":
                    bits = struct.unpack("<H", struct.pack("<e", value))[0]
                else:
                    word = struct.unpack("<I", struct.pack("<f", value))[0]
                    bits = ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF
                bits |= bits << 16
            block = self.root_fragment(op.dst)
            for start in range(0, len(dst), 2):
                pair = dst[start : start + 2]
                offset = int(pair[0][1:]) - block.virtual_base
                if (
                    bits == 0
                    and len(pair) == 2
                    and offset % 2 == 0
                    and pair[1] == block[offset + 1]
                ):
                    self.emit(f"CS2R {{{pair[0]},{pair[1]}}}, SRZ")
                else:
                    for reg in pair:
                        self.e.mov(reg, bits)
        elif op.fn == "convert":
            src = self.fragment_regs(op.src(0))
            a = op.src(0).type.dtype.name
            b = op.dst.type.dtype.name
            if a == b:
                for d, s in zip(dst, src):
                    self.e.mov(d, s)
            elif a == "f32" and b in ("f16", "bf16"):
                for i, d in enumerate(dst):
                    self.emit(
                        f"F2FP.{b.upper()}.F32.PACK_AB {d}, {src[2 * i + 1]}, {src[2 * i]}"
                    )
            elif a in ("f16", "bf16") and b == "f32":
                for i, s in enumerate(src):
                    if a == "f16":
                        self.emit(f"""HADD2.F32 {dst[2 * i]}, -RZ, {s}.H0_H0
                            HADD2.F32 {dst[2 * i + 1]}, -RZ, {s}.H1_H1""")
                        continue
                    self.emit(f"""SHF.L.U32 {dst[2 * i]}, {s}, 16, RZ
                        LOP3.LUT {dst[2 * i + 1]}, {s}, 0xffff0000, RZ, 0xc0, PT""")
            else:
                raise ValueError(f"unsupported SASS conversion {a} to {b}")
        elif op.fn == "mul" and op.dst.type.dtype.name == "f32":
            a, b = (self.fragment_regs(op.src(i)) for i in (0, 1))
            for d, x, y in zip(dst, a, b):
                self.emit(f"FMUL {d}, {x}, {y}")
        elif op.fn in ("sigmoid", "silu") and op.dst.type.dtype.name == "f32":
            src = self.fragment_regs(op.src(0))
            # Keep four independent element chains in flight. Each EX2 result
            # is waited before its FADD, and each RCP result before writeback.
            # Source/destination aliasing is safe because elements are distinct.
            for start in range(0, len(dst), 4):
                chunk = list(zip(dst[start : start + 4], src[start : start + 4]))
                temps = [self.reg("R") for _ in chunk]
                for (_, x), temp in zip(chunk, temps):
                    self.emit(f"FMUL32I {temp[0]}, {x}, 0fBFB8AA3B")
                for j, temp in enumerate(temps):
                    self.emit(f"MUFU.EX2 {temp[0]}, {temp[0]}", wr=j)
                    self.e.pending.clear()
                for j, temp in enumerate(temps):
                    self.e.pending.add(j)
                    self.emit(f"FADD {temp[0]}, {temp[0]}, 0f3F800000")
                    self.emit(f"MUFU.RCP {temp[0]}, {temp[0]}", wr=j)
                    self.e.pending.clear()
                for j, ((d, x), temp) in enumerate(zip(chunk, temps)):
                    self.e.pending.add(j)
                    if op.fn == "silu":
                        self.emit(f"FMUL {d}, {x}, {temp[0]}")
                    else:
                        self.e.mov(d, temp[0])
        else:
            raise ValueError(
                f"unsupported SASS elementwise operation {op.fn} for {op.dst.type.dtype}"
            )
        self.fence_needed = True

    def lane_coordinates(self):
        warp = self.reg("R")
        lane = self.reg("R")
        row = self.reg("R")
        col = self.reg("R")
        self.emit(f"""SHF.R.U32.HI {warp[0]}, RZ, 5, {self.tid[0]}
            SHF.L.U32 {warp[0]}, {warp[0]}, 4, RZ
            LOP3.LUT {lane[0]}, {self.tid[0]}, 31, RZ, 0xc0, PT
            SHF.R.U32.HI {row[0]}, RZ, 2, {lane[0]}
            IADD3 {row[0]}, {row[0]}, {warp[0]}, RZ
            LOP3.LUT {col[0]}, {lane[0]}, 3, RZ, 0xc0, PT
            SHF.L.U32 {col[0]}, {col[0]}, 1, RZ""")
        return row, col

    def move(self, op):
        if op.src in self.deferred_conversions:
            self.stmatrix(op, converted_source=self.deferred_conversions[op.src])
            return
        load = isinstance(op.dst.type, g.WgrmemTensorType)
        fragment = op.dst if load else op.src
        memory = op.src if load else op.dst
        if not isinstance(fragment.type, g.WgrmemTensorType):
            raise ValueError("SASS move currently requires a register fragment")
        if (
            isinstance(memory.type, g.SmemTensorType)
            and self.can_stmatrix(memory)
            and not load
        ):
            self.stmatrix(op)
            return
        if not isinstance(memory.type, (g.GmemTensorType, g.SmemTensorType)):
            raise ValueError("unsupported SASS move memory")
        if memory.type.dtype != fragment.type.dtype:
            raise ValueError("SASS move dtype mismatch")
        typ = memory.type
        if typ.rank != 2 or typ.strides[-1] != 1:
            raise ValueError("SASS fragment moves require contiguous element pairs")
        if typ.dtype.name not in ("f16", "bf16", "f32"):
            raise ValueError("unsupported SASS fragment move dtype")
        row, col = self.lane_coordinates()
        stride = typ.strides[0] * typ.dtype.byte_size
        offset = self.reg("R")
        self.emit(f"""IMAD {offset[0]}, {row[0]}, {stride}, RZ
            IMAD {offset[0]}, {col[0]}, {typ.dtype.byte_size}, {offset[0]}""")
        mem = self.values[memory]
        global_memory = isinstance(typ, g.GmemTensorType)
        if global_memory:
            pointer = self.pointer(mem.root)
            address = self.reg("R", 2, 2, "global pair address")
            self.emit(f"""MOV {address[0]}, {pointer[0]}
                MOV {address[1]}, {pointer[1]}""")
            view = self.materialize(mem.offset, "R")
            self.emit(f"""IADD3 {offset[0]}, {offset[0]}, {view}, RZ
                IMAD.WIDE.U32 {address.group()}, {offset[0]}, 1, {address.group()}""")
            desc = self.memory_descriptor()
        else:
            base = self.materialize(self.shared_address(memory), "R")
            self.emit(f"IADD3 {offset[0]}, {offset[0]}, {base}, RZ")
        block = self.root_fragment(fragment)
        pairs = self.move_pairs(op, fragment, memory)
        for pair_index, (pair, di, dj) in enumerate(pairs):
            static = di * stride + dj * typ.dtype.byte_size
            regs = block.group(2 * pair, 2) if typ.dtype.name == "f32" else block[pair]
            if op in self.zero_stores:
                regs = "RZ"
            if global_memory:
                addr, displacement = address, static
                if not -(1 << 23) <= static < (1 << 23):
                    addr = self.reg("R", 2, 2, "pair address")
                    self.emit(
                        f"IMAD.WIDE.U32 {addr.group()}, {self.materialize(static, 'R')}, 1, {address.group()}"
                    )
                    displacement = 0
                # Validate tails against the root tensor, including view offsets.
                column = self.reg("R")
                predicate = self.reg("P")
                column_origin = self.materialize(
                    mem.origin[-1] if mem.origin else 0, "R"
                )
                self.emit(f"""IADD3 {column[0]}, {col[0]}, {dj}, RZ
                    IADD3 {column[0]}, {column[0]}, {column_origin}, RZ
                    ISETP.LT.U32.AND {predicate[0]}, PT, {column[0]}, {mem.root.type.sizes[-1]}, PT""")
                if mem.root.type.sizes[-1] % 2:
                    raise ValueError("SASS global half pairs require even root columns")
                if stride:
                    r = self.reg("R")
                    row_origin = self.materialize(
                        mem.origin[-2] if mem.origin else 0, "R"
                    )
                    self.emit(f"""IADD3 {r[0]}, {row[0]}, {di}, RZ
                        IADD3 {r[0]}, {r[0]}, {row_origin}, RZ
                        ISETP.LT.U32.AND {predicate[0]}, PT, {r[0]}, {mem.root.type.sizes[-2]}, {predicate[0]}""")
                if load:
                    if typ.dtype.name == "f32":
                        self.e.mov(block[2 * pair], 0)
                        self.e.mov(block[2 * pair + 1], 0)
                    else:
                        self.e.mov(block[pair], 0)
                opcode = "LDG" if load else "STG"
                self.emit(
                    f"@{predicate[0]} {opcode}.E{'.64' if typ.dtype.name == 'f32' else ''} {regs + ', ' if load else ''}desc[{desc.group()}][{addr.group()}+{displacement}]{'' if load else ', ' + regs}",
                    wr=0 if load else 7,
                    rd=1,
                )
            else:
                self.emit(
                    f"{'LDS' if load else 'STS'}{'.64' if typ.dtype.name == 'f32' else ''} {regs + ', ' if load else ''}[{offset[0]}+{static}]{'' if load else ', ' + regs}",
                    wr=0 if load else 7,
                    rd=1,
                    stall=8 if load else 2,
                )
                if not load and pair_index + 1 < len(pairs):
                    # STS addresses use one stable base and immediate offsets;
                    # every source remains live through the final read wait.
                    self.e.pending.clear()
        if load:
            self.fence_needed = True

    @staticmethod
    def move_pairs(op, fragment, memory):
        # The builder represents output slices with a full source fragment
        # and explicit column metadata, rather than a register TileOp.
        start = op.attrs.get("wgmma_col_start", 0)
        if not isinstance(start, int) or start < 0:
            raise ValueError("SASS fragment column offset must be static")
        pairs = [
            (p, row, col - start)
            for p, row, col in g.wgmma_view_pairs(fragment)
            if row < memory.type.sizes[-2]
            and start <= col < start + memory.type.sizes[-1]
        ]
        if not pairs:
            raise ValueError("empty SASS fragment move slice")
        return pairs

    @staticmethod
    def can_stmatrix(value):
        typ = value.type
        if (
            not isinstance(typ, g.SmemTensorType)
            or typ.rank != 2
            or typ.dtype.name not in ("f16", "bf16")
        ):
            return False
        layout = g.physical_layout(typ)
        return bool(
            layout and layout[1] == "block_major" and layout[0] in (16, 32, 64)
        ) or (
            layout is None
            and typ.swizzle is None
            and typ.strides[-1] == 1
            and typ.sizes[-1] % 8 == 0
            and typ.strides[0] >= typ.sizes[-1]
        )

    def stmatrix_base(self, dst, cache=False):
        if dst in self.store_bases:
            return self.store_bases[dst]
        layout = g.physical_layout(dst.type)
        width, order = layout if layout else (dst.type.strides[0], "row_major")
        if not self.can_stmatrix(dst):
            raise ValueError("unsupported SASS STSM layout")
        if order != "row_major" and (
            order != "block_major" or width not in (16, 32, 64)
        ):
            raise ValueError("unsupported SASS STSM layout")
        warp = self.reg("R")
        lane = self.reg("R")
        row = self.reg("R")
        column = self.reg("R")
        base = self.reg("R")
        self.emit(f"""SHF.R.U32.HI {warp[0]}, RZ, 5, {self.tid[0]}
            LOP3.LUT {lane[0]}, {self.tid[0]}, 31, RZ, 0xc0, PT
            LOP3.LUT {row[0]}, {lane[0]}, 7, RZ, 0xc0, PT
            IMAD {row[0]}, {warp[0]}, 16, {row[0]}
            SHF.R.U32.HI {column[0]}, RZ, 3, {lane[0]}
            SHF.L.U32 {column[0]}, {column[0]}, 3, RZ""")
        if order == "row_major":
            self.emit(f"""IMAD {base[0]}, {row[0]}, {width * 2}, RZ
                IMAD {base[0]}, {column[0]}, 2, {base[0]}""")
            shared = self.shared_address(dst)
            if shared != 0:
                self.emit(
                    f"IADD3 {base[0]}, {base[0]}, {self.materialize(shared, 'R')}, RZ"
                )
            if cache:
                self.store_bases[dst] = base
            return base
        if width < 32:
            extra = self.reg("R")
            self.emit(f"""SHF.R.U32.HI {extra[0]}, RZ, {width.bit_length() - 1}, {column[0]}
                LOP3.LUT {column[0]}, {column[0]}, {width - 1}, RZ, 0xc0, PT
                IMAD {row[0]}, {extra[0]}, 64, {row[0]}""")
        self.emit(f"""IMAD {base[0]}, {row[0]}, {width}, {column[0]}
            SHF.L.U32 {base[0]}, {base[0]}, 1, RZ""")
        # Use the same compact swizzle address convention as the hardware map.
        swizzle = self.reg("R")
        self.emit(f"""LOP3.LUT {swizzle[0]}, {base[0]}, {(width // 8 - 1) << 7}, RZ, 0xc0, PT
            SHF.R.U32.HI {swizzle[0]}, RZ, 3, {swizzle[0]}
            LOP3.LUT {base[0]}, {base[0]}, {swizzle[0]}, RZ, 0x3c, PT""")
        shared = self.shared_address(dst)
        if shared != 0:
            self.emit(
                f"IADD3 {base[0]}, {base[0]}, {self.materialize(shared, 'R')}, RZ"
            )
        if cache:
            self.store_bases[dst] = base
        return base

    def stmatrix(self, op, converted_source=None):
        typ = op.dst.type
        layout = g.physical_layout(typ)
        width, order = layout if layout else (typ.strides[0], "row_major")
        if not self.can_stmatrix(op.dst):
            raise ValueError("unsupported SASS STSM layout")
        # STSM register groups follow four 8x8 matrices, selected from actual
        # fragment view coordinates. No canonical-GEMM reconstruction occurs.
        base = self.stmatrix_base(op.dst)
        source = converted_source if converted_source is not None else op.src
        pairs = self.move_pairs(op, source, op.dst)
        block = self.root_fragment(source)
        if len(pairs) % 2:
            raise ValueError("SASS STSM requires complete matrix pairs")
        # Pair ordering is [low rows, high rows] per eight-column group.
        max_store = 4
        scratch = self.reg("R", max_store, max_store, "STSM operand")
        buffer_count = 2
        double_buffer = buffer_count > 1
        buffers = [scratch]
        for _ in range(buffer_count - 1):
            buffers.append(
                self.reg("R", max_store, max_store, "STSM alternate operand")
            )
        outstanding = set()
        addresses = {base}
        issue = 0
        start = 0
        while start < len(pairs) // 2:
            remaining = len(pairs) // 2 - start
            size = min(max_store, 4 if remaining >= 4 else (2 if remaining >= 2 else 1))
            for parity in (0, 1):
                if double_buffer:
                    slot = issue % buffer_count
                    scratch = buffers[slot]
                    if slot + 1 in outstanding:
                        self.e.pending.add(slot + 1)
                        self.emit("NOP", stall=1)
                        outstanding.remove(slot + 1)
                for j in range(size):
                    pair = pairs[2 * (start + j) + parity][0]
                    if converted_source is None:
                        self.e.mov(scratch[j], block[pair])
                    else:
                        self.emit(
                            f"F2FP.{typ.dtype.name.upper()}.F32.PACK_AB {scratch[j]}, {block[2 * pair + 1]}, {block[2 * pair]}"
                        )
                q = start
                if order == "row_major":
                    off = q * 16 + parity * 8 * width * 2
                    operand_address = f"{base[0]}+{off}"
                else:
                    off = (
                        (q // (width // 8)) * 64 * width * 2
                        + (q % (width // 8)) * 16
                        + parity * 8 * width * 2
                    )
                    low = off % (width * 2)
                    # A zero XOR mask permits a direct byte offset. Nonzero
                    # masks must retain XOR because swizzling changes the
                    # corresponding bits in the lane base.
                    if low == 0:
                        operand_address = f"{base[0]}+{off}"
                    else:
                        address = self.reg("R")
                        addresses.add(address)
                        self.emit(f"""LOP3.LUT {address[0]}, {base[0]}, {low}, RZ, 0x3c, PT
                            IADD3 {address[0]}, {address[0]}, {off - low}, RZ""")
                        operand_address = address[0]
                self.emit(
                    f"STSM.16.M88.{size} [{operand_address}], {scratch.group(size=size) if size > 1 else scratch[0]}",
                    rd=slot + 1 if double_buffer else 1,
                    stall=1,
                )
                if double_buffer:
                    outstanding.add(slot + 1)
                    self.e.pending.clear()
                    issue += 1
            start += size
        if double_buffer:
            self.e.pending.update(outstanding)
            self.emit("NOP", stall=1)
            for operand in [*buffers, *addresses]:
                self.ra.touch(operand, len(self.e.lines) - 1)
