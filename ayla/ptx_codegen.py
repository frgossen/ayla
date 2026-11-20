from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import cast

from . import pytree
from .base.ir import Op, Printer, Value, find_ancestor_of_type, find_ops_of_type
from .base.pipeline import Pass, Pipeline
from .gpu_dialect import (
    BlockIdOp,
    ClusterBlockIdsOp,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    LockType,
    ModuleOp,
    MmaDescriptorType,
    PredicateType,
    PredicateOp,
    PtrAddOp,
    PtrType,
    PTXTemplateOp,
    RmemTensorType,
    SmemTensorType,
    TmemTensorType,
    ThreadIdOp,
    TmaDescType,
    ToPtrOp,
    WarpGroupIdOp,
    WarpIdsOp,
    WgrmemTensorType,
)
from .pipelines.canonicalize import Canonicalize
from .passes.insert_syncs import InsertSyncs
from .passes.licm import LICM
from .passes.lower_to_async import LowerToAsync
from .passes.lower_to_tma_move import LowerToTmaMove
from .passes.pack_smem_allocs import ReuseSmemAllocs
from .passes.verify_gpu_dialect import VerificationStage, VerifyGpuDialect


class HoistMmaDescriptors(Pass):
    """LICM of MMA descriptor preparation, which becomes SSA only in LowerToPtxTemplate.

    It hoists through nested issuer guards; the outer-first walk needs another
    round for each enclosing loop.
    """

    def run(self, module: ModuleOp) -> ModuleOp:
        if module.target != "sm_100a" or not module.body.ops[0].attrs.get("hoist_mma_descriptors", False):
            return module
        while True:
            before = repr(module)
            LICM(hoist_pure_through_guards=True).run(module)
            Canonicalize().run(module)
            if repr(module) == before:
                return module


class PtxLoweringPipeline(Pipeline):
    """Lowers a high-level ModuleOp down to a PTX-template ModuleOp."""

    def __init__(self, *, bulk_source_queue_completion: bool = False) -> None:
        super().__init__()
        self.add_instrumentation_pass(
            VerifyGpuDialect(VerificationStage.PTX_LOWERING)
        )
        self.add_pass(LowerToTmaMove())
        self.add_pass(LowerToAsync())
        self.add_pass(Canonicalize())
        self.add_pass(InsertSyncs(bulk_source_queue_completion=bulk_source_queue_completion))
        self.add_pass(ReuseSmemAllocs())
        from .passes.lower_to_ptx_template import LowerToPtxTemplate
        self.add_pass(LowerToPtxTemplate())
        self.add_pass(HoistMmaDescriptors())
        self.add_pass(Canonicalize())


# Matches placeholders like ${r:tid}, ${rd:operand[0].ptr}, ${label:exit}.
PLACEHOLDER_RE = r"\$\{(?P<ty>[^:}]+):(?P<name>[^}]+)\}"


@dataclass
class TypeRegisterMapping:
    member_register_types: tuple | PtxType
    member_names: tuple | str
    param_decl: str = ".param .{type} {name}"
    param_load: str = "ld.param.{type} {reg}, [{name}];"


class PtxType(str, Enum):
    """PTX scalar type codes used in `.reg`/`ld`/etc. instructions.

    `LABEL` is not a runtime register type — it's a marker for label
    name allocation, unified into this enum so codegen can dispatch
    placeholders uniformly.
    """

    B8 = "b8"
    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    F32 = "f32"
    F64 = "f64"
    PRED = "pred"
    LABEL = "label"
    SMEM = "smem"

    def __str__(self) -> str:
        return self.value


# nvcc PTX prefix convention per `PtxType`. Used as the basename for
# `.reg`-declared register names and (for LABEL) for label names.
BASENAME_BY_PTX_TYPE = {
    PtxType.B8: "%rb",
    PtxType.B16: "%rs",
    PtxType.B32: "%r",
    PtxType.B64: "%rd",
    PtxType.F32: "%f",
    PtxType.F64: "%fd",
    PtxType.PRED: "%p",
    PtxType.LABEL: "$L",
    PtxType.SMEM: "_smem_",
}

# Namespace used by `unique_ptx_names` / `unique_ptx_names_for_value`.
# Covers all names declared in the kernel body — registers and labels
# (PtxType.LABEL) — which share a single unique-name allocator. Distinct
# from `PTX_PARAM_NAMESPACE`, which covers names in the `.param` list.
PTX_BODY_NAMESPACE = "ptx_body"


# Namespace used by `unique_param_names_for_value`.
PTX_PARAM_NAMESPACE = "ptx_params"


def map_type_to_registers(type):
    if isinstance(type, GmemTensorType):
        return TypeRegisterMapping(
            member_register_types=PtxType.B64,
            member_names="ptr",
            param_load=(
                "ld.param.{type} {reg}, [{name}];\n"
                "cvta.to.global.u64 {reg}, {reg};"
            ),
        )
    if isinstance(type, RmemTensorType):
        n = type.num_elements
        assert n is not None
        # 16-bit floats are moved as raw bits in register memory.
        ptx_dtype = (
            PtxType.B16 if type.dtype.name in ("bf16", "f16")
            else PtxType(type.dtype.name)
        )
        return TypeRegisterMapping(
            member_register_types=tuple(ptx_dtype for _ in range(n)),
            member_names=tuple(f"r{i}" for i in range(n)),
        )
    if isinstance(type, IndexType):
        return TypeRegisterMapping(
            member_register_types=PtxType.B32,
            member_names="val",
        )
    if isinstance(type, PredicateType):
        return TypeRegisterMapping(
            member_register_types=PtxType.PRED,
            member_names="val",
        )
    if isinstance(type, MmaDescriptorType):
        return TypeRegisterMapping(member_register_types=PtxType.B64, member_names="val")
    if isinstance(type, TmaDescType):
        return TypeRegisterMapping(
            member_register_types=PtxType.B64,
            member_names="ptr",
            param_decl=".param .align 128 .b8 {name}[128]",
            param_load=(
                "mov.b64 {reg}, {name};\n"
                "cvta.param.u64 {reg}, {reg};"
            ),
        )
    if isinstance(type, WgrmemTensorType):
        num_elements_per_thread = type.num_elements_per_thread
        if type.dtype.name in ("bf16", "f16"):
            n_regs = num_elements_per_thread // 2
            ptx_dtype = PtxType.B32
        elif type.dtype.name == "f32":
            n_regs = num_elements_per_thread
            ptx_dtype = PtxType.F32
        else:
            raise NotImplementedError(
                f"map_type_to_registers: wgrmem dtype {type.dtype.name}"
            )
        return TypeRegisterMapping(
            member_register_types=tuple(ptx_dtype for _ in range(n_regs)),
            member_names=tuple(f"r{i}" for i in range(n_regs)),
        )
    if isinstance(type, (SmemTensorType, TmemTensorType)):
        return TypeRegisterMapping(
            member_register_types=PtxType.B32,
            member_names="ptr",
        )
    if isinstance(type, PtrType):
        ptx_type = PtxType.B64 if type.space == "gmem" else PtxType.B32
        return TypeRegisterMapping(
            member_register_types=ptx_type,
            member_names="ptr",
        )
    if isinstance(type, LockType):
        if type.kind in ("mbar", "tcgen05"):
            return TypeRegisterMapping(
                member_register_types=(PtxType.B32, PtxType.B32),
                member_names=("val", "parity"),
            )
        if type.kind == "cluster_mbar":
            return TypeRegisterMapping(
                member_register_types=(PtxType.B32, PtxType.B32),
                member_names=("val", "parity"),
            )
        return TypeRegisterMapping(
            member_register_types=(),
            member_names=(),
        )
    raise NotImplementedError(f"map_type_to_registers: unsupported type: {type}")


class PtxPrinter(Printer):
    """Printer with PTX-specific naming helpers."""

    def __init__(self, indent_str: str = "  ") -> None:
        super().__init__(indent_str)
        self._ptx_names_by_type: dict[PtxType, set[str]] = defaultdict(set)
        # Shared-memory arena (see `_layout_shared_arena`): symbol ->
        # (arena, base register, offset), and the registers known to hold
        # `base + offset`.
        self.shared_offsets: dict[str, tuple[str, str, int]] = {}
        self.pack_shared_declarations = False
        self.shared_declarations: list[tuple[int, str]] = []
        self.shared_addresses: dict[str, tuple[str, int]] = {}
        self.carried_registers: dict[Value, str] = {}
        self.skipped_recurrences: set[Op] = set()

    def unique_param_names_for_value(self, value: Value):
        mapping = map_type_to_registers(value.type)
        base = value.name if value.name is not None else "param"
        basenames = pytree.map(
            lambda n: f"{base}_{n}",
            mapping.member_names,
        )
        return self.unique_names(
            value,
            basenames,
            namespace=PTX_PARAM_NAMESPACE,
        )

    def unique_ptx_names_for_value(self, value: Value):
        if value in self.carried_registers:
            return self.carried_registers[value]
        mapping = map_type_to_registers(value.type)
        return self.unique_ptx_names(value, mapping.member_register_types)

    def unique_ptx_names(
        self,
        obj: object,
        reg_types: tuple | PtxType,
    ) -> str | tuple:
        basenames = pytree.map(
            lambda t: BASENAME_BY_PTX_TYPE[t],
            reg_types,
        )
        names = self.unique_names(
            obj,
            basenames,
            namespace=PTX_BODY_NAMESPACE,
            always_suffix=True,
        )
        for t, name in zip(
            pytree.flatten(reg_types),
            pytree.flatten(names),
        ):
            self._ptx_names_by_type[t].add(name)
        return names

    def ptx_name_count(self, ptx_type: PtxType) -> int:
        """Number of unique names minted for `ptx_type`."""
        return len(self._ptx_names_by_type[ptx_type])


@dataclass
class PtxKernel:
    """A self-contained PTX program: source plus everything the runner needs
    to load and launch it, including entry point, launch dimensions, shared
    memory, and host-side JIT settings.
    """

    ptx: str
    name: str
    signature: list
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    cluster_dim: tuple[int, int, int] | None = None
    shared_mem_bytes: int = 0
    ptx_optimization_level: int | None = None


def _resolve_block_dim(kernel: KernelOp) -> tuple[int, int, int]:
    """Resolve implicit block axes for PTX emission and launch."""
    if all(dim is None for dim in kernel.block_dim):
        raise ValueError(
            f"make_ptx_kernel: block_dim is entirely unset: {kernel.block_dim}"
        )
    return tuple(
        dim if dim is not None else 1 for dim in kernel.block_dim
    )  # type: ignore[return-value]


def _pack_static_shared_declarations(ptx: str) -> str:
    """Preserve allocations and instructions while reducing alignment padding."""
    lines = ptx.splitlines()
    pattern = re.compile(r"^\s*\.shared\s+\.align\s+(\d+)\s+\.b(8|16|32|64)\s+([A-Za-z_$][\w$]*)\[(\d+)\];\s*$")
    declarations, remaining = [], []
    for line in lines:
        match = pattern.match(line)
        if match:
            alignment, width, name, count = int(match[1]), int(match[2]), match[3], int(match[4])
            if alignment < 1 or alignment & (alignment - 1) or count < 1:
                raise ValueError("invalid static shared allocation for packing")
            declarations.append((alignment, width // 8 * count, name, line.strip()))
        else:
            if re.match(r"^\s*(?:\.extern\s+)?\.shared\b", line):
                raise ValueError("shared packing requires known static array declarations")
            remaining.append(line)
    if not declarations:
        return ptx
    if len({d[2] for d in declarations}) != len(declarations) or len(re.findall(r"\.entry\s", ptx)) != 1 or re.search(r"\.func\b", ptx):
        raise ValueError("shared packing requires one kernel and unique shared symbols")
    packed = sorted(declarations, key=lambda d: -d[0])
    def footprint(items):
        end = 0
        for alignment, size, _, _ in items:
            end = (end + alignment - 1) // alignment * alignment + size
        return end
    if footprint(packed) >= footprint(declarations):
        return ptx
    insertion = next(i for i, line in enumerate(remaining) if line.strip() == "{") + 1
    return "\n".join(remaining[:insertion] + ["  " + d[3] for d in packed] + remaining[insertion:]) + "\n"


def make_ptx_kernel(module: ModuleOp, *, bulk_source_queue_completion: bool = False) -> PtxKernel:
    """Lower a single-kernel ModuleOp to a PtxKernel."""
    top_ops = module.body.ops
    if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
        raise ValueError("make_ptx_kernel: module must contain exactly one KernelOp")
    kernel: KernelOp = top_ops[0]
    module = PtxLoweringPipeline(bulk_source_queue_completion=bulk_source_queue_completion).run(module)

    if all(d is None for d in kernel.grid_dim):
        raise ValueError(f"make_ptx_kernel: grid_dim is entirely unset: {kernel.grid_dim}")
    grid_dim = cast(
        tuple[int, int, int],
        tuple(d if d is not None else 1 for d in kernel.grid_dim),
    )
    block_dim = _resolve_block_dim(kernel)
    cluster_dim = (
        cast(
            tuple[int, int, int],
            tuple(d if d is not None else 1 for d in kernel.cluster_dim),
        )
        if any(d is not None for d in kernel.cluster_dim)
        else None
    )
    p = PtxPrinter()
    _emit_module(p, module, block_dim, cluster_dim)
    ptx = p.render()
    if kernel.attrs.get("rolled_shared_ring") and kernel.attrs.get("persistent_output_slots") == 1:
        ptx = _pack_static_shared_declarations(ptx)
    return PtxKernel(
        ptx=ptx,
        name=kernel.name,
        signature=kernel.signature,
        grid_dim=grid_dim,
        block_dim=block_dim,
        cluster_dim=cluster_dim,
        ptx_optimization_level=kernel.ptx_optimization_level,
    )


def _emit_op(p: PtxPrinter, op: Op) -> None:
    if op in p.skipped_recurrences:
        return
    if isinstance(op, ForOp):
        _emit_for(p, op)
    elif isinstance(op, IfOp):
        _emit_if(p, op)
    elif isinstance(op, PTXTemplateOp):
        _emit_ptx_template(p, op)
    elif isinstance(op, ThreadIdOp):
        _emit_thread_id(p, op)
    elif isinstance(op, BlockIdOp):
        _emit_block_id(p, op)
    elif isinstance(op, ClusterBlockIdsOp):
        _emit_cluster_block_ids(p, op)
    elif isinstance(op, IndexConstantOp):
        _emit_index_constant(p, op)
    elif isinstance(op, IndexAddOp):
        _emit_index_add(p, op)
    elif isinstance(op, IndexMulOp):
        _emit_index_mul(p, op)
    elif isinstance(op, IndexDivOp):
        _emit_index_div(p, op)
    elif isinstance(op, IndexLtOp):
        _emit_index_lt(p, op)
    elif isinstance(op, PredicateOp):
        dst = p.unique_ptx_names_for_value(op.result())
        operands = ", ".join(p.unique_ptx_names_for_value(value) for value in op.operands)
        _write_with_comment(p, f"{op.attrs['kind']}.pred {dst}, {operands};", op)
    elif isinstance(op, IndexModOp):
        _emit_index_mod(p, op)
    elif isinstance(op, WarpIdsOp):
        _emit_warp_ids(p, op)
    elif isinstance(op, WarpGroupIdOp):
        _emit_warp_group_id(p, op)
    elif isinstance(op, ToPtrOp):
        _emit_to_ptr(p, op)
    elif isinstance(op, PtrAddOp):
        _emit_ptr_add(p, op)
    else:
        raise NotImplementedError(f"make_ptx_kernel: not implemented for {op.op_id}")


def _emit_block(p: PtxPrinter, block) -> None:
    for op in block.ops:
        _emit_op(p, op)


def _emit_for(p: PtxPrinter, op: ForOp) -> None:
    iv_reg = p.unique_ptx_names(op.iv, PtxType.B32)
    label = p.unique_ptx_names(op, PtxType.LABEL)
    pred = p.unique_ptx_names((op, "pred"), PtxType.PRED)
    p.write_newline_once()
    p.write_line(f"// BEGIN: `{_op_description(p, op)}`")
    p.write_line(f"mov.b32 {iv_reg}, {op.lb};")
    from .ptx_rolled_loop import RolledLoopRecurrences

    previous_skips, previous_registers = p.skipped_recurrences.copy(), p.carried_registers.copy()
    recurrence = RolledLoopRecurrences(p, op, PtxType, _emit_op)
    recurrence.initialize()
    p.skipped_recurrences |= recurrence.skipped
    p.write_line(f"{label}:")
    p.write_newline_once()
    if op.attrs.get("unroll") is False:
        p.write_line('.pragma "nounroll";')
    _emit_block(p, op.body)
    p.write_line(f"// END: `{_op_description(p, op)}`")
    recurrence.advance()
    p.write_line(f"add.s32 {iv_reg}, {iv_reg}, {op.step};")
    p.write_line(f"setp.lt.s32 {pred}, {iv_reg}, {op.ub};")
    p.write_line(f"@{pred} bra {label};")
    p.skipped_recurrences, p.carried_registers = previous_skips, previous_registers
    p.write_newline_once()


def _emit_if(p: PtxPrinter, op: IfOp) -> None:
    condition = p.unique_ptx_names_for_value(op.condition)
    done = p.unique_ptx_names(op, PtxType.LABEL)
    p.write_line(f"@!{condition} bra {done};")
    _emit_block(p, op.body)
    p.write_line(f"{done}:")


def _emit_module(
    p: PtxPrinter,
    op: ModuleOp,
    block_dim: tuple[int, int, int],
    cluster_dim: tuple[int, int, int] | None,
) -> None:
    p.write_line(f".version {op.version}")
    p.write_line(f".target {op.target}")
    p.write_line(f".address_size {op.address_size}")
    p.write_newline_once()
    [kernel] = op.body.ops
    assert isinstance(kernel, KernelOp)
    _emit_kernel(p, kernel, block_dim, cluster_dim)


def _emit_kernel(
    p: PtxPrinter,
    op: KernelOp,
    block_dim: tuple[int, int, int],
    cluster_dim: tuple[int, int, int] | None,
) -> None:

    p.write_line(f".visible .entry {op.name}(")
    with p.indented():
        param_lines: list[str] = []
        for arg in op.body.arguments:
            mapping = map_type_to_registers(arg.type)
            types = list(pytree.flatten(mapping.member_register_types))
            names = list(pytree.flatten(
                p.unique_param_names_for_value(arg),
            ))
            for t, name in zip(types, names):
                param_lines.append(
                    mapping.param_decl.format(type=t, name=name)
                )
        for i, line in enumerate(param_lines):
            sep = "," if i < len(param_lines) - 1 else ""
            p.write_line(f"{line}{sep}")
    p.write_line(")")
    p.write_line(f".reqntid {block_dim[0]}, {block_dim[1]}, {block_dim[2]}")
    if cluster_dim is not None:
        p.write_line(
            ".reqnctapercluster "
            f"{cluster_dim[0]}, {cluster_dim[1]}, {cluster_dim[2]}"
        )
    p.write_line("{")
    with p.indented():
        decl_mark = p.mark()
        for arg in op.body.arguments:
            mapping = map_type_to_registers(arg.type)
            types = list(pytree.flatten(mapping.member_register_types))
            reg_names = list(pytree.flatten(
                p.unique_ptx_names_for_value(arg),
            ))
            param_names = list(pytree.flatten(
                p.unique_param_names_for_value(arg),
            ))
            for t, rd, param in zip(types, reg_names, param_names):
                for line in mapping.param_load.format(
                    type=t, reg=rd, name=param,
                ).splitlines():
                    p.write_line(line)
        p.write_newline_once()
        arena = _layout_shared_arena(p, op)
        p.pack_shared_declarations = bool(op.attrs.get("pack_shared_declarations"))
        _emit_tma_descriptor_prefetch(p, op, block_dim)
        _emit_block(p, op.body)
        p.write_line("ret;")
    p.write_line("}")

    with p.insert_at(decl_mark):
        if arena:
            p.write_line(arena)
        # Small event declarations before swizzled buffers can waste almost
        # 1 KiB on alignment, enough to lose a resident CTA. These symbols are
        # statically allocated for the kernel; declaration order need not
        # follow initialization order. Pack higher alignments first.
        for _, declaration in sorted(p.shared_declarations, key=lambda item: -item[0]):
            p.write_line(declaration)
        for ptx_type in PtxType:
            if ptx_type in (PtxType.LABEL, PtxType.SMEM):
                continue
            n = p.ptx_name_count(ptx_type)
            if n > 0:
                prefix = BASENAME_BY_PTX_TYPE[ptx_type]
                p.write_line(f".reg .{ptx_type:4s} {prefix}<{n}>;")
        p.write_newline_once()


_SHARED_DECLARATION_RE = re.compile(
    r"\.shared\s+\.align\s+(?P<align>\d+)\s+\.b(?P<bits>\d+)\s+"
    r"\$\{smem:(?P<name>[^}]+)\}\[(?P<count>\d+)\];"
)
_MAPA_SOURCE_RE = re.compile(
    r"mapa\.shared::cluster\.u32\s+[^,]+,\s*\$\{b32:operand\.(?P<operand>\w+)\.\w+\}"
)


def _layout_shared_arena(p: PtxPrinter, kernel: KernelOp) -> str | None:
    """Place the shared symbols that cluster `mapa`s translate in one arena.

    ptxas keeps one remote address per `mapa` source live in a uniform
    register; with ~30 per-bank sources it runs out and moves bases to vector
    registers (+IADD3/R2UR per operand). `mapa` is affine in the shared window,
    so `mapa(arena + offset)` becomes `mapa(arena) + offset` (immediate
    offsets): one remote base per rank. Other symbols stay separate, since
    single-CTA kernels regress when arena addresses get CSEd into TMA regions.

    Writes the arena base register and returns the arena declaration.
    """
    templates = find_ops_of_type(kernel, PTXTemplateOp)
    sources = {
        source
        for op in templates
        for m in _MAPA_SOURCE_RE.finditer(op.template)
        if op.operand_names and m["operand"] in op.operand_names
        and isinstance(source := op.operands[op.operand_names.index(m["operand"])].definition,
                       PTXTemplateOp)
    }
    declarations = [
        (int(m["align"]), int(m["bits"]) // 8 * int(m["count"]),
         cast(str, p.unique_ptx_names((op, m["name"]), PtxType.SMEM)))
        for op in templates if op in sources
        for m in _SHARED_DECLARATION_RE.finditer(op.template)
    ]
    if not declarations:
        return None
    arena = cast(str, p.unique_ptx_names((kernel, "shared_arena"), PtxType.SMEM))
    base = cast(str, p.unique_ptx_names((kernel, "shared_arena_base"), PtxType.B32))
    # Higher alignments first: no padding between events and swizzled buffers.
    size = 0
    for alignment, n_bytes, symbol in sorted(declarations, key=lambda d: -d[0]):
        size = -(-size // alignment) * alignment
        p.shared_offsets[symbol], size = (arena, base, size), size + n_bytes
    p.write_line(f"mov.u32 {base}, {arena};")
    p.write_newline_once()
    return f".shared .align {max(d[0] for d in declarations)} .b8 {arena}[{size}];"


def _emit_tma_descriptor_prefetch(
    p: PtxPrinter, op: KernelOp, block_dim: tuple[int, int, int]
) -> None:
    """Warm immutable tensor-map parameters once per CTA from warp zero.

    The parameter loads have already converted these addresses to generic
    pointers. This advisory operation needs no new synchronization and stays
    outside every operand/output loop. Omission emits byte-identical PTX.
    """
    enabled = op.attrs.get("prefetch_tma_descriptors", False)
    if type(enabled) is not bool:
        raise ValueError("prefetch_tma_descriptors must be a boolean")
    descriptors = [arg for arg in op.body.arguments if isinstance(arg.type, TmaDescType)]
    if not enabled or not descriptors:
        return
    thread = [p.unique_ptx_names((op, "descriptor_prefetch", axis), PtxType.B32)
              for axis in "xyz"]
    elected = p.unique_ptx_names((op, "descriptor_prefetch"), PtxType.PRED)
    done = p.unique_ptx_names((op, "descriptor_prefetch_done"), PtxType.LABEL)
    p.write_line("// Advisory TMA descriptor prefetch, once per CTA.")
    for axis, reg in zip("xyz", thread):
        p.write_line(f"mov.u32 {reg}, %tid.{axis};")
    p.write_line(f"mad.lo.u32 {thread[1]}, {thread[2]}, {block_dim[1]}, {thread[1]};")
    p.write_line(f"mad.lo.u32 {thread[0]}, {thread[1]}, {block_dim[0]}, {thread[0]};")
    # Warp membership follows the linear thread index, including XYZ blocks.
    # Every active lane in the first warp has the same descriptor pointer;
    # the hardware emits one uniform UTMACCTL.PF per descriptor, avoiding a
    # divergent single-lane region and its reconvergence bookkeeping.
    p.write_line(f"setp.lt.u32 {elected}, {thread[0]}, 32;")
    p.write_line(f"@!{elected} bra.uni {done};")
    for arg in descriptors:
        pointer = p.unique_ptx_names_for_value(arg)
        p.write_line(f"prefetch.tensormap [{pointer}];")
    p.write_line(f"{done}:")
    p.write_newline_once()


def _emit_ptx_template(p: PtxPrinter, op: PTXTemplateOp) -> None:

    placeholder_values: dict[str, str] = {}

    def bind_values(role: str, values, names: list[str] | None = None) -> None:
        for i, value in enumerate(values):
            mapping = map_type_to_registers(value.type)
            member_names = pytree.flatten(mapping.member_names)
            reg_names = pytree.flatten(
                p.unique_ptx_names_for_value(value),
            )
            bases = [f"{role}[{i}]"]
            if names:
                bases.append(f"{role}.{names[i]}")
            for mname, rname in zip(member_names, reg_names):
                for base in bases:
                    key = base if mname == "" else f"{base}.{mname}"
                    placeholder_values[key] = rname

    bind_values("operand", op.operands, op.operand_names)
    bind_values("result", op.results, op.result_names)

    def repl(m: re.Match) -> str:
        ty, name = m.group("ty"), m.group("name")
        if name not in placeholder_values:
            placeholder_values[name] = cast(
                str,
                p.unique_ptx_names((op, name), PtxType(ty)),
            )
        return placeholder_values[name]

    expanded = re.sub(PLACEHOLDER_RE, repl, op.template)

    p.write_newline_once()
    p.write_line(f"// PTX template `{_op_description(p, op)}`")
    for line in expanded.splitlines():
        _write_template_line(p, line.strip())
    p.write_newline_once()


def _write_template_line(p: PtxPrinter, line: str) -> None:
    """Write one template line with shared symbols placed in the arena."""
    declaration = re.fullmatch(r"\.shared\s+\.align\s+\d+\s+\.b\d+\s+(\S+)\[\d+\];", line)
    if declaration and declaration[1] in p.shared_offsets:
        return
    address = re.fullmatch(r"mov\.u32\s+(%\w+),\s*(\w+);", line)
    remote = re.fullmatch(r"mapa\.shared::cluster\.u32\s+(%\w+),\s*(%\w+),\s*(%\w+);", line)
    destination = re.match(r"(?:@!?%\w+\s+)?[a-z][\w.:]*\s+(%\w+)\s*,", line)
    if destination:
        p.shared_addresses.pop(destination[1], None)
    if address and address[2] in p.shared_offsets:
        arena, base, offset = p.shared_offsets[address[2]]
        p.write_line(f"mov.u32 {address[1]}, {arena}+{offset};")
        p.shared_addresses[address[1]] = (base, offset)
    elif remote and remote[2] in p.shared_addresses:
        dst, source, rank = remote.groups()
        base, offset = p.shared_addresses[source]
        p.write_line(f"mapa.shared::cluster.u32 {dst}, {base}, {rank};")
        p.write_line(f"add.u32 {dst}, {dst}, {offset};")
    else:
        for symbol in re.findall(r"\[(\w+)\]", line):
            if symbol in p.shared_offsets:
                arena, _, offset = p.shared_offsets[symbol]
                line = line.replace(f"[{symbol}]", f"[{arena}+{offset}]")
        if declaration and p.pack_shared_declarations:
            p.shared_declarations.append((int(line.split()[2]), line))
        else:
            p.write_line(line)


def _emit_thread_id(p: PtxPrinter, op: ThreadIdOp) -> None:
    result = p.unique_ptx_names_for_value(op.value)
    axis = op.axis.name.lower()
    p.write_newline_once()
    p.write_line(f"mov.u32 {result}, %tid.{axis};")
    p.write_newline_once()


def _emit_block_id(p: PtxPrinter, op: BlockIdOp) -> None:
    result = p.unique_ptx_names_for_value(op.value)
    axis = op.axis.name.lower()
    p.write_newline_once()
    p.write_line(f"mov.u32 {result}, %ctaid.{axis};")
    p.write_newline_once()


# TODO: Make the cluster block ID op take an axis attribute and return one
# value instead of always materializing x, y, z.
def _emit_cluster_block_ids(p: PtxPrinter, op: ClusterBlockIdsOp) -> None:
    rx = p.unique_ptx_names_for_value(op.x)
    ry = p.unique_ptx_names_for_value(op.y)
    rz = p.unique_ptx_names_for_value(op.z)
    p.write_newline_once()
    p.write_line(f"mov.u32 {rx}, %cluster_ctaid.x;")
    p.write_line(f"mov.u32 {ry}, %cluster_ctaid.y;")
    p.write_line(f"mov.u32 {rz}, %cluster_ctaid.z;")
    p.write_newline_once()


# TODO: Emit S index computations instead.
def _emit_warp_ids(p: PtxPrinter, op: WarpIdsOp) -> None:
    # Broadcast from the warp's first thread (CUTLASS's
    # canonical_warp_idx_sync), so ptxas can prove each index warp-uniform and
    # branch on it without diverging.
    p.write_newline_once()
    for value, sreg, shift in (
        (op.x, "%tid.x", 5), (op.y, "%tid.y", 0), (op.z, "%tid.z", 0),
    ):
        reg = p.unique_ptx_names_for_value(value)
        p.write_line(f"mov.u32 {reg}, {sreg};")
        if shift:
            p.write_line(f"shr.u32 {reg}, {reg}, {shift};")
        p.write_line(f"shfl.sync.idx.b32 {reg}, {reg}, 0, 31, 0xffffffff;")
    p.write_newline_once()


def _emit_warp_group_id(p: PtxPrinter, op: WarpGroupIdOp) -> None:
    rx = p.unique_ptx_names_for_value(op.x)
    ry = p.unique_ptx_names_for_value(op.y)
    rz = p.unique_ptx_names_for_value(op.z)
    p.write_newline_once()
    p.write_line(f"mov.u32 {rx}, %tid.x;")
    p.write_line(f"shr.u32 {rx}, {rx}, 7;")
    p.write_line(f"mov.u32 {ry}, 0;")
    p.write_line(f"mov.u32 {rz}, 0;")
    p.write_newline_once()


def _write_with_comment(p: PtxPrinter, instr: str, op: Op, col: int = 30) -> None:
    """Write an instruction with an aligned trailing comment."""
    comment = _op_description(p, op)
    padding = max(1, col - len(instr))
    p.write_line(f"{instr}{' ' * padding}// {comment}")


def _const_val(v: Value) -> int | None:
    if isinstance(v.definition, IndexConstantOp):
        return v.definition.value
    return None


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _emit_index_constant(p: PtxPrinter, op: IndexConstantOp) -> None:
    reg = p.unique_ptx_names_for_value(op.result())
    _write_with_comment(p, f"mov.u32 {reg}, {op.value};", op)


def _emit_index_add(p: PtxPrinter, op: IndexAddOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    lhs = p.unique_ptx_names_for_value(op.lhs)
    rhs_val = _const_val(op.rhs)
    if rhs_val is not None:
        _write_with_comment(p, f"add.s32 {dst}, {lhs}, {rhs_val};", op)
    else:
        rhs = p.unique_ptx_names_for_value(op.rhs)
        _write_with_comment(p, f"add.s32 {dst}, {lhs}, {rhs};", op)


def _emit_index_lt(p: PtxPrinter, op: IndexLtOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    lhs = p.unique_ptx_names_for_value(op.lhs)
    rhs = p.unique_ptx_names_for_value(op.rhs)
    _write_with_comment(p, f"setp.lt.u32 {dst}, {lhs}, {rhs};", op)


def _emit_index_mul(p: PtxPrinter, op: IndexMulOp) -> None:
    # TODO: Raise power-of-2 shift optimization to the IR level (IndexShlOp).
    dst = p.unique_ptx_names_for_value(op.result())
    lhs = p.unique_ptx_names_for_value(op.lhs)
    rhs_val = _const_val(op.rhs)
    lhs_val = _const_val(op.lhs)
    if rhs_val is not None and _is_power_of_2(rhs_val):
        shift = rhs_val.bit_length() - 1
        _write_with_comment(p, f"shl.b32 {dst}, {lhs}, {shift};", op)
    elif lhs_val is not None and _is_power_of_2(lhs_val):
        rhs = p.unique_ptx_names_for_value(op.rhs)
        shift = lhs_val.bit_length() - 1
        _write_with_comment(p, f"shl.b32 {dst}, {rhs}, {shift};", op)
    elif rhs_val is not None:
        _write_with_comment(p, f"mul.lo.s32 {dst}, {lhs}, {rhs_val};", op)
    else:
        rhs = p.unique_ptx_names_for_value(op.rhs)
        _write_with_comment(p, f"mul.lo.s32 {dst}, {lhs}, {rhs};", op)


def _emit_index_div(p: PtxPrinter, op: IndexDivOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    lhs = p.unique_ptx_names_for_value(op.lhs)
    rhs_val = _const_val(op.rhs)
    if rhs_val is not None and _is_power_of_2(rhs_val):
        shift = rhs_val.bit_length() - 1
        _write_with_comment(p, f"shr.b32 {dst}, {lhs}, {shift};", op)
    elif rhs_val is not None and 1 < rhs_val < (1 << 32) and _integer_index_division(op):
        _emit_unsigned_constant_divmod(p, op, dst, lhs, rhs_val, remainder=False)
    elif rhs_val is not None:
        _write_with_comment(p, f"div.u32 {dst}, {lhs}, {rhs_val};", op)
    else:
        rhs = p.unique_ptx_names_for_value(op.rhs)
        _write_with_comment(p, f"div.u32 {dst}, {lhs}, {rhs};", op)


def _emit_index_mod(p: PtxPrinter, op: IndexModOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    lhs = p.unique_ptx_names_for_value(op.lhs)
    rhs_val = _const_val(op.rhs)
    if rhs_val is not None and _is_power_of_2(rhs_val):
        mask = rhs_val - 1
        _write_with_comment(p, f"and.b32 {dst}, {lhs}, {mask};", op)
    elif rhs_val is not None and 1 < rhs_val < (1 << 32) and _integer_index_division(op):
        _emit_unsigned_constant_divmod(p, op, dst, lhs, rhs_val, remainder=True)
    elif rhs_val is not None:
        _write_with_comment(p, f"rem.u32 {dst}, {lhs}, {rhs_val};", op)
    else:
        rhs = p.unique_ptx_names_for_value(op.rhs)
        _write_with_comment(p, f"rem.u32 {dst}, {lhs}, {rhs};", op)


def _integer_index_division(op: Op) -> bool:
    kernel = find_ancestor_of_type(op, KernelOp)
    return kernel is not None and kernel.attrs.get("integer_index_division", False)


def _emit_unsigned_constant_divmod(p, op, dst, lhs, divisor, *, remainder):
    """Exact unsigned division using a floor reciprocal and one correction.

    For every uint32 x, mulhi(x, floor(2**32/d)) is either floor(x/d)
    or one less: the reciprocal error times x is strictly below one.
    The tentative remainder therefore fits uint32 and is below 2*d.
    Correct it once when it reaches d. This avoids ptxas's floating
    reciprocal setup and two correction chains for div/rem.u32.
    """
    quotient = p.unique_ptx_names((op, "quotient"), PtxType.B32) if remainder else dst
    residual = dst if remainder else p.unique_ptx_names((op, "residual"), PtxType.B32)
    correction = p.unique_ptx_names((op, "correction"), PtxType.PRED)
    p.write_line(f"mul.hi.u32 {quotient}, {lhs}, {(1 << 32) // divisor};")
    p.write_line(f"mad.lo.u32 {residual}, {quotient}, {-divisor}, {lhs};")
    p.write_line(f"setp.ge.u32 {correction}, {residual}, {divisor};")
    instruction = (f"@{correction} add.u32 {dst}, {dst}, {-divisor if remainder else 1};")
    _write_with_comment(p, instruction, op)


def _emit_to_ptr(p: PtxPrinter, op: ToPtrOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    src = p.unique_ptx_names_for_value(op.arg)
    ptr_type = op.result().type
    assert isinstance(ptr_type, PtrType)
    if ptr_type.space == "gmem":
        _write_with_comment(p, f"mov.u64 {dst}, {src};", op)
    else:
        _write_with_comment(p, f"mov.u32 {dst}, {src};", op)


def _emit_ptr_add(p: PtxPrinter, op: PtrAddOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    ptr = p.unique_ptx_names_for_value(op.ptr)
    offset_val = _const_val(op.byte_offset)
    ptr_type = op.result().type
    assert isinstance(ptr_type, PtrType)
    if ptr_type.space == "gmem":
        if offset_val is not None:
            _write_with_comment(p, f"add.u64 {dst}, {ptr}, {offset_val};", op)
        else:
            offset = p.unique_ptx_names_for_value(op.byte_offset)
            tmp = p.unique_ptx_names((op, "off64"), PtxType.B64)
            p.write_line(f"cvt.u64.u32 {tmp}, {offset};")
            _write_with_comment(p, f"add.u64 {dst}, {ptr}, {tmp};", op)
    else:
        if offset_val is not None:
            _write_with_comment(p, f"add.u32 {dst}, {ptr}, {offset_val};", op)
        else:
            offset = p.unique_ptx_names_for_value(op.byte_offset)
            _write_with_comment(p, f"add.u32 {dst}, {ptr}, {offset};", op)


def _op_description(p: PtxPrinter, op: Op, max_len: int = 80) -> str:
    """One-line description of `op` for log/comment use.

    Uses `expr_str` when available (with PTX register names),
    else `op.attrs["description"]`, else `str(op)`.
    """
    expr_str = getattr(op, "expr_str", None)
    if callable(expr_str):
        return cast(
            str,
            expr_str(
                val_str=lambda v, prec=0: p.unique_ptx_names_for_value(v),
            ),
        )
    description = op.attrs.get("description")
    s = description if isinstance(description, str) else str(op)
    # Allocation IDs are process-global counters; keep them out of the PTX so
    # identical kernels share a CUDA JIT cache key.
    s = re.sub(r"(?<=attrs=\{)id: \d+(?:, )?", "", s)
    nl = s.find("\n")
    cut = min(nl, max_len) if nl != -1 else min(len(s), max_len)
    if cut < len(s):
        return s[:cut] + "..."
    return s
