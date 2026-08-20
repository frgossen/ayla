from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from . import pytree
from .gpu_dialect import (
    BlockIdsOp,
    GmemTensorType,
    IndexType,
    KernelOp,
    ForOp,
    LockType,
    ModuleOp,
    PTXTemplateOp,
    PtrType,
    RmemTensorType,
    SmemTensorType,
    ThreadIdsOp,
    TmaDescType,
    TokenType,
    ToPtrOp,
    LaneIdOp,
    WrmemTensorType,
)
from .base.ir import Op, Printer, Value
from .passes.folding import Folding
from .passes.cse import CSE
from .passes.dce import DCE
from .passes.insert_syncs import InsertSyncs
from .passes.infer_tma_multicast import InferTmaMulticast
from .passes.lower_to_async import LowerToAsync
from .passes.lower_to_tma_move import LowerToTmaMove
from .passes.schedule_async import ScheduleAsync
from .passes.select_stmatrix import SelectStMatrix
from .passes.lower_to_ptx_template import (
    LowerScalarToPtxTemplate,
    LowerToPtxTemplate,
)
from .base.pipeline import Pipeline


class PtxLoweringPipeline(Pipeline):
    """Lowers a high-level ModuleOp down to a PTX-template ModuleOp."""

    def __init__(
        self,
        *,
        schedule_tma_coordinates: bool = False,
    ) -> None:
        super().__init__()
        self.add_pass(InferTmaMulticast())
        self.add_pass(LowerToTmaMove())
        # The high-level scheduler has already placed waits and TMA issues at
        # shared-memory reuse boundaries. LowerToTmaMove creates only the
        # descriptor-coordinate dependency cone that still needs hoisting.
        if schedule_tma_coordinates:
            self.add_pass(ScheduleAsync(sink_awaits=False))
        self.add_pass(LowerToAsync())
        self.add_pass(Folding())
        self.add_pass(CSE())
        self.add_pass(DCE())
        self.add_pass(Folding(fold_accumulator_init=True))
        self.add_pass(InsertSyncs())
        self.add_pass(SelectStMatrix())
        self.add_pass(LowerToPtxTemplate())
        self.add_pass(Folding())
        self.add_pass(CSE())
        self.add_pass(DCE())
        # Scalar lowering runs after folding so constants embedded as PTX
        # immediates cease to be SSA operands; ordinary DCE then removes their
        # pure definitions before PTX register allocation.
        self.add_pass(LowerScalarToPtxTemplate())
        self.add_pass(CSE())
        self.add_pass(DCE())


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
        # bf16 is stored in 16-bit registers (no native `mov.bf16`).
        ptx_dtype = (
            PtxType.B16 if type.dtype.name == "bf16"
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
    if isinstance(type, WrmemTensorType):
        num_elements_per_thread = type.num_elements_per_thread
        if type.dtype.name in ("bf16", "f16"):
            n_regs = num_elements_per_thread // 2
            ptx_dtype = PtxType.B32
        elif type.dtype.name == "f32":
            n_regs = num_elements_per_thread
            ptx_dtype = PtxType.F32
        else:
            raise NotImplementedError(
                f"map_type_to_registers: wrmem dtype {type.dtype.name}"
            )
        return TypeRegisterMapping(
            member_register_types=tuple(ptx_dtype for _ in range(n_regs)),
            member_names=tuple(f"r{i}" for i in range(n_regs)),
        )
    if isinstance(type, SmemTensorType):
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
    if isinstance(type, TokenType):
        return TypeRegisterMapping(
            member_register_types=PtxType.B32,
            member_names="val",
        )
    if isinstance(type, LockType):
        if type.kind == "mbar":
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
    to load and launch it (entry point, grid/block, shmem, and — in the
    future — tensor-core / arg-layout settings).
    """

    ptx: str
    name: str
    signature: list
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    cluster_dim: tuple[int, int, int] = (1, 1, 1)
    shared_mem_bytes: int = 0


def make_ptx_kernel(
    module: ModuleOp,
    *,
    schedule_tma_coordinates: bool = False,
) -> PtxKernel:
    """Lower a single-kernel ModuleOp to a PtxKernel."""
    module = PtxLoweringPipeline(
        schedule_tma_coordinates=schedule_tma_coordinates,
    ).run(module)
    top_ops = module.body.entry_block.ops
    if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
        raise ValueError("make_ptx_kernel: module must contain exactly one KernelOp")
    kernel: KernelOp = top_ops[0]

    if all(d is None for d in kernel.grid_dim):
        raise ValueError(f"make_ptx_kernel: grid_dim is entirely unset: {kernel.grid_dim}")
    if all(d is None for d in kernel.block_dim):
        raise ValueError(f"make_ptx_kernel: block_dim is entirely unset: {kernel.block_dim}")
    grid_dim = tuple(d if d is not None else 1 for d in kernel.grid_dim)
    block_dim = tuple(d if d is not None else 1 for d in kernel.block_dim)
    if kernel.num_warp_groups is not None and kernel.num_warp_groups > 0:
        block_dim = (kernel.num_warp_groups * 128, block_dim[1], block_dim[2])
    cluster_dim = kernel.cluster_dim
    if any(grid % cluster != 0 for grid, cluster in zip(grid_dim, cluster_dim)):
        raise ValueError(
            f"grid_dim {grid_dim} must be divisible by cluster_dim {cluster_dim}"
        )
    p = PtxPrinter()
    _emit_op(p, module)
    return PtxKernel(
        ptx=p.render(),
        name=kernel.name,
        signature=kernel.signature,
        grid_dim=grid_dim,
        block_dim=block_dim,
        cluster_dim=cluster_dim,
    )


def _emit_op(p: PtxPrinter, op: Op) -> None:
    if isinstance(op, ModuleOp):
        _emit_module(p, op)
    elif isinstance(op, KernelOp):
        _emit_kernel(p, op)
    elif isinstance(op, ForOp):
        _emit_for(p, op)
    elif isinstance(op, PTXTemplateOp):
        _emit_ptx_template(p, op)
    elif isinstance(op, ThreadIdsOp):
        _emit_thread_ids(p, op)
    elif isinstance(op, BlockIdsOp):
        _emit_block_ids(p, op)
    elif isinstance(op, LaneIdOp):
        _emit_lane_id(p, op)
    elif isinstance(op, ToPtrOp):
        _emit_to_ptr(p, op)
    else:
        raise NotImplementedError(f"make_ptx_kernel: not implemented for {op.op_id}")


def _emit_region(p: PtxPrinter, region) -> None:
    for block in region.blocks:
        _emit_block(p, block)


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
    p.write_line(f"{label}:")
    p.write_newline_once()
    _emit_region(p, op.body)
    p.write_line(f"// END: `{_op_description(p, op)}`")
    p.write_line(f"add.s32 {iv_reg}, {iv_reg}, {op.step};")
    p.write_line(f"setp.lt.s32 {pred}, {iv_reg}, {op.ub};")
    p.write_line(f"@{pred} bra {label};")
    p.write_newline_once()


def _emit_module(p: PtxPrinter, op: ModuleOp) -> None:
    p.write_line(f".version {op.version}")
    p.write_line(f".target {op.target}")
    p.write_line(f".address_size {op.address_size}")
    p.write_newline_once()
    _emit_region(p, op.body)


def _emit_kernel(p: PtxPrinter, op: KernelOp) -> None:

    p.write_line(f".visible .entry {op.name}(")
    with p.indented():
        param_lines: list[str] = []
        for arg in op.body.entry_block.arguments:
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
    if op.attrs.get("emit_reqntid", False):
        block_dim = list(op.block_dim)
        if op.num_warp_groups is not None and op.num_warp_groups > 0:
            block_dim[0] = op.num_warp_groups * 128
        if any(d is None for d in block_dim):
            raise ValueError(
                "emit_reqntid requires a fully static kernel block dimension"
            )
        p.write_line(f".reqntid {block_dim[0]}, {block_dim[1]}, {block_dim[2]}")
    p.write_line("{")
    with p.indented():
        decl_mark = p.mark()
        for arg in op.body.entry_block.arguments:
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
        _emit_region(p, op.body)
        p.write_line("ret;")
    p.write_line("}")

    with p.insert_at(decl_mark):
        for ptx_type in PtxType:
            if ptx_type in (PtxType.LABEL, PtxType.SMEM):
                continue
            n = p.ptx_name_count(ptx_type)
            if n > 0:
                prefix = BASENAME_BY_PTX_TYPE[ptx_type]
                p.write_line(f".reg .{ptx_type:4s} {prefix}<{n}>;")
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

    inline = op.attrs.get("inline") is True
    if inline:
        # Scalar lowering historically named the destination before sources.
        # Keep that stable while the operation is represented as a template.
        bind_values("result", op.results, op.result_names)
        bind_values("operand", op.operands, op.operand_names)
    else:
        bind_values("operand", op.operands, op.operand_names)
        bind_values("result", op.results, op.result_names)

    def repl(m: re.Match) -> str:
        ty, name = m.group("ty"), m.group("name")
        if name not in placeholder_values:
            placeholder_values[name] = p.unique_ptx_names(
                (op, name),
                PtxType(ty),
            )
        return placeholder_values[name]

    expanded = re.sub(PLACEHOLDER_RE, repl, op.template)

    if inline:
        for line in expanded.splitlines():
            p.write_line(line.strip())
        return

    p.write_newline_once()
    p.write_line(f"// PTX template `{_op_description(p, op)}`")
    for line in expanded.splitlines():
        p.write_line(line.strip())
    p.write_newline_once()


def _emit_thread_ids(p: PtxPrinter, op: ThreadIdsOp) -> None:
    rx = p.unique_ptx_names_for_value(op.x)
    ry = p.unique_ptx_names_for_value(op.y)
    rz = p.unique_ptx_names_for_value(op.z)
    p.write_newline_once()
    p.write_line(f"mov.u32 {rx}, %tid.x;")
    p.write_line(f"mov.u32 {ry}, %tid.y;")
    p.write_line(f"mov.u32 {rz}, %tid.z;")
    p.write_newline_once()


def _emit_block_ids(p: PtxPrinter, op: BlockIdsOp) -> None:
    rx = p.unique_ptx_names_for_value(op.x)
    ry = p.unique_ptx_names_for_value(op.y)
    rz = p.unique_ptx_names_for_value(op.z)
    p.write_newline_once()
    p.write_line(f"mov.u32 {rx}, %ctaid.x;")
    p.write_line(f"mov.u32 {ry}, %ctaid.y;")
    p.write_line(f"mov.u32 {rz}, %ctaid.z;")
    p.write_newline_once()


def _write_with_comment(p: PtxPrinter, instr: str, op: Op, col: int = 30) -> None:
    """Write an instruction with an aligned trailing comment."""
    comment = _op_description(p, op)
    padding = max(1, col - len(instr))
    p.write_line(f"{instr}{' ' * padding}// {comment}")


def _emit_lane_id(p: PtxPrinter, op: LaneIdOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    _write_with_comment(p, f"mov.u32 {dst}, %laneid;", op)


def _emit_to_ptr(p: PtxPrinter, op: ToPtrOp) -> None:
    dst = p.unique_ptx_names_for_value(op.result())
    src = p.unique_ptx_names_for_value(op.arg)
    ptr_type = op.result().type
    if ptr_type.space == "gmem":
        _write_with_comment(p, f"mov.u64 {dst}, {src};", op)
    else:
        _write_with_comment(p, f"mov.u32 {dst}, {src};", op)


def _op_description(p: PtxPrinter, op: Op, max_len: int = 80) -> str:
    """One-line description of `op` for log/comment use.

    Uses `expr_str` when available (with PTX register names),
    else `op.attrs["description"]`, else `str(op)`.
    """
    if hasattr(op, "expr_str"):
        return op.expr_str(val_str=lambda v, prec=0: p.unique_ptx_names_for_value(v))
    s = op.attrs.get("description") if "description" in op.attrs else str(op)
    nl = s.find("\n")
    cut = min(nl, max_len) if nl != -1 else min(len(s), max_len)
    if cut < len(s):
        return s[:cut] + "..."
    return s
