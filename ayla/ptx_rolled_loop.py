"""Register recurrences for the bounded rolled operand loop at PTX emission.

Keep the analysis IR in SSA form. Only the final emitter carries registers
across the backedge, after cloning, folding, and synchronization are finished.
"""

from .base.ir import find_ancestor_of_type, iter_ops
from .analysis.operand_ring import _affine, _constant
from .gpu_dialect import (
    ForOp, IndexDivOp, IndexModOp,
    PTXTemplateOp, MmaDescriptorOp, MmaDescriptorLowOp, PtrAddOp, PtrType, ToPtrOp,
)


class RolledLoopRecurrences:
    def __init__(self, printer, loop, register_type, emit_op):
        self.p, self.loop, self.ty, self.emit_op = printer, loop, register_type, emit_op
        self.groups = {}
        self.nodes = {}
        self.descriptors = []
        self.full_descriptors = []
        self.invariants = []
        self.skipped = set()
        if loop.step not in (1, 2):
            return
        leaves = [op for op in iter_ops(loop)
                  if find_ancestor_of_type(op, ForOp) is loop]
        if not any(isinstance(op, MmaDescriptorOp) for op in leaves):
            return
        # SSA constants and CTA identity stay outside the steady loop. Their
        # templates read only constants or CTA-uniform special registers.
        constants = {}
        for op in leaves:
            if isinstance(op, PTXTemplateOp) and op.attrs.get('pure') and not op.operands:
                key = op.template, tuple(map(repr, (result.type for result in op.results)))
                prior = constants.get(key)
                if prior is None:
                    constants[key] = op
                    self.invariants.append(op)
                else:
                    for result, original in zip(op.results, prior.results):
                        printer.carried_registers[result] = printer.unique_ptx_names_for_value(original)
                self.skipped.add(op)
        for op in leaves:
            if not isinstance(op, (IndexModOp, IndexDivOp)):
                continue
            divisor = _constant(op.rhs)
            affine = _affine(op.lhs, loop.iv)
            if (divisor is None or divisor < max(2, loop.step) or affine is None or affine[0] != 1
                    or loop.lb + affine[1] < 0 or loop.ub + affine[1] >= 2**32):
                continue
            offset = affine[1]
            key = divisor, offset % divisor
            group = self.groups.setdefault(key, {
                'bank': printer.unique_ptx_names((loop, key, 'bank'), register_type.B32),
                'wrap': printer.unique_ptx_names((loop, key, 'wrap'), register_type.PRED),
                'quotients': {}, 'phases': {},
            })
            if isinstance(op, IndexModOp):
                reg = group['bank']
            else:
                reg = group['quotients'].setdefault(offset,
                    printer.unique_ptx_names((loop, key, offset, 'epoch'), register_type.B32))
            self.nodes[op.result()] = key, offset
            printer.carried_registers[op.result()] = reg
            self.skipped.add(op)
        for op in leaves:
            if (isinstance(op, IndexModOp) and _constant(op.rhs) == 2
                    and isinstance(op.lhs.definition, IndexDivOp) and op.lhs in self.nodes):
                key, offset = self.nodes[op.lhs]
                reg = self.groups[key]['phases'].setdefault(offset,
                    printer.unique_ptx_names((loop, key, offset, 'phase'), register_type.B32))
                printer.carried_registers[op.result()] = reg
                self.skipped.add(op)
        # Descriptor address bits move by the stage stride; all other fields
        # stay constant. The SMEM arena fits in the descriptor's 14-bit field.
        for op in leaves:
            if not (isinstance(op, MmaDescriptorLowOp) and op.attrs.get('pure')
                    and len(op.operands) == 1):
                continue
            for value, (key, _) in self.nodes.items():
                if not isinstance(value.definition, IndexModOp):
                    continue
                stride = self._pointer_stride(op.operands[0], value)
                if stride is not None and stride > 0 and stride % 16 == 0:
                    self.descriptors.append((op, key, stride // 16))
                    self.skipped.add(op)
                    break
        for op in leaves:
            if isinstance(op, MmaDescriptorOp) and op.attrs.get('pure'):
                low = op.operands[0].definition
                match = next(((key, stride) for desc, key, stride in self.descriptors if desc is low), None)
                if match is not None:
                    self.full_descriptors.append((op, *match))
                    self.skipped.add(op)

    def _pointer_stride(self, value, bank):
        op = value.definition
        if not isinstance(value.type, PtrType) or value.type.space != 'smem':
            return None
        if find_ancestor_of_type(op, ForOp) is not self.loop:
            return 0
        if (isinstance(op, ToPtrOp)
                and find_ancestor_of_type(op.arg.definition, ForOp) is not self.loop):
            return 0
        if not isinstance(op, PtrAddOp):
            return None
        base = self._pointer_stride(op.ptr, bank)
        offset = _affine(op.byte_offset, bank)
        return None if base is None or offset is None else base + offset[0]

    def initialize(self):
        for op in self.invariants:
            self.emit_op(self.p, op)
        for (divisor, offset), group in self.groups.items():
            self.p.write_line(f"mov.u32 {group['bank']}, {(self.loop.lb + offset) % divisor};")
            for shift, reg in group['quotients'].items():
                self.p.write_line(f"mov.u32 {reg}, {(self.loop.lb + shift) // divisor};")
            for shift, reg in group['phases'].items():
                self.p.write_line(f"mov.u32 {reg}, {((self.loop.lb + shift) // divisor) % 2};")
        emitted = set()

        def dependency(value):
            op = value.definition
            if (not hasattr(op, 'operands') or op in emitted or op in self.skipped
                    or find_ancestor_of_type(op, ForOp) is not self.loop):
                return
            for operand in op.operands:
                dependency(operand)
            self.emit_op(self.p, op)
            emitted.add(op)

        for op, _, _ in self.descriptors:
            dependency(op.operands[0])
            self.emit_op(self.p, op)
        for op, _, _ in self.full_descriptors:
            self.emit_op(self.p, op)
        # Remove setup calculations with no consumer left in the backedge.
        changed = True
        while changed:
            changed = False
            for op in emitted - self.skipped:
                if all(user in self.skipped for result in op.results for user in result.users()):
                    self.skipped.add(op)
                    changed = True

    def advance(self):
        for key, group in self.groups.items():
            divisor, _ = key
            bank, wrap = group['bank'], group['wrap']
            self.p.write_line(f"add.u32 {bank}, {bank}, {self.loop.step};")
            self.p.write_line(f"setp.ge.u32 {wrap}, {bank}, {divisor};")
            self.p.write_line(f"@{wrap} sub.u32 {bank}, {bank}, {divisor};")
            for reg in group['quotients'].values():
                self.p.write_line(f"@{wrap} add.u32 {reg}, {reg}, 1;")
            for reg in group['phases'].values():
                self.p.write_line(f"@{wrap} xor.b32 {reg}, {reg}, 1;")
            for op, descriptor_key, stride in self.descriptors:
                if descriptor_key != key:
                    continue
                if all(user in self.skipped for user in op.result().users()):
                    continue
                reg = self.p.unique_ptx_names_for_value(op.result())
                self.p.write_line(f"@!{wrap} add.u32 {reg}, {reg}, {self.loop.step * stride};")
                self.p.write_line(f"@{wrap} sub.u32 {reg}, {reg}, {(divisor - self.loop.step) * stride};")
            deltas = {}
            for op, descriptor_key, stride in self.full_descriptors:
                if descriptor_key != key:
                    continue
                if stride not in deltas:
                    delta = self.p.unique_ptx_names((self.loop, key, stride, 'descriptor_delta'), self.ty.B32)
                    deltas[stride] = delta
                    self.p.write_line(f'selp.s32 {delta}, {(self.loop.step - divisor) * stride}, '
                                      f'{self.loop.step * stride}, {wrap};')
                reg = self.p.unique_ptx_names_for_value(op.result())
                low = self.p.unique_ptx_names((op, 'low'), self.ty.B32)
                high = self.p.unique_ptx_names((op, 'high'), self.ty.B32)
                self.p.write_line(f'mov.b64 {{{low}, {high}}}, {reg};')
                self.p.write_line(f'add.u32 {low}, {low}, {deltas[stride]};')
                self.p.write_line(f'mov.b64 {reg}, {{{low}, {high}}};')
