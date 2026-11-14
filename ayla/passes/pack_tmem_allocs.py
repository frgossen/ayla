"""Pack equal-shaped TMEM buffers into static column views of one arena."""
from ..base.ir import Block, Op, find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import AllocOp, BarrierOp, DeallocOp, KernelOp, ModuleOp, TileOp, TmemTensorType


def _blocks(op: Op) -> list[Block]:
    result = []
    while op.parent_block is not None:
        result.append(op.parent_block)
        op = op.parent_block.parent_op
        if op is None:
            break
    return result


class PackTmemAllocs(Pass):
    """Give equal-shaped, nonescaping allocations independent column views.

    The arena spans their common scope and fits the hardware's 512 columns.
    Former release boundaries retain CTA synchronization; the arena is freed
    once at the end of the common scope. LICM can then reuse this sole lifetime.
    """

    def __init__(self, *, track_column_regions=False, preserve_release_barriers=True):
        super().__init__(track_column_regions=track_column_regions,
                         preserve_release_barriers=preserve_release_barriers)

    def run(self, module: ModuleOp) -> ModuleOp:
        kernels = find_ops_of_type(module, KernelOp)
        for kernel in kernels:
            allocs = [a for a in find_ops_of_type(kernel, AllocOp)
                      if isinstance(a.result().type, TmemTensorType)]
            if len(allocs) < 2:
                continue
            typ = allocs[0].result().type
            if any(a.result().type != typ for a in allocs):
                raise ValueError('TMEM packing requires equal-shaped allocations')
            rows, width = typ.sizes
            total = width * len(allocs)
            if total > 512:
                raise ValueError('packed TMEM arena exceeds 512 columns')
            chains = [_blocks(a) for a in allocs]
            common = next(b for b in chains[0] if all(b in chain for chain in chains))
            releases = []
            for alloc in allocs:
                users = alloc.result().users()
                frees = [op for op in users if isinstance(op, DeallocOp)]
                if len(frees) != 1 or any(common not in _blocks(op) for op in users):
                    raise ValueError('TMEM packing requires a local, nonescaping lifetime')
                releases.extend(frees)
            attrs = {key: value for key, value in typ.attrs.items()
                     if key not in ("sizes", "dtype", "columns")}
            arena_type = TmemTensorType.build([rows, total], typ.dtype, attrs=attrs)
            alloc_attrs = {'label': 'tmem_arena'}
            if self.track_column_regions:
                alloc_attrs['track_column_regions'] = True
            arena = AllocOp.build_with_fresh_id(arena_type, attrs=alloc_attrs)
            common.ops.insert(0, arena)
            for index, alloc in enumerate(allocs):
                view = TileOp.build(arena.result(), [0, index], typ, attrs=alloc.attrs)
                # Replacing operands also updates their maintained use lists.
                for user, operand in list(alloc.result().uses):
                    user.operands[operand] = view.result()
                block = alloc.parent_block
                block.ops[block.ops.index(alloc)] = view
            for release in releases:
                block = release.parent_block
                index = block.ops.index(release)
                if self.preserve_release_barriers:
                    block.ops[index] = BarrierOp.build(attrs={'resource_lifetime_reuse': True})
                else:
                    del block.ops[index]
            common.ops.append(DeallocOp.build(arena.result()))
        return module
