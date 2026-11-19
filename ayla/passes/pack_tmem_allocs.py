"""Pack native TMEM accumulators into static column views of one arena."""
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
    ``compose_rows`` instead packs one M128 and one M64 tile into explicit
    native views and retires the arena at the last native fragment read.
    """

    def __init__(self, *, track_column_regions=False, preserve_release_barriers=True,
                 compose_rows=False):
        super().__init__(track_column_regions=track_column_regions,
                         preserve_release_barriers=preserve_release_barriers,
                         compose_rows=compose_rows)

    def run(self, module: ModuleOp) -> ModuleOp:
        kernels = find_ops_of_type(module, KernelOp)
        for kernel in kernels:
            allocs = [a for a in find_ops_of_type(kernel, AllocOp)
                      if isinstance(a.result().type, TmemTensorType)]
            if len(allocs) < 2:
                continue
            typ = allocs[0].result().type
            if self.compose_rows:
                if (len(allocs) != 2
                        or sorted(a.result().type.sizes[0] for a in allocs) != [64,128]
                        or any(a.result().type.sizes[1] != typ.sizes[1]
                               or a.result().type.dtype != typ.dtype
                               or a.result().type.cta_group != typ.cta_group for a in allocs)):
                    raise ValueError('M192 packing requires one M128 and one M64 accumulator of equal width')
            elif any(a.result().type != typ for a in allocs):
                raise ValueError('TMEM packing requires equal-shaped allocations')
            rows, width = typ.sizes
            if self.compose_rows:
                rows = 128
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
                view_attrs = dict(alloc.attrs)
                if self.compose_rows:
                    view_attrs['native_tmem_view'] = True
                view = TileOp.build(arena.result(), [0, index], alloc.result().type, attrs=view_attrs)
                # Replacing operands also updates their maintained use lists.
                for user, operand in list(alloc.result().uses):
                    user.operands[operand] = view.result()
                block = alloc.parent_block
                block.ops[block.ops.index(alloc)] = view
            # A composed tile ends at the last native fragment read, rather
            # than holding its arena through the remaining conversion/stores.
            last_release = (max(releases,key=lambda op: common.ops.index(op))
                            if self.compose_rows and all(op.parent_block is common for op in releases)
                            else None)
            for release in releases:
                block = release.parent_block
                index = block.ops.index(release)
                if release is last_release:
                    block.ops[index] = DeallocOp.build(arena.result())
                elif self.preserve_release_barriers:
                    block.ops[index] = BarrierOp.build(attrs={'resource_lifetime_reuse': True})
                else:
                    del block.ops[index]
            if last_release is None:
                common.ops.append(DeallocOp.build(arena.result()))
        return module
