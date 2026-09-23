# Tiled memory-access analysis plan

This is a local design/implementation plan for PR #292. Keep it in the local
commit while doing the work, but remove it before the final upstream update.

## Goal

Represent the memory regions touched by every effect-bearing operation directly
from retained view chains, then use a conservative affine proof to decide
whether two dynamic occurrences can overlap. Moves and matmuls are the first
clients, but the representation belongs to memory-effect analysis rather than
to either operation.

The analysis must remain sound: failure to prove disjointness is a hazard. It
is acceptable to insert a stronger barrier or reject a genuinely grid-wide
hazard; it is not acceptable to suppress a required barrier.

This work recognizes true partitioning only: every pair of distinct relevant
occurrences must access disjoint physical regions. It does not use partial
overlap information to weaken a barrier from one `CollectiveScope` to another.

## Core model

Keep the existing memory-effect information and add two pieces of access
information:

1. **Existing effect semantics**: read/write kind, `CollectiveScope`, and async
   completion state. `CollectiveScope` remains the authoritative input to the
   existing barrier lattice.
2. **Region**: per-dimension affine indices and static sizes. The root buffer
   remains the dataflow map key rather than being duplicated in every region.
   Since tile views preserve their root's strides, the prover obtains the
   physical layout once from that key. Sizes are required: different indices
   alone do not prove non-overlap.
3. **Occurrences**: the bounded loop and hardware coordinates that enumerate
   distinct executions of the operation. The relevant coordinates are
   enclosing loop IVs and block, thread, warp-group, and warp IDs. Include every
   coordinate at which the operation is not collective, even when that
   coordinate is absent from the address expression.

The region and occurrence analysis sits on top of `CollectiveScope`; it does
not replace it or infer a weaker scope. For example, TMA keeps its current
block-leader collective semantics where applicable.

Adopt the operation invariant that each memory effect covers its entire operand
view for one collectively executed occurrence. The view chain therefore defines
the footprint; operation-specific effect code only selects the read/write
operands and their `CollectiveScope`.

The initial internal representation is:

```python
AffineExpr = constant + sum(coefficient * Symbol)
ParallelIV = (collective_level, axis)
IVRange = (iv, lower, upper)
AffineTileAccess = (indices, sizes, iv_ranges)
ConcurrentMemoryEffect = (kind, collective_scope, tile_access, pending_await)
```

The original operation, accessed leaf value, and root are not duplicated in
`ConcurrentMemoryEffect`: extraction consumes the operation and leaf, while
the root is already the surrounding dataflow-map key. Unsupported view chains
produce `UnknownTileAccess`.

Use semantic symbols for hardware coordinates, such as `block.x`, rather than
the identity of a particular `BlockIdsOp`. Use the owning `ForOp` (or its IV)
as the identity of a loop symbol.

Make the IR's existing root-alias contract explicit: distinct kernel buffer
arguments are assumed not to alias, and callers are responsible for satisfying
that contract. Views derived from one argument continue to share its root.
Accordingly, this work compares regions only when `root_buffers()` associates
their effects with the same root value; it does not inspect runtime pointers or
discover aliasing between distinct arguments. The retained gmem operand—not its
TMA descriptor—is the alias identity for a TMA access. Supporting aliasing
between kernel arguments is a separate problem and remains out of scope.

## 1. Preserve the tile/view chain through TMA lowering

- Redefine `TmaMoveOp` so its source and destination are the original tiled
  gmem/smem values, just like `MoveOp`, and carry the TMA descriptor as a third
  operand. The gmem operand therefore retains its complete `TileOp` chain. The
  distinct opcode remains useful as a hardware-lowering marker; only its
  memory-view contract is unified with `MoveOp`.
- Remove the eagerly materialized index-arithmetic operands from the
  analysis-visible TMA move. `TmaMoveOp` retains the gmem tile chain instead.
- Keep `LowerToTmaMove` as a small conversion that creates/reuses the descriptor
  and replaces `MoveOp(tma=True)` with the new three-operand `TmaMoveOp`, while
  preserving the gmem view operand unchanged.
- Keep descriptor creation and reuse in `LowerToTmaMove`. Move only the index
  computations currently performed by `_build_tile_chain_indices` into
  `LowerToPtxTemplate`. There, async TMA load/store lowering derives those
  indices from the retained gmem tile chain, lowers the index operations, and
  uses the already-created descriptor to emit the TMA instruction.
- Verify that a TMA move's leaf gmem view, smem operand, and descriptor tile
  shape agree in rank, element type, and logical extents. The gmem tile chain
  must narrow the root to exactly the region transferred by one TMA occurrence.
  A `maybe_partial` tile may retain its nominal extent as a conservative
  over-approximation of the in-bounds memory touched.
- Verify the descriptor/root construction invariant as well: the descriptor's
  `arg_idx` must identify the kernel argument at the root of the retained gmem
  tile chain, and its `gmem_type` (including shape, element type, strides, and
  relevant physical-layout metadata) must agree with that root. This is true by
  construction in `LowerToTmaMove`, but the verifier must reject malformed
  textual IR or later rewrites that make analysis inspect one buffer/layout
  while PTX accesses another.
- Preserve individual effects when an `AsyncOp` contains multiple moves. Do
  not collapse them into one anonymous container effect.

Acceptance check: immediately before `InsertSyncs`, a TMA access can be traced
from its gmem operand through every `TileOp` to the kernel argument/root buffer.

## 2. Make operation collectivity available to occurrence analysis

- Give `MoveOp` the same per-axis `collective` representation and textual
  syntax as `MatmulOp`; carry it unchanged into `TmaMoveOp`.
- Factor parsing, printing, accessors, and validation shared with `MatmulOp` so
  the two attributes cannot drift.
- Keep this per-axis operation attribute distinct from the existing
  `CollectiveScope` stored on each memory effect. The attribute tells us which
  hardware coordinates enumerate distinct occurrences; `CollectiveScope`
  continues to determine the barrier needed when two effects may alias.
- Make builders choose collectivity explicitly. In particular:
  - a TMA transfer is logically one occurrence per block and keeps the current
    `BLOCK_LEADER` effect semantics where applicable;
  - scalar/register moves are normally per-thread;
  - warp-group-distributed register moves use their actual warp-group level.
- Update transformations that split or tile a move to preserve or deliberately
  narrow its collectivity, as `TileMatmul` already does for matmuls.
- Use the same occurrence model for `MatmulOp`. Its lhs and rhs produce reads,
  and its destination produces both a read and a write, each with a region
  recovered from that operand's view chain and index ranges derived from the
  matmul's existing collective attribute.
- Extend this representation to elementwise operations, initialized
  allocations, and future effect-bearing operations. An operation whose access
  geometry or collectivity cannot yet be described produces an unknown region
  and is handled conservatively; it must not bypass the general model.
- Verify that all axes are present and that the collective choice is compatible
  with the operand memory spaces and lowering selected. Avoid an implicit
  default for effect-bearing production IR; a missing value should fail before
  sync analysis rather than silently claim unsafe independence.
- Verify collective uniformity transitively through index computations. For
  each hardware axis, every value that determines an operation's accessed
  region or execution must be at least as uniform as the declared collective
  level on that axis. Follow tile/view indices, explicit move indices, derived
  index arithmetic, and execution predicates; do not follow the producers of
  payload data stored in the referenced buffers.
- Treat an operation with an already-verified collective annotation as a
  dependency boundary: use its declared collective level as the uniformity of
  the relevant result rather than recursively inspecting finer-grained payload
  work behind it. Recurse only through operations that do not provide such a
  collective summary.
- For example, an operation declared `x.block` may depend on `block.x`, but
  must not depend on `wg.x`, `warp.x`, or `thread.x` through its index or
  execution computations. Do not assume an expression is uniform merely
  because finer-grained index inputs might algebraically cancel.
- Apply the same check to enclosing control-flow predicates. A block-collective
  operation cannot be conditionally executed by only a thread, warp, or warp
  group within that block.
- Decide cluster support explicitly. If moves cannot yet be cluster-collective,
  reject it; otherwise add `CLUSTER` to the shared collective-level model.

## 3. Extract regions from view chains

Add one shared operand-access extractor used by all memory-effecting operations:

- Walk each accessed value from root to leaf without destroying the chain.
- For a `TileOp`, add `index[dim] * result_size[dim]` to that dimension's
  origin. Nested tiles accumulate in root coordinates.
- Record the final view size as the accessed extent in each dimension. Compose
  each logical coordinate with the root's physical layout rather than assuming
  that logically distinct coordinates occupy distinct storage.
- For an ordinary strided tensor, model each accessed element with the byte
  address
  `base + item_size * sum(stride[d] * (origin[d] + local[d]))`, where
  `0 <= local[d] < extent[d]`. This handles noncanonical and even potentially
  overlapping strides without an implicit injectivity assumption.
- Represent swizzles, reinterpretations, and other layouts as explicit address
  mappings and compose them through the view chain. If a mapping cannot yet be
  represented or compared, return unknown rather than falling back to logical
  box separation.
- Recognize affine expressions formed from constants, addition, and
  multiplication by a constant over:
  - enclosing `ForOp` induction variables;
  - `block.{x,y,z}`;
  - `cluster_block.{x,y,z}` where supported;
  - `thread.{x,y,z}`, warp-group IDs, and lane IDs where supported.
- Normalize coefficients by semantic symbol so separately materialized ID ops
  compare equal.
- Treat non-affine multiplication, unsupported division/modulo, incompatible
  reshape/reinterpret chains, dynamic extents, and ambiguous `SelectOp` roots
  conservatively as unknown. A union of known roots may be represented as
  multiple regions rather than discarded.

Two regions on the same root are disjoint only if their physical byte-address
sets cannot intersect. Logical separation along one dimension is a useful
shortcut only after the composed physical mapping proves that the relevant
coordinate is injective and cannot alias through another dimension.

Under the whole-operand-view invariant, operation-specific code states only
which operands are read or written and their existing `CollectiveScope`. It
delegates root/view/region construction to the shared extractor. In particular,
`MatmulOp`, `MoveOp`, `TmaMoveOp`, and `ElementwiseOp` must not grow separate
tile-disjointness implementations. Any future operation that touches only a
masked or otherwise partial subset must provide an explicit footprint instead
of silently violating this invariant.

## 4. Build occurrence domains

For every effect, collect occurrence variables independently of which symbols
appear in its address:

- Every enclosing loop contributes its IV domain: `lb <= iv < ub` with the
  loop's positive `step` and congruence to `lb`. Represent it as
  `iv = lb + iteration * step` with
  `0 <= iteration < trip_count`, rather than treating the IV as every integer
  in the half-open bounds.
- A block coordinate below the operation's collective level contributes
  `0 <= block.axis < kernel.grid_dim[axis]`.
- Thread, warp-group, and warp IDs below the collective level contribute their
  corresponding semantic ID symbols. Thread IDs may use `block_dim` as bounds;
  warp and warp-group proofs should work from their ID relationships without
  depending on `num_warp_groups`.
- `grid_dim`, `block_dim`, and similar launch metadata are range information,
  not iteration symbols. Never create an occurrence dimension merely because a
  launch-count attribute is present.
- Dimensions with extent one may be omitted after normalization.
- Unknown launch bounds remain symbolic and may reduce proof power, never
  soundness.

This means a loop or hardware dimension that does not affect the tile address
still creates multiple occurrences. The proof will correctly find that those
occurrences overlap.

## 5. Prove pairwise disjointness before applying the barrier lattice

Compare only effects with a common root and at least one write.

- Preserve shared loop IVs when two effects occur in the same dynamic loop
  iteration. Independently quantify hardware occurrences and effect-local
  previous-iteration summaries when comparing arbitrary dynamic instances.
- Partition comparisons by the hardware coordinates on which the two instances
  differ. Respect operation collectivity by omitting coordinates that do not
  enumerate the operation.
- For each relevant class, ask whether there are occurrence values and bounded
  local coordinates for which the two physical byte ranges intersect.
- Prove non-overlap by showing that this intersection is impossible under the
  symbol domains and address mappings. For layouts proven injective and
  dimension-separable, use the cheaper shortcut of finding one logical
  dimension whose intervals cannot overlap. Otherwise compare the composed
  byte-address expressions directly.
- Implement a deliberately small, one-sided prover with only two outcomes:
  `PROVEN_DISJOINT` and `MAY_OVERLAP`. It supports normalized constants and
  linear combinations of recognized symbols, cancellation of identical
  symbols, independent-symbol range bounds, the loop relation
  `previous + step <= current`, and an ordered pair of distinct instances of
  one hardware-ID symbol.
- Bound case splitting by choosing the first differing occurrence coordinate
  in a fixed order and checking its two orderings. Do not enumerate Cartesian
  products of launch dimensions or recursively explore arbitrary disjunctions.
- Do not add a general integer, Presburger, or SMT solver. Non-affine
  expressions, unsupported symbol relationships, failed normalization, or a
  proof-budget limit immediately produce `MAY_OVERLAP`.
- If every relevant occurrence pair is disjoint, discard that candidate
  hazard. Otherwise, pass the two effects' unchanged `CollectiveScope` values
  to the existing `_barrier_scope` calculation. The geometric proof only
  answers aliasing; it does not redefine the synchronization lattice.
- Unknown expressions or domains mean “may overlap.”

Reducing a `GRID` hazard to `BLOCK`, `WARPGROUP`, or another narrower scope
because overlap is confined to matching hardware coordinates is explicitly out
of scope. Unless the access sets form a true partition, retain the original
hazard and its existing collective-scope calculation.

Add small proof primitives with direct tests (`prove_leq`, interval/address
overlap, bound-variable instantiation, and occurrence-order constraints) rather
than embedding the math directly in `_transfer_memory_effects`.

## 6. Model loop backedges explicitly

The generic dataflow framework currently joins a loop body's output straight
back into its header. Add an analysis hook for transforming backedge state.

For sync analysis, the hook should:

- replace the current loop IV in carried regions/occurrences with one canonical,
  stable `previous(loop)` summary symbol;
- constrain it to valid earlier iterations, including
  `previous + step <= current` for a positive-step loop;
- preserve the normal bounds and step congruence so the prover uses the actual
  minimum separation between iterations, not an assumed unit step;
- interpret `previous(loop)` as a bound variable local to each effect summary.
  Reusing the canonical symbol does not assert that two stored effects came
  from the same concrete iteration; pairwise queries rename their local binders
  apart when necessary;
- keep the same summary representation on every fixpoint iteration so the state
  remains finite, rather than producing `previous(previous(...))` symbols;
- apply this independently at each nesting level.

Two effects produced in the same loop iteration retain the exact same current
symbol `k`. Only when an effect crosses the backedge does its `k` become the
effect-local `k_prev`, denoting any valid earlier iteration. For an access with
origin `64*k` and extent `64`, the constraint `k_prev + 1 <= k` proves the
previous tile ends before the current tile begins. If an outer loop IV is not
used in the address, its two occurrences still overlap and the proof must
retain the hazard.

Add a verifier invariant that every `ForOp.step` is an integer greater than
zero. Define empty loops as `ub <= lb`, and fix `ForOp.trip_count` for the
remaining positive-step loops to use ceiling division:
`(ub - lb + step - 1) // step`. The occurrence domain and backedge prover must
use this exact trip count and step width. Generalize the verifier, trip-count
calculation, and inequalities together if negative steps are supported later;
zero must remain invalid.

## 7. Integrate with synchronization state

- Replace tuple-shaped `memory_effects` results with structured effects carrying
  regions and occurrences alongside the existing kind, `CollectiveScope`, and
  pending-await state.
- Store the structured effect directly in `ConcurrentMemoryEffect`; remove the
  special `_tma_move` recovery walk.
- Keep async completion tracking orthogonal: an await removes only effects that
  its lock and collective semantics actually complete.
- Use the general region/occurrence comparison for all structured effects,
  including matmul reads and writes. Operations not yet modeled produce unknown
  regions and retain today's conservative behavior.
- Keep `_barrier_scope`, barrier joins, and named-barrier allocation unchanged.
  The new analysis only decides whether a candidate effect pair aliases before
  those existing rules run.
- Do not confuse the dataflow join at an `IfOp` with barrier convergence. The
  join conservatively merges live effects from the taken and skipped paths, but
  a barrier inserted inside the body is reached only by participants for which
  the predicate is true. Initially reject any inferred non-thread barrier under
  control flow that is not proven uniform at the inferred barrier scope. This
  check uses the scope selected for the actual effect pair, which may be coarser
  than the current operation's declared collective level. Hoisting a barrier to
  a uniform dominator is a possible later extension, not part of the initial
  implementation. Current generated GEMM writebacks contain no `IfOp` and are
  unaffected by this restriction.
- Document that a required grid barrier is currently representable in GPU IR
  but unsupported by PTX lowering. Add an explicit verifier diagnostic before
  final lowering until a real grid synchronization mechanism exists.

## Test matrix

1. Region extraction: single and nested tiles, static and affine offsets,
   extents, canonical/noncanonical/overlapping strides, physical layouts,
   preservation of the kernel-argument no-alias contract, TMA descriptor access
   through its retained gmem root, and conservative unsupported mappings.
   Verify that TMA gmem, smem, and descriptor shapes agree, including nominal
   extents for `maybe_partial` tiles.
2. Pairwise geometry: equal boxes, adjacent boxes, partially overlapping boxes,
   constant-offset siblings, different affine coefficients, and unsupported
   expressions that conservatively return `MAY_OVERLAP`.
3. Hardware occurrences: block-, warp-group-, and thread-indexed accesses;
   unused non-collective coordinates must prevent a disjointness proof.
4. Loop history: empty loops, non-divisible trip counts, same-iteration IV
   correlation, effect-local previous binders, unit and non-unit steps, nested
   loops using every IV, nested loops omitting an outer IV, and two distinct
   operations across a backedge.
5. Collectivity: parser/printer coverage, transformations that preserve or
   narrow collectivity, verifier failures for direct/transitively derived/
   control-flow-dependent finer-grained IDs, and acceptance of fine-grained
   payload producers that do not affect indexing or execution.
6. Async groups: multiple TMA accesses to one root retain separate regions;
   await behavior remains unchanged.
7. Cross-operation hazards: move/move, move/matmul, matmul/matmul, and
   elementwise/move pairs over shared roots.
8. Barrier selection: no barrier for disjoint occurrences and correct thread,
   block-leader, block, cluster, or grid scope for possible overlap. Reject a
   non-thread barrier inside a predicate that is divergent at that barrier's
   scope; accept a predicate proven uniform at that scope.
9. End to end: existing GEMM pipelines lower to PTX without a grid barrier;
   an intentionally overlapping cross-block program fails with the explicit
   unsupported-grid-sync diagnostic.

## Implementation order

1. Enforce positive loop steps, define empty-loop behavior, and correct
   `ForOp.trip_count`; then add affine/region/occurrence data types and unit
   tests without changing the current analysis.
2. Generalize collectivity, add it to moves, and migrate all builders/parsers/
   printers. Defer transitive operand and control-flow uniformity verification
   until the region analysis is integrated.
3. Preserve tiled operands through TMA lowering and move tile-chain index
   computation and lowering to `LowerToPtxTemplate`.
4. Produce structured memory effects for moves, matmuls, elementwise
   operations, and initialized allocations; retain every effect in async groups.
5. Add pairwise overlap filtering ahead of the existing barrier-scope logic.
6. Add the loop-backedge hook and previous-iteration summaries.
7. Switch `SyncAnalysis` to the new proof and run end-to-end kernels.
8. Run the full suite, lint, type checking, and representative CUDA execution.
9. Remove this plan file before the final upstream update.

## Completion criteria

- TMA and ordinary moves expose the same tiled operands to analysis.
- Moves and matmuls have verified collectivity and conservative access
  descriptions; all other memory effects participate through the same model.
- Every current memory effect covers its complete operand view; TMA view and
  transfer shapes are verified to agree.
- Every loop has a strictly positive step, and collective operations are
  independent of finer-grained hardware IDs on each declared axis.
- `CollectiveScope` and the existing barrier lattice retain their current role;
  region analysis only removes pairs proven not to alias.
- No barrier is weakened based on partial partitioning; the optimization applies
  only when all distinct relevant occurrences are physically disjoint.
- Root identity continues to come from `root_buffers()`. Distinct kernel buffer
  arguments are explicitly assumed non-aliasing; validating or supporting
  aliasing between them is out of scope.
- Loop-carried effects distinguish current from prior iterations finitely.
- Disjoint block/loop tiles avoid false grid hazards; overlapping accesses still
  request the correct barrier scope.
- Existing generated kernels retain their synchronization behavior and pass
  end-to-end PTX/CUDA tests.
