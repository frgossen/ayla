"""Hoist loop-invariant WGMMA descriptor construction out of the K loop (PTX text level).

The lowering emits, for every wgmma group:
    wgmma.fence.sync.aligned;
    [setp.ne.u32 ...]            <- scale_d predicate (NOT hoisted)
    shr/cvt/or/add ...           <- descriptor construction
    wgmma.mma_async ...
Descriptor construction depends only on the multi-buffer slot's SMEM address,
which is loop-invariant, so the whole block can move to the loop preheader.
"""
import re, sys, collections

REG = re.compile(r'%\w+')
DESC_OPS = ('shr.b32', 'cvt.u64.u32', 'or.b64', 'add.u64', 'add.u32')

def transform(ptx, verbose=False):
    lines = ptx.splitlines()
    lab = {}
    for i, l in enumerate(lines):
        m = re.match(r'\s*(\$L\d+):\s*$', l)
        if m: lab[m.group(1)] = i
    loops = []
    for i, l in enumerate(lines):
        m = re.search(r'bra\s+(\$L\d+);', l)
        if m and m.group(1) in lab and lab[m.group(1)] < i:
            loops.append((lab[m.group(1)], i, m.group(1)))
    loops = [x for x in loops if x[1] - x[0] > 100]

    def parse(l):
        s = l.strip()
        if not s or s.startswith('//') or s.startswith('$L') or s.startswith('.'):
            return None
        op = s.split()[0]
        rest = s[len(op):]
        dst = None; srcs = []
        m = re.match(r'\s*(%\w+)\s*,(.*)', rest, re.S)
        if m:
            dst = m.group(1); srcs = REG.findall(m.group(2))
        return op, dst, srcs

    edits = set(); inserts = collections.defaultdict(list); total = 0
    for h, b, tgt in loops:
        if any(re.search(r'bra\s+' + re.escape(tgt) + r';', lines[i]) for i in range(h)):
            if verbose: print(f"  loop@{h}: skipped (forward entry)")
            continue
        body = list(range(h + 1, b + 1))
        bodyset = set(body)
        # defs within this loop body
        defs_in = collections.defaultdict(list)
        for i in body:
            p = parse(lines[i])
            if p and p[1]: defs_in[p[1]].append(i)
        # locate fence -> first mma_async regions
        regions = []
        i = h + 1
        while i <= b:
            if lines[i].strip().startswith('wgmma.fence'):
                j = i + 1
                while j <= b and not lines[j].strip().startswith('wgmma.mma_async'):
                    j += 1
                if j <= b: regions.append((i + 1, j - 1))
                i = j
            i += 1
        cand = []
        for lo, hi in regions:
            blk = [k for k in range(lo, hi + 1) if parse(lines[k])]
            ops = [parse(lines[k])[0] for k in blk]
            if any(o not in DESC_OPS for o in ops):
                if verbose:
                    bad = set(ops) - set(DESC_OPS)
                    print(f"    region {lo}-{hi}: not pure desc math ({bad}) - partial")
            keep = [k for k in blk if parse(lines[k])[0] in DESC_OPS]
            cand.extend(keep)
        # verify invariance: every source either defined outside the loop, or
        # defined only by candidate instructions
        candset = set(cand)
        ok = True
        for k in cand:
            op, dst, srcs = parse(lines[k])
            for s in srcs:
                for d in defs_in.get(s, []):
                    if d not in candset:
                        ok = False
                        if verbose: print(f"    !! {lines[k].strip()} src {s} defined in-loop at {d}")
        # verify each candidate dest is not read by any non-candidate def-chain issue:
        # dest regs must only be defined by candidates
        for k in cand:
            op, dst, srcs = parse(lines[k])
            for d in defs_in.get(dst, []):
                if d not in candset:
                    ok = False
                    if verbose: print(f"    !! dest {dst} also defined at {d}")
        if not ok:
            print(f"  loop@{h}: INVARIANCE CHECK FAILED, skipping"); continue
        total += len(cand)
        edits |= candset
        inserts[h].extend(lines[k] for k in sorted(cand))
        if verbose:
            print(f"  loop@{h}-{b}: hoisted {len(cand)} of {b-h} body lines "
                  f"({len(regions)} wgmma groups)")
    out = []
    for i, l in enumerate(lines):
        if i in inserts:
            out.append("  // --- hoisted loop-invariant wgmma descriptors ---")
            out.extend(inserts[i])
        if i in edits: continue
        out.append(l)
    return "\n".join(out) + "\n", total

if __name__ == "__main__":
    out, n = transform(open(sys.argv[1]).read(), verbose=True)
    open(sys.argv[2], "w").write(out)
    print(f"  => hoisted {n} instructions -> {sys.argv[2]}")
