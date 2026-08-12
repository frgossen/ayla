"""Taint-analyze: does any operand of a cp.async.bulk.tensor / mbarrier.arrive
inside a thread-0 guard depend on %tid / %laneid ?  If so, electing a lane other
than 0 would change behaviour."""
import re, sys, collections
REG = re.compile(r'%\w+')
lines = open(sys.argv[1]).read().splitlines()
defs = collections.defaultdict(list)
for i, l in enumerate(lines):
    s = l.strip()
    if not s or s.startswith(('//','$L','.')): continue
    m = re.match(r'(?:@!?%p\d+\s+)?[a-z][\w.]*\s+(%\w+)\s*,(.*)', s, re.S)
    if m: defs[m.group(1)].append((i, m.group(2)))
tainted = set()
# seed: anything assigned from %tid / %laneid
for i, l in enumerate(lines):
    if re.search(r'%tid\.|%laneid', l):
        m = re.match(r'\s*(?:@!?%p\d+\s+)?[a-z][\w.]*\s+(%\w+)\s*,', l)
        if m: tainted.add(m.group(1))
changed = True
while changed:
    changed = False
    for r, ds in defs.items():
        if r in tainted: continue
        for _, rhs in ds:
            if any(s in tainted for s in REG.findall(rhs)):
                tainted.add(r); changed = True; break
print(f"tainted-by-tid registers: {len(tainted)}")
bad = 0
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith(('cp.async.bulk.tensor.2d.shared','cp.async.bulk.tensor.3d.shared','mbarrier.arrive.expect_tx')):
        used = REG.findall(s)
        t = [r for r in used if r in tainted]
        if t:
            bad += 1
            print(f"  line {i+1}: TID-DEPENDENT operands {t}\n      {s[:110]}")
print("RESULT:", "SAFE - no TMA-load operand depends on tid" if not bad else f"UNSAFE ({bad} sites)")
