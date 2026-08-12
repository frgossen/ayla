import re, sys, collections
path = sys.argv[1]
lines = open(path).read().splitlines()
INS = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s+(.*?);\s*$')
insns = []   # (addr, predicate, opcode, full)
for ln in lines:
    m = INS.match(ln)
    if not m: continue
    addr = int(m.group(1), 16)
    body = m.group(2).strip()
    pred = ''
    if body.startswith('@'):
        p, _, rest = body.partition(' ')
        pred = p; body = rest.strip()
    op = body.split()[0] if body.split() else ''
    insns.append((addr, pred, op, body))
print(f"{len(insns)} instructions")
base = collections.Counter(op.split('.')[0] for _,_,op,_ in insns)
print("\n=== top opcodes (base) ===")
for k,v in base.most_common(28): print(f"{v:5d}  {k}")
# find branch targets that jump backwards = loops
labels = {}
for i,ln in enumerate(lines):
    m = re.match(r'^(\.L_x_\d+):', ln.strip())
    if m: labels[m.group(1)] = i
print("\n=== backward branches (loops) ===")
lab_addr = {}
cur = None
for ln in lines:
    m = re.match(r'^(\.L_x_\d+):', ln.strip())
    if m: cur = m.group(1); continue
    mi = INS.match(ln)
    if mi and cur is not None:
        lab_addr[cur] = int(mi.group(1),16); cur=None
for addr,pred,op,body in insns:
    if op.startswith('BRA') or op.startswith('BRX'):
        m = re.search(r'`\((\.L_x_\d+)\)', body)
        if m and m.group(1) in lab_addr and lab_addr[m.group(1)] < addr:
            print(f"  0x{addr:06x} {pred} {body}   -> target 0x{lab_addr[m.group(1)]:06x} (span {(addr-lab_addr[m.group(1)])//16} insns)")
