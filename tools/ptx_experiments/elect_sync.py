"""Replace the `tid.x != 0` guard around TMA loads with a warp0 + elect.sync guard.

ptxas cannot prove that `@(tid.x != 0) bra skip` leaves at most one lane active,
so it wraps every UTMALDG in an ELECT/BRA.U.ANY serialization loop. `elect.sync`
states the single-lane property directly.
"""
import re, sys

def transform(ptx, verbose=False):
    lines = ptx.splitlines()
    out = []
    i = 0
    n = 0
    while i < len(lines):
        m1 = re.match(r'^(\s*)mov\.u32 (%r\d+), %tid\.x;\s*$', lines[i])
        m2 = re.match(r'^\s*setp\.ne\.s32 (%p\d+), (%r\d+), 0;\s*$', lines[i+1]) if i+1 < len(lines) else None
        m3 = re.match(r'^\s*@(%p\d+) bra (\$L\d+);\s*$', lines[i+2]) if i+2 < len(lines) else None
        if m1 and m2 and m3 and m2.group(2) == m1.group(2) and m3.group(1) == m2.group(1):
            lab = m3.group(2)
            # collect body until the label
            j = i + 3
            body = []
            while j < len(lines) and not re.match(r'^\s*' + re.escape(lab) + r':\s*$', lines[j]):
                body.append(lines[j]); j += 1
            stmts = [b.strip() for b in body if b.strip() and not b.strip().startswith('//')]
            is_tma = (stmts and all(s.startswith(('cp.async.bulk.tensor', 'mbarrier.arrive')) for s in stmts)
                      and any(s.startswith('cp.async.bulk.tensor') for s in stmts))
            if is_tma and j < len(lines):
                ind, reg, pred = m1.group(1), m1.group(2), m2.group(1)
                out.append(f"{ind}mov.u32 {reg}, %tid.x;")
                out.append(f"{ind}setp.gt.u32 {pred}, {reg}, 31;")
                out.append(f"{ind}@{pred} bra {lab};")
                out.append(f"{ind}elect.sync _|{pred}, -1;")
                out.append(f"{ind}@!{pred} bra {lab};")
                out.extend(body)
                out.append(lines[j])
                i = j + 1; n += 1
                continue
        out.append(lines[i]); i += 1
    if verbose: print(f"  rewrote {n} TMA thread-0 guards")
    return "\n".join(out) + "\n", n

if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    out, n = transform(src, verbose=True)
    open(sys.argv[2], "w").write(out)
