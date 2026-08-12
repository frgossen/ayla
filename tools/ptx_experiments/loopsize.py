import re,sys,collections
INS = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s+(.*?);\s*$')
def loops(path):
    lines=open(path).read().splitlines()
    lab={}; cur=None
    for ln in lines:
        m=re.match(r'^(\.L_x_\d+):',ln.strip())
        if m: cur=m.group(1); continue
        mi=INS.match(ln)
        if mi and cur: lab[cur]=int(mi.group(1),16); cur=None
    ins=[]
    for ln in lines:
        mi=INS.match(ln)
        if mi: ins.append((int(mi.group(1),16), mi.group(2)))
    out=[]
    for a,b in ins:
        m=re.search(r'`\((\.L_x_\d+)\)',b)
        if m and b.split()[0].startswith(('BRA','@')) and 'BRA' in b:
            t=lab.get(m.group(1))
            if t is not None and t<a and (a-t)//16>50:
                out.append((t,a))
    return ins,out
for tag in sys.argv[1:]:
    ins,ls = loops(tag)
    print(f"\n### {tag}")
    for t,a in ls:
        n=(a-t)//16+1
        body=[x for x in ins if t<=x[0]<=a]
        c=collections.Counter(x[1].split()[0].lstrip('@!').split('.')[0] if not x[1].startswith('@') else x[1].split()[1].split('.')[0] for x in body)
        hg=sum(v for k,v in c.items() if k=='HGMMA')
        print(f"  loop 0x{t:05x}-0x{a:05x}: {n} insns, {hg} HGMMA, {n/hg:.1f} insn/HGMMA" if hg else f"  loop: {n} insns")
        print("     ", dict(c.most_common(10)))
