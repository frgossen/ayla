r"""Port instruction-level idioms found in the CuTe/Triton PTX into Ayla's PTX.

Derived from an opcode inventory diff of
`generated_ptx/compiler_comparison/{ayla,cute,triton}.ptx`.  Only the idioms
*not* already covered by that directory's README transfer experiments are here.

    prefetch    CuTe issues `prefetch.tensormap` for all three tensor maps from
                warp 0 at kernel entry; Ayla never warms the tensormap cache, so
                its first TMA of each descriptor eats a cold descriptor fetch.
    storehint   CuTe tags *every* TMA (14 loads and 8 stores) with
                `.L2::cache_hint`; Ayla hints only the 28 LHS loads and leaves
                the 8 output stores unhinted.  C is write-only and never re-read,
                so an evict-first store keeps A/B resident instead.
    rhshint     Same policy on the 28 unhinted RHS loads.
    ctascope    Triton emits `cp.async.bulk.tensor...shared::cta.global`; Ayla
                emits `shared::cluster.global` even though it launches no
                clusters.
    plainmbar   CuTe/Triton use bare `mbarrier.try_wait.parity` /
                `mbarrier.arrive.expect_tx`; Ayla spells out `.acquire.cta` /
                `.release.cta`.  Those are the ISA defaults, so this should be a
                no-op -- it is here to confirm that rather than assume it.
"""
from __future__ import annotations

import re
import sys

CVTA_PARAM = re.compile(r"^(\s*)cvta\.param\.u64 (%rd\d+), %rd\d+;\s*$")
POLICY = re.compile(r"^\s*createpolicy\.fractional\.L2::evict_first\.b64 (%rd\d+), [\d.]+;")
TMA_STORE = re.compile(
    r"^(\s*)cp\.async\.bulk\.tensor\.(\d)d\.global\.shared::cta\.bulk_group "
    r"\[(%rd\d+), \{([^}]*)\}\], \[(%r\d+)\];\s*$")
TMA_LOAD_PLAIN = re.compile(
    r"^(\s*)cp\.async\.bulk\.tensor\.(\d)d\.shared::cluster\.global"
    r"\.mbarrier::complete_tx::bytes (\[.*\]);\s*$")
TID_READ = re.compile(r"^(\s*)mov\.u32 (%r\d+), %tid\.x;\s*$")


def prefetch(ptx: str, verbose: bool = False) -> tuple[str, int]:
    """Warm the tensormap cache for all descriptors from warp 0, as CuTe does."""
    lines = ptx.splitlines()
    descs, tid_reg, tid_line = [], None, None
    for i, line in enumerate(lines):
        m = CVTA_PARAM.match(line)
        if m:
            descs.append((i, m.group(2), m.group(1)))
        m = TID_READ.match(line)
        if m and tid_reg is None:
            tid_reg, tid_line = m.group(2), i
    if not descs or tid_reg is None:
        return ptx, 0
    last, indent = descs[-1][0], descs[-1][2]
    assert tid_line > last, "expected %tid.x read after the descriptor cvtas"
    block = [
        f"{indent}// tensormap prefetch (CuTe idiom): warp 0 only",
        f"{indent}mov.u32 %rPF, %tid.x;",
        f"{indent}setp.gt.u32 %pPF, %rPF, 31;",
        f"{indent}@%pPF bra $LPF;",
    ]
    block += [f"{indent}prefetch.tensormap [{r}];" for _, r, _ in descs]
    block += [f"{indent}$LPF:"]
    lines[last + 1:last + 1] = block + [""]
    out = "\n".join(lines) + "\n"
    # Allocate the two scratch names inside the existing declarations.
    out = out.replace(".reg .b32  %r<", ".reg .b32  %rPF;\n  .reg .b32  %r<", 1)
    out = out.replace(".reg .pred %p<", ".reg .pred %pPF;\n  .reg .pred %p<", 1)
    if verbose:
        print(f"  prefetch: {len(descs)} tensormaps")
    return out, len(descs)


def _policy_reg(lines: list[str]) -> str:
    for line in lines:
        m = POLICY.match(line)
        if m:
            return m.group(1)
        raise_if = None
    raise AssertionError("no createpolicy in this PTX")


def storehint(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    pol = _policy_reg(lines)
    n = 0
    for i, line in enumerate(lines):
        m = TMA_STORE.match(line)
        if not m:
            continue
        ind, rank, desc, idx, src = m.groups()
        lines[i] = (f"{ind}cp.async.bulk.tensor.{rank}d.global.shared::cta"
                    f".bulk_group.L2::cache_hint [{desc}, {{{idx}}}], [{src}], {pol};")
        n += 1
    if verbose:
        print(f"  storehint: {n} TMA stores")
    return "\n".join(lines) + "\n", n


def storefraction(
    ptx: str, fraction: float, verbose: bool = False,
) -> tuple[str, int]:
    """Give output stores a separate fractional evict-first policy."""
    ptx, n = storehint(ptx)
    if not n:
        return ptx, 0
    lines = ptx.splitlines()
    load_policy = _policy_reg(lines)
    declaration = next(
        i for i, line in enumerate(lines) if ".reg .b64" in line
    )
    lines.insert(declaration, "  .reg .b64  %rd_store_policy;")
    policy = next(i for i, line in enumerate(lines) if POLICY.match(line))
    lines.insert(
        policy + 1,
        ("  createpolicy.fractional.L2::evict_first.b64 "
         f"%rd_store_policy, {fraction};"),
    )
    changed = 0
    for i, line in enumerate(lines):
        if (".global.shared::cta.bulk_group.L2::cache_hint" not in line
                or not line.rstrip().endswith(f"{load_policy};")):
            continue
        lines[i] = line.rsplit(load_policy, 1)[0] + "%rd_store_policy;"
        changed += 1
    assert changed == n
    if verbose:
        print(f"  storefraction: {n} stores at fraction {fraction}")
    return "\n".join(lines) + "\n", n


def rhshint(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    pol = _policy_reg(lines)
    n = 0
    for i, line in enumerate(lines):
        m = TMA_LOAD_PLAIN.match(line)
        if not m:
            continue
        ind, rank, rest = m.groups()
        lines[i] = (f"{ind}cp.async.bulk.tensor.{rank}d.shared::cluster.global"
                    f".mbarrier::complete_tx::bytes.L2::cache_hint {rest}, {pol};")
        n += 1
    if verbose:
        print(f"  rhshint: {n} unhinted TMA loads")
    return "\n".join(lines) + "\n", n


def ctascope(ptx: str, verbose: bool = False) -> tuple[str, int]:
    n = ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global")
    ptx = ptx.replace("cp.async.bulk.tensor.2d.shared::cluster.global",
                      "cp.async.bulk.tensor.2d.shared::cta.global")
    if verbose:
        print(f"  ctascope: {n} TMA loads retargeted to shared::cta")
    return ptx, n


def plainmbar(ptx: str, verbose: bool = False) -> tuple[str, int]:
    n = (ptx.count("mbarrier.try_wait.parity.acquire.cta.shared::cta.b64")
         + ptx.count("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"))
    ptx = ptx.replace("mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                      "mbarrier.try_wait.parity.shared::cta.b64")
    ptx = ptx.replace("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                      "mbarrier.arrive.expect_tx.shared::cta.b64")
    if verbose:
        print(f"  plainmbar: {n} mbarrier ops stripped of explicit sem/scope")
    return ptx, n


def swaploads(ptx: str, verbose: bool = False) -> tuple[str, int]:
    """Issue the larger RHS TMA before the LHS TMA in every operand stage."""
    lines = ptx.splitlines()
    n = 0
    prefix = "cp.async.bulk.tensor.2d.shared::"
    for i in range(len(lines) - 1):
        if (lines[i].lstrip().startswith(prefix)
                and lines[i + 1].lstrip().startswith(prefix)):
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
            n += 1
    if verbose:
        print(f"  swaploads: {n} operand-load pairs")
    return "\n".join(lines) + "\n", n


TRANSFORMS = {
    "prefetch": prefetch,
    "storehint": storehint,
    "rhshint": rhshint,
    "ctascope": ctascope,
    "plainmbar": plainmbar,
    "swaploads": swaploads,
    "storefrac25": lambda p, verbose=False: storefraction(p, 0.25, verbose),
    "storefrac50": lambda p, verbose=False: storefraction(p, 0.5, verbose),
    "storefrac75": lambda p, verbose=False: storefraction(p, 0.75, verbose),
}


def transform(ptx: str, which: str, verbose: bool = False) -> tuple[str, int]:
    total = 0
    for name in which.split("+"):
        ptx, n = TRANSFORMS[name](ptx, verbose=verbose)
        total += n
    return ptx, total


if __name__ == "__main__":
    out, n = transform(open(sys.argv[1]).read(), sys.argv[3], verbose=True)
    open(sys.argv[2], "w").write(out)
