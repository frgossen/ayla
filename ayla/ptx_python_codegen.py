"""Lift generated PTX into readable, executable Python data structures.

This module is intentionally separate from the executable PTX path.  For Ayla
PTX it groups the source by ``// PTX template`` comments.  Compiler-generated
PTX (for example Triton or CuTe DSL) normally has no such comments, so long
unannotated regions are additionally split at recognizable pipeline
instructions.  Every occurrence is retained with its actual registers and
labels.  The complete block schedule uses the largest repeated sequences it
can find, and instruction-level affine loops provide secondary compression.

Calling ``PTX_PROGRAM()`` expands the schedule back into complete PTX.  Blank
line placement is normalized and comments are omitted, but every directive and
instruction is reconstructed with its original registers, labels, and values.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .gpu_dialect import ModuleOp
    from .ptx_codegen import PtxKernel, make_ptx_kernel
else:
    # Also support ``python ayla/ptx_python_codegen.py``. Package execution
    # with ``python -m ayla.ptx_python_codegen`` remains the preferred form.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ayla.gpu_dialect import ModuleOp
    from ayla.ptx_codegen import PtxKernel, make_ptx_kernel


_DESCRIPTION_RE = re.compile(r"^// (?:PTX template|BEGIN:|END:) `(?P<text>.*)`$")
_AFFINE_TOKEN_RE = re.compile(
    r"%(?:rd|rs|rb|r|fd|f|p)\d+|\$L\d+|_smem_\d+|(?<=\+)\d+"
)
_BINDING_TOKEN_RE = re.compile(
    r"%(?:rd|rs|rb|r|fd|f|p)\d+|\$L\d+|_smem_\d+|(?<=\+)\d+|"
    r"(?P<immediate>(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_]))"
)
_NUMBER_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")
_LABEL_RE = re.compile(r"^\s*(?P<label>\$L\d+):\s*$")
_COUNTED_BACKEDGE_RE = re.compile(
    r"add\.(?:s32|u32)\s+(?P<register>%r\d+),\s*(?P=register),\s*"
    r"(?P<step>-?\d+);\s*"
    r"setp\.(?P<comparison>lt|le|gt|ge)\.(?:s32|u32)\s+"
    r"(?P<predicate>%p\d+),\s*(?P=register),\s*(?P<bound>-?\d+);\s*"
    r"@(?P=predicate)\s+bra\s+(?P<label>\$L\d+);"
)
_BRANCH_RE = re.compile(r"\bbra(?:\.[A-Za-z0-9_]+)*\s+(?P<label>\$L\d+);")


@dataclass(frozen=True)
class _Block:
    name: str
    description: str | None
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Text:
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Loop:
    count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _ScheduleLoop:
    count: int
    body: tuple[str | "_ScheduleLoop", ...]


@dataclass
class _Template:
    name: str
    semantic_name: str
    lines: tuple[str, ...]
    prefixes: tuple[str, ...]
    occurrences: list[tuple[tuple[int, ...], ...]]
    originals: list[tuple[str, ...]]


def make_python_ptx(module: ModuleOp) -> str:
    """Lower ``module`` normally and return a readable Python/PTX lift."""
    return ptx_kernel_to_python(make_ptx_kernel(module))


def ptx_kernel_to_python(kernel: PtxKernel) -> str:
    """Return the readable lift for an already generated PTX kernel."""
    header = [
        f"# Kernel: {kernel.name}",
        f"# Grid: {kernel.grid_dim}; block: {kernel.block_dim}",
        "",
    ]
    return ptx_to_python(kernel.ptx, header=header)


def ptx_to_python(ptx: str, *, header: list[str] | None = None) -> str:
    """Convert raw PTX into Python that expands to the complete program.

    Schedule-level repetition is considered first and may cover arbitrarily
    many PTX blocks.  Within an occurrence, a loop is emitted only for
    at least three consecutive copies of a one-to-eight-line pattern, and every
    changing register, label, shared symbol, or address offset must be an
    arithmetic progression.  Unnamed ``PTX`` runs are retained because they
    contain scalar glue, directives, braces, and compiler metadata.
    """
    source_lines = ptx.splitlines()
    nonempty_lines = [line.strip() for line in source_lines if line.strip()]
    blocks = [
        block for block in _split_blocks(ptx)
        if _executable_lines(block.lines)
    ]
    templates, schedule = _factor_blocks(blocks)
    execution_weights, exact_execution_counts = _execution_profile(blocks)
    execution_qualifier = (
        "exact logical" if exact_execution_counts else "conservative estimated"
    )
    execution_counts: dict[str, int] = {}
    for name, weight in zip(schedule, execution_weights):
        execution_counts[name] = execution_counts.get(name, 0) + weight
    scheduled_instruction_executions = sum(
        weight * _instruction_statement_count(_executable_lines(block.lines))
        for block, weight in zip(blocks, execution_weights)
    )
    block_kinds = {template.semantic_name for template in templates}
    output = [
        '"""Readable, complete PTX lift; call PTX_PROGRAM() to expand it."""',
        "",
        "import re",
        "",
        (
            r'_TOKEN_RE = re.compile(r"(?P<prefix>%(?:rd|rs|rb|r|fd|f|p)|'
            r'\$L|_smem_)(?P<number>\d+)|(?P<offset>(?<=\+)\d+)|'
            r'(?P<immediate>\$I)(?P<immediate_number>\d+)")'
        ),
        "",
        (
            f"# PTX source: {len(source_lines)} total lines; "
            f"{len(nonempty_lines)} non-empty; "
            f"{len(set(nonempty_lines))} unique non-empty"
        ),
        (
            f"# Semantic schedule: {len(schedule)} static block occurrences; "
            f"{sum(execution_weights)} {execution_qualifier} PTX block "
            "executions per CTA; "
            f"{len(block_kinds)} unique block kinds"
        ),
        (
            "# Logical block counts include recognized inter-block loops; "
            "retry iterations inside a block are not expanded"
        ),
        (
            f"# Scheduled PTX instruction executions per CTA: "
            f"{scheduled_instruction_executions}; extra same-block retry "
            "iterations are excluded"
        ),
        "",
        *(header or []),
    ]
    for template in templates:
        executions = execution_counts[template.name]
        output.append(
            f"# {executions} {execution_qualifier} PTX block execution(s) "
            "per CTA from "
            f"{len(template.occurrences)} static occurrence(s)"
        )
        statements = _instruction_statement_count(template.originals[0])
        output.append(
            f"# {executions * statements} scheduled PTX instruction "
            f"execution(s) per CTA ({statements} per block execution)"
        )
        output.extend(_render_template(template))
        output.append("")

    output.extend(_render_schedule(_compress_schedule(schedule)))
    return "\n".join(output).rstrip() + "\n"


def _execution_profile(blocks: list[_Block]) -> tuple[list[int], bool]:
    """Return per-block logical execution counts and whether they are exact.

    Simple counted do-while loops emitted by Ayla are multiplied exactly.
    Same-block retry loops do not alter logical block invocation counts. An
    unrecognized branch between blocks makes the profile conservative.
    """
    weights = [1] * len(blocks)
    labels: dict[str, int] = {}
    executable = [_executable_lines(block.lines) for block in blocks]
    for index, lines in enumerate(executable):
        for line in lines:
            if match := _LABEL_RE.match(line):
                labels[match.group("label")] = index

    recognized_backedges: set[tuple[int, str]] = set()
    for end, lines in enumerate(executable):
        match = _COUNTED_BACKEDGE_RE.search("\n".join(lines))
        if match is None:
            continue
        start = labels.get(match.group("label"))
        if start is None or start >= end:
            continue
        register = re.escape(match.group("register"))
        label = re.escape(match.group("label"))
        init_match = re.search(
            rf"mov\.(?:b32|s32|u32)\s+{register},\s*(-?\d+);\s*{label}:",
            "\n".join(executable[start]),
        )
        if init_match is None:
            continue
        trip_count = _counted_loop_trip_count(
            int(init_match.group(1)),
            int(match.group("step")),
            int(match.group("bound")),
            match.group("comparison"),
        )
        if trip_count is None:
            continue
        recognized_backedges.add((end, match.group("label")))
        # The initialization in the label block executes once. The body begins
        # after that block and includes the latch/backedge block.
        for index in range(start + 1, end + 1):
            weights[index] *= trip_count

    exact = True
    for index, lines in enumerate(executable):
        for branch in _BRANCH_RE.finditer("\n".join(lines)):
            label = branch.group("label")
            target = labels.get(label)
            if target is None:
                exact = False
            elif target == index:
                continue
            elif target < index and (index, label) in recognized_backedges:
                continue
            else:
                exact = False
    return weights, exact


def _counted_loop_trip_count(
    initial: int, step: int, bound: int, comparison: str,
) -> int | None:
    """Return a do-while trip count for a monotonic counted backedge."""
    if step > 0 and comparison in ("lt", "le"):
        distance = bound - initial
        count = (
            (distance + step - 1) // step
            if comparison == "lt"
            else distance // step + 1
        )
    elif step < 0 and comparison in ("gt", "ge"):
        positive_step = -step
        distance = initial - bound
        count = (
            (distance + positive_step - 1) // positive_step
            if comparison == "gt"
            else distance // positive_step + 1
        )
    else:
        return None
    return max(1, count)


def _default_readable_ptx() -> str:
    """Generate the currently retained best GEMM kernel."""
    # Importing a GEMM collection module normally registers its kernels with
    # torch and builds the CUDA launcher. Readable PTX generation needs only
    # the pure-Python IR builder, so disable those registration side effects
    # before importing the module.
    import ayla.ptx_runtime as runtime
    import ayla.torch_custom_call as custom_call

    runtime.make_ptx_launcher = lambda kernel, **kwargs: None
    custom_call.make_torch_custom_call = lambda *args, **kwargs: None

    from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
        best_auto_matmul_4096_wgmma256_wg_tma_store_module,
    )

    module = best_auto_matmul_4096_wgmma256_wg_tma_store_module()
    return make_python_ptx(module)


def main(argv: list[str] | None = None) -> int:
    """Generate readable Python from the default kernel or an existing PTX."""
    parser = argparse.ArgumentParser(
        description="Lift complete PTX into readable, expandable Python.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "existing .ptx file to convert; omit it to build the repository's "
            "currently retained best GEMM kernel"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "output .py file; defaults to stdout for an input file, or "
            "generated_ptx/auto_matmul_k3_4tiles_readable_ptx.py for the "
            "built-in kernel"
        ),
    )
    args = parser.parse_args(argv)

    if args.input is None:
        source = _default_readable_ptx()
        output = args.output or (
            Path(__file__).resolve().parents[1]
            / "generated_ptx"
            / "auto_matmul_k3_4tiles_readable_ptx.py"
        )
    else:
        source = ptx_to_python(args.input.read_text())
        output = args.output

    if output is None:
        sys.stdout.write(source)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(source)
        print(f"wrote {output} ({len(source)} chars)")
    return 0


def _split_blocks(ptx: str) -> list[_Block]:
    segments: list[list[str]] = []
    current: list[str] = []
    raw_lines = ptx.splitlines()
    for line_index, line in enumerate(raw_lines):
        if line.strip():
            current.append(line)
        elif current:
            # Some templates start with a formatting newline.  Keep the
            # following instructions attached to their naming comment, but
            # recognize a genuinely empty template when another comment is
            # the next non-empty line.
            if (
                len(current) == 1
                and _description(current[0].strip()) is not None
            ):
                next_line = next(
                    (candidate.strip() for candidate in raw_lines[line_index + 1:]
                     if candidate.strip()),
                    "",
                )
                if _description(next_line) is None:
                    continue
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    counts: dict[str, int] = {}
    blocks: list[_Block] = []
    for segment in segments:
        # Keep source indentation and naming comments: PTX_PROGRAM() must be
        # able to reproduce every non-empty input line.
        lines = "\n".join(segment).splitlines()
        marker = lines[0].strip()
        description = _description(lines[0])
        payload_lines = lines[1:] if description is not None else lines
        if marker.startswith("// BEGIN:"):
            base = "LOOP_BEGIN"
        elif marker.startswith("// END:"):
            base = "LOOP_END"
        else:
            base = _block_name(description, payload_lines)
        # NVVM/Triton output generally contains one enormous function-body
        # segment because it lacks Ayla's template comments and blank lines.
        # Split that segment into semantic runs so its pipeline is visible in
        # the same schedule representation.  Keep short preambles intact.
        external = (
            _split_external_lines(lines)
            if description is None and len(lines) >= 24
            else [(base, lines)]
        )
        for external_base, external_lines in external:
            counts[external_base] = counts.get(external_base, 0) + 1
            occurrence = counts[external_base]
            name = (
                external_base
                if occurrence == 1
                else f"{external_base}_{occurrence}"
            )
            blocks.append(_Block(name, description, tuple(external_lines)))
    return blocks


def _split_external_lines(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split compiler-generated PTX into consecutive semantic runs.

    This is deliberately lexical rather than a PTX parser.  It does not alter
    or execute the input; it only gives important asynchronous operations names
    and leaves ordinary scalar instructions in compact ``PTX`` runs.
    """
    result: list[tuple[str, list[str]]] = []
    in_kernel_declaration = False
    saw_open_brace = False
    for line in lines:
        text = line.strip()
        lower = text.lower()
        starts_entry = lower.startswith(".entry") or (
            lower.startswith(".visible") and ".entry" in lower
        )
        if starts_entry:
            in_kernel_declaration = True

        if in_kernel_declaration:
            if saw_open_brace and not _is_kernel_declaration_line(text):
                in_kernel_declaration = False
            else:
                name = "KERNEL_DECLARATION"
                saw_open_brace = saw_open_brace or text == "{"
                if result and result[-1][0] == name:
                    result[-1][1].append(line)
                else:
                    result.append((name, [line]))
                continue

        name = _external_line_name(line)
        if result and result[-1][0] == name:
            result[-1][1].append(line)
        else:
            result.append((name, [line]))
    return result


def _is_kernel_declaration_line(text: str) -> bool:
    """Whether ``text`` can remain in an entry's declaration/prologue."""
    lower = text.lower()
    return (
        not text
        or text.startswith("//")
        or text == "{"
        or lower.startswith((
            ".reg", ".shared", ".extern .shared", ".local", ".param",
            ".reqntid", ".maxntid", ".minnctapersm", ".maxnreg",
        ))
    )


def _external_line_name(line: str) -> str:
    text = line.strip()
    lower = text.lower()
    if not text or text.startswith("//") or text in {"{", "}"}:
        return "PTX"
    if lower.startswith((".version", ".target", ".address_size")):
        return "PTX_HEADER"
    if lower.startswith((
        ".visible", ".entry", ".param", ".reqntid", ".maxntid",
        ".minnctapersm", ".maxnreg",
    )):
        return "KERNEL_DECLARATION"
    if lower.startswith(".reg"):
        return "REGISTER_DECLARATION"
    if lower.startswith((".shared", ".extern .shared")):
        return "SHARED_DECLARATION"
    if "wgmma.mma_async" in lower:
        return "WGMMA"
    if "wgmma.fence" in lower:
        return "WGMMA_FENCE"
    if "wgmma.commit_group" in lower:
        return "WGMMA_COMMIT"
    if "wgmma.wait_group" in lower:
        return "WGMMA_WAIT"
    if "cp.async.bulk.tensor" in lower:
        if ".global.shared" in lower:
            return "TMA_STORE"
        return "TMA_LOAD"
    if "cp.async.bulk.commit_group" in lower:
        return "TMA_STORE_COMMIT"
    if "cp.async.bulk.wait_group" in lower:
        return "TMA_STORE_WAIT"
    if "mbarrier.init" in lower:
        return "MBARRIER_INIT"
    if "mbarrier.try_wait" in lower:
        return "TMA_LOAD_WAIT"
    if "mbarrier.arrive.expect_tx" in lower:
        return "TMA_LOAD_ARM"
    if "mbarrier.arrive" in lower:
        return "MBARRIER_ARRIVE"
    if lower.startswith(("bar.sync", "bar.arrive")):
        return "BARRIER"
    if "stmatrix." in lower:
        return "STMATRIX_STORE"
    if "setmaxnreg." in lower:
        return "REGISTER_BUDGET"
    if lower.startswith("prefetch.tensormap"):
        return "TMA_DESCRIPTOR_PREFETCH"
    if lower.startswith(("fence.mbarrier", "fence.proxy.async")):
        return "ASYNC_FENCE"
    if lower.startswith("ret"):
        return "KERNEL_RETURN"
    if text.endswith(":") or re.search(r"\bbra(?:\.\w+)*\b", lower):
        return "CONTROL_FLOW"
    return "PTX"


def _description(line: str) -> str | None:
    match = _DESCRIPTION_RE.match(line.strip())
    return match.group("text") if match else None


def _block_name(description: str | None, lines: list[str]) -> str:
    if description is None:
        joined = "\n".join(lines)
        if any(".visible .entry" in line for line in lines):
            return "KERNEL_DECLARATION"
        if lines and lines[0].startswith(".version"):
            return "PTX_HEADER"
        external_names = {
            _external_line_name(line)
            for line in lines
            if _external_line_name(line) != "PTX"
        }
        if len(external_names) == 1:
            return external_names.pop()
        if "%ctaid" in joined:
            return "BLOCK_IDS"
        if "shfl.sync.idx" in joined and "%tid" in joined:
            return "WARP_GROUP_ID"
        if "%tid" in joined:
            return "THREAD_IDS"
        if "ret;" in joined:
            return "KERNEL_RETURN"
        if any(opcode in joined for opcode in (
            "add.", "mul.", "mad.", "shl.", "shr.", "cvt.", "mov.",
        )):
            return "ADDRESS_CALCULATION"
        return "PTX"

    lower = description.lower()
    if "create tma l2 cache" in lower:
        return "TMA_CACHE_POLICY"
    if "stmatrix_store" in lower:
        return "STMATRIX_STORE"
    if "tma_move(" in lower:
        first_arg = lower.split("tma_move(", 1)[1].split(",", 1)[0]
        return "TMA_STORE" if "tma_desc" in first_arg else "TMA_LOAD"
    if "async(" in lower and "lock<mbar>" in lower:
        return "TMA_LOAD"
    if "await(" in lower and "lock<mbar>" in lower:
        return "TMA_LOAD_WAIT"
    if "await(" in lower and "lock<bulk>" in lower:
        return "TMA_STORE_WAIT"
    if "await(" in lower and "lock<wgmma>" in lower:
        return "WGMMA_WAIT"
    if "lock<wgmma>" in lower or "matmul(" in lower:
        return "WGMMA"
    if "convert(" in lower:
        return "CONVERT"
    if "barrier(" in lower:
        if 'reason: "mbarrier_init"' in lower:
            return "MBARRIER_INIT_SYNC"
        scope = re.search(r'scope: "([a-z_]+)"', lower)
        if scope is not None:
            return f"{_identifier(scope.group(1))}_BARRIER"
        return "BARRIER"
    if lower.startswith("for ("):
        return "LOOP_CONTROL"
    if re.match(r"^%\S+:\s*smem_tensor", lower):
        return "SMEM_ALLOC"
    if re.match(r"^%\S+:\s*wrmem_tensor", lower):
        return "WRMEM_ALLOC"
    if re.match(r"^%\S+:\s*lock<mbar>", lower):
        return "MBARRIER_ALLOC"
    if "alloc()" in lower:
        return "ALLOC"

    operation = re.search(r"(?:=\s*)?([a-z_][a-z0-9_]*)\(", lower)
    if operation:
        return _identifier(operation.group(1))
    return "PTX_TEMPLATE"


def _identifier(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    if not name or name[0].isdigit():
        name = f"PTX_{name}"
    return name


def _semantic_name(name: str) -> str:
    """Drop only the uniqueness suffix added by :func:`_split_blocks`."""
    return re.sub(r"_\d+$", "", name)


def _factor_blocks(blocks: list[_Block]) -> tuple[list[_Template], list[str]]:
    """Factor occurrences into alpha-equivalent parameterized templates."""
    templates: list[_Template] = []
    by_shape: dict[tuple[str, tuple[str, ...]], _Template] = {}
    variants: dict[str, int] = {}
    schedule: list[str] = []
    for block in blocks:
        lines = _executable_lines(block.lines)
        if not lines:
            continue
        semantic = _semantic_name(block.name)
        normalized, prefixes, occurrence = _alpha_template(lines)
        key = (semantic, normalized)
        template = by_shape.get(key)
        if template is None:
            variant = variants.get(semantic, 0) + 1
            variants[semantic] = variant
            name = semantic if variant == 1 else f"{semantic}_VARIANT_{variant}"
            template = _Template(
                name,
                semantic,
                normalized,
                prefixes,
                [],
                [],
            )
            by_shape[key] = template
            templates.append(template)
        else:
            assert template.prefixes == prefixes
        template.occurrences.append(occurrence)
        template.originals.append(lines)
        schedule.append(template.name)
    # A one-off block has nothing to factor. Keep its actual spelling and
    # avoid paying for a canonical template plus a binding table.
    for template in templates:
        if len(template.occurrences) == 1:
            template.lines = template.originals[0]
            template.prefixes = ()
            template.occurrences = [()]
    return templates, schedule


def _executable_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Drop comments while retaining every PTX directive and instruction."""
    result: list[str] = []
    for line in lines:
        code = line.split("//", 1)[0].rstrip()
        if code.strip():
            result.append(code)
    return tuple(result)


def _instruction_statement_count(lines: tuple[str, ...]) -> int:
    """Count PTX statements that execute, rather than declarations/labels."""
    return sum(
        line.strip().endswith(";") and not line.lstrip().startswith(".")
        for line in lines
    )


def _alpha_template(
    lines: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Return canonical source plus compact register/label bindings."""
    indices: dict[tuple[str, int], int] = {}
    values: dict[str, list[int]] = {}
    prefixes: list[str] = []

    def replace(match: re.Match) -> str:
        token = match.group()
        if match.lastgroup == "immediate":
            prefix = "$I"
            value = int(token)
        else:
            parsed = _NUMBER_SUFFIX_RE.match(token)
            assert parsed is not None
            prefix = parsed.group("prefix")
            value = int(parsed.group("number"))
        if prefix not in values:
            values[prefix] = []
            prefixes.append(prefix)
        key = (prefix, value)
        if key not in indices:
            indices[key] = len(values[prefix])
            values[prefix].append(value)
        return f"{prefix}{indices[key]}"

    normalized = tuple(_BINDING_TOKEN_RE.sub(replace, line) for line in lines)
    prefix_tuple = tuple(prefixes)
    occurrence = tuple(tuple(values[prefix]) for prefix in prefix_tuple)
    return normalized, prefix_tuple, occurrence


def _render_template(template: _Template) -> list[str]:
    """Render one structural definition and its compact occurrence bindings."""
    result: list[str] = []
    parts = _compress_lines(list(template.lines))
    singular = len(template.occurrences) == 1 and not template.prefixes
    value_name = template.name if singular else f"_{template.name.lower()}_template"
    if len(parts) == 1 and isinstance(parts[0], _Text):
        result.extend(_assignment(value_name, parts[0].lines))
    else:
        result.append(f"{value_name} = []")
        for part in parts:
            if isinstance(part, _Text):
                result.extend(_append(value_name, part.lines))
            else:
                result.append(f"for i in range({part.count}):")
                for line in part.lines:
                    result.extend(
                        _append(
                            value_name,
                            (line,),
                            indent=4,
                            formatted=True,
                        )
                    )
        result.append(f'{value_name} = "\\n".join({value_name})')

    if singular:
        return result

    prefixes = repr(template.prefixes)
    occurrences = _occurrences_expr(template.occurrences)
    result.extend([
        f"{template.name} = (",
        f"    {value_name},",
        f"    {prefixes},",
        *(f"    {line}" for line in occurrences),
        ")",
    ])
    return result


def _occurrences_expr(
    occurrences: list[tuple[tuple[int, ...], ...]],
) -> list[str]:
    expressions = [_occurrence_expr(occurrence) for occurrence in occurrences]
    if len(set(expressions)) == 1 and len(expressions) > 1:
        return [f"({expressions[0]},) * {len(expressions)}"]
    if len(expressions) == 1:
        return [f"({expressions[0]},)"]
    return ["(", *(f"    {expression}," for expression in expressions), ")"]


def _occurrence_expr(occurrence: tuple[tuple[int, ...], ...]) -> str:
    items = [_integer_sequence_expr(values) for values in occurrence]
    if not items:
        return "()"
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


def _integer_sequence_expr(values: tuple[int, ...]) -> str:
    if len(values) >= 2:
        step = values[1] - values[0]
        if step and all(
            value == values[0] + step * index
            for index, value in enumerate(values)
        ):
            return f"range({values[0]}, {values[-1] + step}, {step})"
    return repr(values)


def _compress_schedule(names: list[str]) -> tuple[str | _ScheduleLoop, ...]:
    """Greedily fold the largest adjacent repeated block sequences."""
    result: list[str | _ScheduleLoop] = []
    index = 0
    while index < len(names):
        remaining = len(names) - index
        best: tuple[int, int, int] | None = None
        for width in range(1, remaining // 2 + 1):
            count = 1
            pattern = names[index:index + width]
            while (
                index + (count + 1) * width <= len(names)
                and names[
                    index + count * width:index + (count + 1) * width
                ] == pattern
            ):
                count += 1
            if count < 2:
                continue
            saved = count * width - width - 1
            candidate = (saved, count * width, width)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[0] <= 0:
            result.append(names[index])
            index += 1
            continue
        _, span, width = best
        count = span // width
        body = _compress_schedule(names[index:index + width])
        result.append(_ScheduleLoop(count, body))
        index += span
    return tuple(result)


def _render_schedule(nodes: tuple[str | _ScheduleLoop, ...]) -> list[str]:
    output = [
        "def PTX_PROGRAM():",
        "    output = []",
        "    positions = {}",
        "    def emit(block):",
        "        if isinstance(block, str):",
        "            output.append(block)",
        "            return",
        "        template, prefixes, occurrences = block",
        "        key = id(block)",
        "        index = positions.get(key, 0)",
        "        if not prefixes:",
        "            output.append(template)",
        "        else:",
        "            bindings = dict(zip(prefixes, occurrences[index]))",
        "            def replace(match):",
        "                if match.group('immediate') is not None:",
        "                    prefix = '$I'",
        "                    number = int(match.group('immediate_number'))",
        "                elif match.group('offset') is not None:",
        "                    prefix = ''",
        "                    number = int(match.group('offset'))",
        "                else:",
        "                    prefix = match.group('prefix')",
        "                    number = int(match.group('number'))",
        "                output_prefix = '' if prefix == '$I' else prefix",
        "                return output_prefix + str(bindings[prefix][number])",
        "            output.append(_TOKEN_RE.sub(replace, template))",
        "        positions[key] = index + 1",
    ]
    loop_counter = [0]

    def render(items: tuple[str | _ScheduleLoop, ...], indent: int) -> None:
        pad = " " * indent
        for item in items:
            if isinstance(item, str):
                output.append(f"{pad}emit({item})")
                continue
            loop_name = f"repeat_{loop_counter[0]}"
            loop_counter[0] += 1
            output.append(f"{pad}for {loop_name} in range({item.count}):")
            render(item.body, indent + 4)

    render(nodes, 4)
    output.extend([
        '    return "\\n\\n".join(output)',
        "",
    ])
    return output


def _assignment(name: str, lines: tuple[str, ...]) -> list[str]:
    return _string_statement(f"{name} = ", lines)


def _append(
    name: str,
    lines: tuple[str, ...],
    *,
    indent: int = 0,
    formatted: bool = False,
) -> list[str]:
    prefix = "f" if formatted else "r"
    return _string_statement(
        f"{name}.append(", lines, indent=indent, prefix=prefix, suffix=")",
    )


def _string_statement(
    statement: str,
    lines: tuple[str, ...],
    *,
    indent: int = 0,
    prefix: str = "r",
    suffix: str = "",
) -> list[str]:
    """Format a triple-quoted assignment without wrapper-only lines."""
    pad = " " * indent
    if not lines:
        return [f'{pad}{statement}""{suffix}']
    if len(lines) == 1:
        return [f'{pad}{statement}{prefix}"""{lines[0]}"""{suffix}']
    return [
        f'{pad}{statement}{prefix}"""{lines[0]}',
        *(f"{pad}{line}" for line in lines[1:-1]),
        f'{pad}{lines[-1]}"""{suffix}',
    ]


def _compress_lines(lines: list[str]) -> list[_Text | _Loop]:
    parts: list[_Text | _Loop] = []
    pending: list[str] = []
    index = 0
    while index < len(lines):
        loop = _best_loop(lines, index)
        if loop is None:
            pending.append(lines[index])
            index += 1
            continue
        if pending:
            parts.append(_Text(tuple(pending)))
            pending = []
        parts.append(loop)
        index += loop.count * len(loop.lines)
    if pending:
        parts.append(_Text(tuple(pending)))
    return parts or [_Text(())]


def _best_loop(lines: list[str], start: int) -> _Loop | None:
    remaining = len(lines) - start
    best: tuple[int, _Loop] | None = None
    for width in range(1, min(8, remaining // 3) + 1):
        max_count = remaining // width
        for count in range(3, max_count + 1):
            samples = [
                lines[start + iteration * width:start + (iteration + 1) * width]
                for iteration in range(count)
            ]
            templates = _infer_chunk(samples)
            if templates is None:
                break
            saved_lines = count * width - width - 1
            candidate = _Loop(count, tuple(templates))
            if best is None or saved_lines > best[0]:
                best = (saved_lines, candidate)
    return best[1] if best is not None and best[0] > 0 else None


def _infer_chunk(chunks: list[list[str]]) -> list[str] | None:
    return_lines: list[str] = []
    for column in zip(*chunks):
        inferred = _infer_affine_line(list(column))
        if inferred is None:
            return None
        return_lines.append(inferred)
    return return_lines


def _infer_affine_line(lines: list[str]) -> str | None:
    tokenized = [_tokenize(line) for line in lines]
    literals, prefixes, _ = tokenized[0]
    if any(item[:2] != (literals, prefixes) for item in tokenized[1:]):
        return None

    values_by_token = list(zip(*(item[2] for item in tokenized)))
    steps: list[int] = []
    for values in values_by_token:
        step = values[1] - values[0]
        if any(value != values[0] + step * i for i, value in enumerate(values)):
            return None
        steps.append(step)

    output: list[str] = []
    for index, literal in enumerate(literals):
        output.append(_escape_fstring(literal))
        if index == len(prefixes):
            continue
        prefix = _escape_fstring(prefixes[index])
        base = values_by_token[index][0]
        step = steps[index]
        output.append(prefix + (str(base) if step == 0 else "{" + _affine(base, step) + "}"))
    return "".join(output)


def _tokenize(line: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    literals: list[str] = []
    prefixes: list[str] = []
    values: list[int] = []
    cursor = 0
    for match in _AFFINE_TOKEN_RE.finditer(line):
        literals.append(line[cursor:match.start()])
        token = match.group()
        number = _NUMBER_SUFFIX_RE.match(token)
        assert number is not None
        prefixes.append(number.group("prefix"))
        values.append(int(number.group("number")))
        cursor = match.end()
    literals.append(line[cursor:])
    return tuple(literals), tuple(prefixes), tuple(values)


def _affine(base: int, step: int) -> str:
    if step == 1:
        return f"{base} + i"
    return f"{base} {('+' if step > 0 else '-')} {abs(step)} * i"


def _escape_fstring(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


if __name__ == "__main__":
    raise SystemExit(main())
