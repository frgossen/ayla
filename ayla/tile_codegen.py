"""PTX acquire/publication for the tile scheduler readiness events."""

from .analysis.cross_loop_scheduler import StaticU32Expression


def static_u32_ptx(
    expression: StaticU32Expression, tile: str, result: str, prefix: str
) -> str:
    base, terms = expression
    code = f"mov.u32 {result}, {base};\n"
    for i, (coefficient, stride, period) in enumerate(terms):
        coordinate = f"${{b32:{prefix}_{i}}}"
        if stride == 1:
            code += f"mov.u32 {coordinate}, {tile};\n"
        elif stride & (stride - 1) == 0:
            code += f"shr.u32 {coordinate}, {tile}, {stride.bit_length() - 1};\n"
        else:
            code += f"div.u32 {coordinate}, {tile}, {stride};\n"
        if period & (period - 1) == 0:
            code += f"and.b32 {coordinate}, {coordinate}, {period - 1};\n"
        else:
            code += f"rem.u32 {coordinate}, {coordinate}, {period};\n"
        code += f"mad.lo.u32 {result}, {coordinate}, {coefficient}, {result};\n"
    return code


def static_publication_ptx(expressions: tuple[StaticU32Expression, ...]) -> str:
    code = """
        cp.async.bulk.wait_group 0;
        bar.sync 0;
        mov.u32 ${b32:tid}, %tid.x;
        setp.ne.u32 ${pred:other}, ${b32:tid}, 0;
        @${pred:other} bra ${label:done};
    """
    for i, expression in enumerate(expressions):
        event = f"${{b32:event_{i}}}"
        code += static_u32_ptx(
            expression, "${b32:operand.offset.val}", event, f"event_key_{i}"
        )
        code += f"""
            mul.wide.u32 ${{b64:counter_offset}}, {event}, 128;
            add.u64 ${{b64:counter}}, ${{b64:operand.sync.ptr}}, ${{b64:counter_offset}};
            red.release.gpu.global.add.u32 [${{b64:counter}}], 1;
        """
    return code + "${label:done}:\nbar.sync 0;\n"


TILE_WAIT_PTX = """
// Event zero means no new acquire is needed. The preceding publication
// already joined this CTA; external inputs are ordered by kernel launch.
setp.eq.u32 ${pred:no_wait}, ${b32:operand.event.val}, 0;
@${pred:no_wait} bra ${label:local};
mov.u32 ${b32:tid}, %tid.x;
setp.ne.u32 ${pred:other}, ${b32:tid}, 0;
@${pred:other} bra ${label:ready};
mul.wide.u32 ${b64:offset}, ${b32:operand.event.val}, 128;
add.u64 ${b64:counter}, ${b64:operand.sync.ptr}, ${b64:offset};
${label:wait}:
ld.acquire.gpu.global.u32 ${b32:seen}, [${b64:counter}];
setp.lt.u32 ${pred:pending}, ${b32:seen}, ${b32:operand.expected.val};
@${pred:pending} nanosleep.u32 64;
@${pred:pending} bra ${label:wait};
${label:ready}:
bar.sync 0;
fence.proxy.async.global;
bra ${label:done};
${label:local}:
setp.eq.u32 ${pred:no_fence}, ${b32:operand.expected.val}, 0;
@${pred:no_fence} bra ${label:done};
fence.proxy.async.global;
${label:done}:
"""

TILE_PUBLISH_PTX = """
// Read completion only permits SMEM reuse. Full completion makes
// every TMA store globally visible before the release counter.
cp.async.bulk.wait_group 0;
bar.sync 0;
mov.u32 ${b32:tid}, %tid.x;
setp.ne.u32 ${pred:other}, ${b32:tid}, 0;
@${pred:other} bra ${label:done};
mov.u32 ${b32:i}, 0;
mov.u64 ${b64:table}, ayla_tile_publications;
${label:next}:
setp.ge.u32 ${pred:end}, ${b32:i}, ${b32:operand.length.val};
@${pred:end} bra ${label:done};
add.u32 ${b32:entry}, ${b32:operand.offset.val}, ${b32:i};
mul.wide.u32 ${b64:offset}, ${b32:entry}, 4;
add.u64 ${b64:address}, ${b64:table}, ${b64:offset};
ld.global.u32 ${b32:event}, [${b64:address}];
mul.wide.u32 ${b64:offset}, ${b32:event}, 128;
add.u64 ${b64:counter}, ${b64:operand.sync.ptr}, ${b64:offset};
// Publication only releases completed producer writes. The consumer's
// acquire observes the release sequence of all producer RMW arrivals.
// No producer needs the old counter value or an acquire here.
red.release.gpu.global.add.u32 [${b64:counter}], 1;
add.u32 ${b32:i}, ${b32:i}, 1;
bra ${label:next};
${label:done}:
bar.sync 0;
"""

# The common first==last case polls once. Crossing a projection boundary
# acquires both flags before one block barrier and async proxy fence.
TILE_WAIT_PAIR_PTX = """
or.b32 ${b32:flags}, ${b32:operand.first.val}, ${b32:operand.last.val};
setp.eq.u32 ${pred:empty}, ${b32:flags}, 0;
@${pred:empty} bra ${label:done};
mov.u32 ${b32:tid}, %tid.x;
setp.ne.u32 ${pred:other}, ${b32:tid}, 0;
@${pred:other} bra ${label:ready};
setp.eq.u32 ${pred:no_first}, ${b32:operand.first.val}, 0;
@${pred:no_first} bra ${label:second};
mul.wide.u32 ${b64:offset}, ${b32:operand.first.val}, 128;
add.u64 ${b64:counter}, ${b64:operand.sync.ptr}, ${b64:offset};
${label:wait_first}:
ld.acquire.gpu.global.u32 ${b32:seen}, [${b64:counter}];
setp.eq.u32 ${pred:pending}, ${b32:seen}, 0;
@${pred:pending} nanosleep.u32 64;
@${pred:pending} bra ${label:wait_first};
${label:second}:
setp.eq.u32 ${pred:same}, ${b32:operand.first.val}, ${b32:operand.last.val};
@${pred:same} bra ${label:ready};
setp.eq.u32 ${pred:no_last}, ${b32:operand.last.val}, 0;
@${pred:no_last} bra ${label:ready};
mul.wide.u32 ${b64:offset}, ${b32:operand.last.val}, 128;
add.u64 ${b64:counter}, ${b64:operand.sync.ptr}, ${b64:offset};
${label:wait_last}:
ld.acquire.gpu.global.u32 ${b32:seen}, [${b64:counter}];
setp.eq.u32 ${pred:pending}, ${b32:seen}, 0;
@${pred:pending} nanosleep.u32 64;
@${pred:pending} bra ${label:wait_last};
${label:ready}:
bar.sync 0;
fence.proxy.async.global;
${label:done}:
"""
