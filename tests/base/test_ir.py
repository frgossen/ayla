"""Tests for the generic IR primitives (no dialect ops)."""

import textwrap

from ayla.base.ir import Block, Dialect, Op, Type, Value, clone_op


def assert_print_and_roundtrip(op: Op, expected: str) -> None:
    """Assert `op` prints to `expected` and round-trips through
    Dialect.empty().parse equal to itself.
    """
    text = repr(op)
    assert text == expected
    assert repr(Dialect.empty().parse(text)) == text


def test_minimal():
    op = Op(operands=[], results=[])
    assert_print_and_roundtrip(op, "generic()")


def test_with_attrs_and_blocks():
    op = Op(
        operands=[],
        results=[],
        attrs={
            "i": 42,
            "f": 3.5,
            "b": True,
            "s": "hello",
            "lst": [1, "x", [2, 3]],
            "d": {"k": [True, None]},
        },
        blocks=[
            Block(name="a"),
            Block(name="b", ops=[Op(operands=[], results=[])]),
            Block(name="c", ops=[Op(operands=[], results=[])]),
            Block(name="d"),
        ],
    )
    expected = textwrap.dedent(
        """\
        generic() attrs={b: True, d: {k: [True, None]}, f: 3.5, i: 42, lst: [1, "x", [2, 3]], s: "hello"} {
        ^a:
        }
        {
        ^b:
          generic()
        }
        {
        ^c:
          generic()
        }
        {
        ^d:
        }"""
    )  # noqa: E501
    assert_print_and_roundtrip(op, expected)


def test_with_operands_and_results():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    c = Value(name="c", type=t)
    inner = Op(operands=[a, b], results=[c])
    block = Block(name="entry", arguments=[a, b], ops=[inner])
    op = Op(operands=[], results=[], blocks=[block])
    expected = textwrap.dedent(
        """\
        generic() {
        ^entry(%a: generic, %b: generic):
          %c: generic = generic(%a: generic, %b: generic)
        }"""
    )
    assert_print_and_roundtrip(op, expected)


def test_with_block_args():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    block = Block(name="entry", arguments=[a, b])
    op = Op(operands=[], results=[], blocks=[block])
    expected = textwrap.dedent(
        """\
        generic() {
        ^entry(%a: generic, %b: generic):
        }"""
    )
    assert_print_and_roundtrip(op, expected)


def test_nested_ops():
    t = Type()
    ta = Type(attrs={"bits": 32})
    a = Value(name="a", type=ta)
    b = Value(name="b", type=t)
    entry = Block(
        name="entry",
        arguments=[a, b],
        ops=[
            Op(operands=[a, b], results=[Value(name=None, type=t)]),
        ],
    )
    cont = Block(
        name="cont",
        ops=[
            Op(operands=[a], results=[], attrs={"flag": True}),
        ],
    )
    c = Value(name="c", type=t)
    other = Block(arguments=[c], ops=[Op(operands=[c], results=[])])

    op = Op(operands=[], results=[], blocks=[entry, cont, other])
    expected = textwrap.dedent(
        """\
        generic() {
        ^entry(%a: generic<bits: 32>, %b: generic):
          %0: generic = generic(%a: generic<bits: 32>, %b: generic)
        }
        {
        ^cont:
          generic(%a: generic<bits: 32>) attrs={flag: True}
        }
        {
        ^bb0(%c: generic):
          generic(%c: generic)
        }"""
    )
    assert_print_and_roundtrip(op, expected)


# ---------------------------------------------------------------------------
# Use-list invariant.
#
#   value.uses contains (op, i) iff op.operands[i] is value and op is anchored
#   (held by a block, directly or nested inside an anchored op's blocks).
#
# These drive mutations the *plain* way — no helper methods — to prove the
# invariant is self-maintaining through the two observable containers.
# ---------------------------------------------------------------------------


def test_operand_item_assignment_moves_use():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    op = Op(operands=[a], results=[])
    Block(ops=[op])  # anchors op

    assert a.uses == {(op, 0)}

    op.operands[0] = b
    assert a.uses == set()
    assert b.uses == {(op, 0)}


def test_operand_list_reassignment_reorder_and_grow():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    c = Value(name="c", type=t)
    op = Op(operands=[a, b], results=[])
    Block(ops=[op])

    assert a.uses == {(op, 0)}
    assert b.uses == {(op, 1)}

    # reorder + grow
    op.operands = [b, a, c]
    assert b.uses == {(op, 0)}
    assert a.uses == {(op, 1)}
    assert c.uses == {(op, 2)}

    # shrink
    op.operands = [c]
    assert a.uses == set()
    assert b.uses == set()
    assert c.uses == {(op, 0)}


def test_free_floating_op_reassignment_does_not_register():
    t = Type()
    a = Value(name="a", type=t)
    op = Op(operands=[], results=[])  # never placed in a block
    op.operands = [a]
    assert a.uses == set()


def test_block_ops_inplace_mutations():
    t = Type()
    a = Value(name="a", type=t)
    op1 = Op(operands=[a], results=[])
    op2 = Op(operands=[a], results=[])
    block = Block(ops=[])

    # free-floating ops are not registered
    assert a.uses == set()

    block.ops.append(op1)
    assert (op1, 0) in a.uses

    block.ops.insert(0, op2)
    assert (op2, 0) in a.uses

    assert block.ops.pop() is op1
    assert (op1, 0) not in a.uses
    assert (op2, 0) in a.uses

    block.ops.remove(op2)
    assert a.uses == set()

    op3 = Op(operands=[a], results=[])
    block.ops.append(op3)
    assert (op3, 0) in a.uses
    del block.ops[0]
    assert a.uses == set()


def test_block_ops_reassignment_diffs():
    t = Type()
    a = Value(name="a", type=t)
    op1 = Op(operands=[a], results=[])
    op2 = Op(operands=[a], results=[])
    block = Block(ops=[op1])

    assert a.uses == {(op1, 0)}

    block.ops = [op2]  # dropped op1, added op2
    assert a.uses == {(op2, 0)}


def test_removing_block_op_clears_nested_uses():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    outer = Op(
        operands=[],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    root = Block(ops=[outer])

    assert (inner, 0) in a.uses

    root.ops = []  # cascade: dropping outer clears inner's uses too
    assert a.uses == set()


def test_transplant_keeps_use_removes_before_adds():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    wrapper = Op(
        operands=[],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    root = Block(ops=[wrapper])

    assert (inner, 0) in a.uses

    # Move the inner op into a fresh group, then drop the old wrapper by
    # reassigning root.ops. Removes-before-adds must keep inner anchored.
    new_group = Block(ops=[inner])
    new_wrapper = Op(
        operands=[],
        results=[],
        blocks=[new_group],
    )
    root.ops = [new_wrapper]

    assert (inner, 0) in a.uses


def test_build_is_free_floating_until_placed():
    t = Type()
    a = Value(name="a", type=t)
    op = Op.build(operands=[a])
    assert a.uses == set()

    Block(ops=[op])
    assert a.uses == {(op, 0)}


def test_clone_subtree_no_double_count():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    outer = Op(
        operands=[a],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    Block(ops=[outer])

    cloned = clone_op(outer, {})
    Block(ops=[cloned])

    # a is used once by each of: outer, inner, cloned outer, cloned inner.
    assert len(a.uses) == 4
    assert len(a.users()) == 4


def test_users_dedupes_repeated_operand():
    t = Type()
    a = Value(name="a", type=t)
    op = Op(operands=[a, a], results=[])
    Block(ops=[op])

    assert a.uses == {(op, 0), (op, 1)}
    assert a.users() == [op]


def test_parsed_ir_registers_uses():
    src = textwrap.dedent(
        """\
        generic() {
        ^entry(%a: generic):
          generic(%a: generic)
        }"""
    )
    op = Dialect.empty().parse(src)
    block = op.blocks[0]
    a = block.arguments[0]
    inner = block.ops[0]
    assert a.uses == {(inner, 0)}
