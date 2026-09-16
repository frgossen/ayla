"""Tests for the generic IR primitives (no dialect ops)."""

import textwrap

from ayla.base.ir import (
    Block,
    Dialect,
    Op,
    Type,
    Value,
    clone_op,
    clone_type,
    find_ancestor,
    find_ancestor_of_type,
    find_ancestors,
    find_ancestors_of_type,
    iter_ancestors,
)


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


def test_find_ancestors():
    class ContainerOp(Op):
        pass

    leaf = Op(operands=[], results=[])
    inner = ContainerOp(operands=[], results=[], blocks=[Block(ops=[leaf])])
    outer = ContainerOp(operands=[], results=[], blocks=[Block(ops=[inner])])

    def is_container(op: Op) -> bool:
        return isinstance(op, ContainerOp)

    assert list(iter_ancestors(leaf)) == [inner, outer]
    assert find_ancestors(leaf, is_container) == [inner, outer]
    assert find_ancestor(leaf, is_container) is inner
    assert find_ancestors_of_type(leaf, ContainerOp) == [inner, outer]
    assert find_ancestor_of_type(leaf, ContainerOp) is inner
    assert find_ancestor_of_type(inner, ContainerOp) is outer
    assert find_ancestor_of_type(outer, ContainerOp) is None


# ---------------------------------------------------------------------------
# Use-list invariant.
#
#   value.uses  contains (op, i) iff op.operands[i] is value -- structural,
#               regardless of where the op sits.
#   value.users() keeps only those sharing a tree root with value.definition.
#
# These drive mutations the *plain* way -- no helper methods -- to prove both
# stay right through the observable containers. Values are block arguments so
# they have a definition, and therefore a tree to be judged against.
# ---------------------------------------------------------------------------


def test_operand_item_assignment_moves_use():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    op = Op(operands=[a], results=[])
    Block(arguments=[a], ops=[op])

    assert a.users() == [op]

    op.operands[0] = b
    assert a.uses == set()
    assert b.uses == {(op, 0)}


def test_operand_list_reassignment_reorder_and_grow():
    t = Type()
    a = Value(name="a", type=t)
    b = Value(name="b", type=t)
    c = Value(name="c", type=t)
    op = Op(operands=[a, b], results=[])
    Block(arguments=[a, b], ops=[op])

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


def test_op_outside_the_tree_is_not_a_user():
    t = Type()
    a = Value(name="a", type=t)
    Block(arguments=[a])  # `a` now belongs to that block's tree
    op = Op(operands=[], results=[])  # never placed in it
    op.operands = [a]

    assert a.uses == {(op, 0)}  # structural: the reference is real
    assert a.users() == []  # but the op is not part of `a`'s tree


def test_block_ops_inplace_mutations():
    t = Type()
    a = Value(name="a", type=t)
    op1 = Op(operands=[a], results=[])
    op2 = Op(operands=[a], results=[])
    block = Block(arguments=[a])

    # ops outside the tree are not users of it
    assert a.users() == []

    block.ops.append(op1)
    assert a.users() == [op1]

    block.ops.insert(0, op2)
    assert set(a.users()) == {op1, op2}

    assert block.ops.pop() is op1
    assert a.users() == [op2]

    block.ops.remove(op2)
    assert a.users() == []

    op3 = Op(operands=[a], results=[])
    block.ops.append(op3)
    assert a.users() == [op3]
    del block.ops[0]
    assert a.users() == []


def test_block_ops_reassignment_diffs():
    t = Type()
    a = Value(name="a", type=t)
    op1 = Op(operands=[a], results=[])
    op2 = Op(operands=[a], results=[])
    block = Block(arguments=[a], ops=[op1])

    assert a.users() == [op1]

    block.ops = [op2]  # dropped op1, added op2
    assert a.users() == [op2]


def test_removing_block_op_clears_nested_uses():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    outer = Op(
        operands=[],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    root = Block(arguments=[a], ops=[outer])

    assert a.users() == [inner]

    root.ops = []  # dropping outer takes the whole subtree with it
    assert a.users() == []


def test_transplant_keeps_use():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    wrapper = Op(
        operands=[],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    root = Block(arguments=[a], ops=[wrapper])

    assert (inner, 0) in a.uses

    # Move the inner op into a fresh group, then drop the old wrapper by
    # reassigning root.ops. Inner must end up rooted under the new wrapper,
    # whichever order the two lists happen to sync in.
    new_group = Block(ops=[inner])
    new_wrapper = Op(
        operands=[],
        results=[],
        blocks=[new_group],
    )
    root.ops = [new_wrapper]

    assert (inner, 0) in a.uses


def test_build_is_not_a_user_until_placed():
    t = Type()
    a = Value(name="a", type=t)
    block = Block(arguments=[a])
    op = Op(operands=[a], results=[])
    assert a.users() == []

    block.ops.append(op)
    assert a.users() == [op]


def test_clone_subtree_no_double_count():
    t = Type()
    a = Value(name="a", type=t)
    inner = Op(operands=[a], results=[])
    outer = Op(
        operands=[a],
        results=[],
        blocks=[Block(ops=[inner])],
    )
    Block(arguments=[a], ops=[outer])

    cloned = clone_op(outer, {})

    # The clone references `a` too, but sits in its own tree...
    assert len(a.uses) == 4
    assert len(a.users()) == 2

    # ...until it is spliced into the same one.
    outer.parent_block.ops.append(cloned)
    assert len(a.users()) == 4


def test_clone_type_overrides_one_attr():
    """The override wins, the rest carries over, and the original is untouched."""
    t = Type(attrs={"sizes": [4, 4], "swizzle": "128B"})

    cloned = clone_type(t, {"swizzle": None})

    assert cloned.attrs == {"sizes": [4, 4], "swizzle": None}
    assert t.attrs == {"sizes": [4, 4], "swizzle": "128B"}


def test_users_dedupes_repeated_operand():
    t = Type()
    a = Value(name="a", type=t)
    op = Op(operands=[a, a], results=[])
    Block(arguments=[a], ops=[op])

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


def test_parent_links_track_containers():
    """op.parent_block / block.parent_op follow placement, both ways."""
    inner = Op(operands=[], results=[])
    body = Block(ops=[inner])
    root = Op(operands=[], results=[], blocks=[body])

    assert inner.parent_block is body
    assert body.parent_op is root
    assert root.parent_block is None  # free-floating

    # Removal clears the link; reassignment diffs it.
    body.ops = []
    assert inner.parent_block is None
    root.blocks = []
    assert body.parent_op is None

    # Transplant: the new owner claims inner first, so the old block's later
    # sync must not steal the link back.
    old_block = Block(ops=[inner])
    new_block = Block(ops=[inner])
    old_block.ops = []
    assert inner.parent_block is new_block
