"""Tests for the generic IR primitives (no dialect ops)."""

import textwrap

from ayla.base.ir import Block, Dialect, Op, Region, Type, Value


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


def test_with_attrs_and_regions():
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
        regions=[
            Region(
                blocks=[
                    Block(name="a"),
                    Block(name="b", ops=[Op(operands=[], results=[])]),
                ]
            ),
            Region(
                blocks=[
                    Block(name="c", ops=[Op(operands=[], results=[])]),
                    Block(name="d"),
                ]
            ),
        ],
    )
    expected = textwrap.dedent(
        """\
        generic() attrs={b: True, d: {k: [True, None]}, f: 3.5, i: 42, lst: [1, "x", [2, 3]], s: "hello"} {
        ^a:
        ^b:
          generic()
        }
        {
        ^c:
          generic()
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
    op = Op(operands=[], results=[], regions=[Region(blocks=[block])])
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
    op = Op(operands=[], results=[], regions=[Region(blocks=[block])])
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
    region_a = Region(blocks=[entry, cont])

    c = Value(name="c", type=t)
    region_b = Region(
        blocks=[
            Block(arguments=[c], ops=[Op(operands=[c], results=[])]),
        ]
    )

    op = Op(operands=[], results=[], regions=[region_a, region_b])
    expected = textwrap.dedent(
        """\
        generic() {
        ^entry(%a: generic<bits: 32>, %b: generic):
          %0: generic = generic(%a: generic<bits: 32>, %b: generic)
        ^cont:
          generic(%a: generic<bits: 32>) attrs={flag: True}
        }
        {
        ^bb0(%c: generic):
          generic(%c: generic)
        }"""
    )
    assert_print_and_roundtrip(op, expected)
