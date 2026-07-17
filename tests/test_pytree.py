from ayla import pytree


def test_flatten():
    tree = ((1, 2), 3, (4, (5, 6)))
    expected = [1, 2, 3, 4, 5, 6]
    assert list(pytree.flatten(tree)) == expected


def test_map():
    tree = ((1, 2), 3, (4, (5,)))
    expected = ((2, 3), 4, (5, (6,)))
    assert pytree.map(lambda x: x + 1, tree) == expected
