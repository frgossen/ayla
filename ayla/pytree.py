"""Tiny tuple-shaped pytree helpers."""
from collections.abc import Callable, Iterator
from typing import Any


type PyTree[T] = T | tuple[PyTree[T], ...]


def map[T, U](fn: Callable[[T], U], t: PyTree[T]) -> PyTree[U]:
    """Map `fn` over the leaves of a tuple-shaped pytree."""
    if isinstance(t, tuple):
        return tuple(map(fn, x) for x in t)
    return fn(t)


def flatten(t: Any) -> Iterator[Any]:
    """Yield the leaves of a tuple-shaped pytree in order."""
    if isinstance(t, tuple):
        for x in t:
            yield from flatten(x)
    else:
        yield t
