"""Restrict a tiled contraction's K loop before operand promotion/pipelining.

Ported from gemm-chains a590eaa's _ReductionRange. Ordinary contractions
without a reduction_range attribute keep their authored loop bounds.
"""
from ..base.ir import find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import ForOp, MatmulOp


class ReductionRange(Pass):
    def __init__(self, label):
        super().__init__(label=label)

    def run(self, module):
        for loop in find_ops_of_type(module, ForOp):
            if loop.attrs.get("label") != self.label:
                continue
            heroes = find_ops_of_type(loop, MatmulOp)
            ranges = {tuple(op.attrs["reduction_range"]) for op in heroes
                      if "reduction_range" in op.attrs}
            if not ranges:
                continue
            if len(ranges) != 1:
                raise ValueError("one K loop cannot contain different reduction ranges")
            start, end = ranges.pop()
            if (type(start) is not int or type(end) is not int
                    or not loop.lb <= start < end <= loop.ub):
                raise ValueError("reduction range must be a nonempty subset of K tiles")
            loop.lb, loop.ub = start, end
            for hero in heroes:
                hero.attrs.pop("reduction_range", None)
        return module
