"""Physical layout requirements for logical global-memory region proofs."""

from ..gpu_dialect import GmemTensorType


def has_nonoverlapping_gmem_layout(typ):
    """Prove distinct logical elements occupy distinct physical words.

    Each non-singleton axis must start beyond the complete address span of
    all faster axes. This covers dense, padded and transposed layouts.
    Other injective layouts may remain unknown; overlapping or unresolved
    strides must never justify a logical-box disjointness proof.
    """
    if (not isinstance(typ, GmemTensorType)
            or len(typ.sizes) != len(typ.strides)
            or any(type(size) is not int or size <= 0 for size in typ.sizes)
            or any(type(stride) is not int or stride < 0 for stride in typ.strides)):
        return False
    span = 1
    for stride, size in sorted((stride, size) for size, stride in zip(typ.sizes, typ.strides) if size > 1):
        if stride < span:
            return False
        span += (size - 1) * stride
    return True
