"""A small collection of pre-built GEMM kernels.

Each kernel lives in its own module here: a ``*_module`` builder plus
self-registration as a ``torch.ops.ayla.*`` overload at import time. Importing
this package imports them all, so the ops are ready to use:

    import ayla.gemm_collection  # registers all ops
    torch.ops.ayla.naive_matmul_4096_loop(a, b)  # returns c

Each is built for one specific shape and dtype -- the shape is a literal in its
module source -- so there is no op for a shape nobody has added.
"""

from .registry import register, register_gemm  # noqa: F401

from . import (  # noqa: F401,E402  (each import registers its op)
    auto_matmul_4096,
    auto_matmul_4096_wgmma256_wg,
    auto_matmul_4096_wgmma256_wg_sigmoid,
    auto_matmul_4096_wgmma256_wg_tma_store,
    fastcu_5_1,
    loop_4x4_matmul_4096,
    loop_8x8_matmul_4096,
    naive_matmul_4096_loop,
    pranjalssh_4096,
    pranjalssh_4096_pipelined,
    tma_matmul_4096,
)
