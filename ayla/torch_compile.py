"""Run Ayla's post-grad FX region rewrites under ``torch.compile``."""
from __future__ import annotations

import torch._inductor.config as inductor_config
from torch._inductor.pattern_matcher import PatternMatcherPass

from .gemm_collection.fx_gemm import register_gemm_fx_rules
from .gemm_collection.muse_swiglu import register_muse_swiglu


_POST_GRAD_PATTERNS = PatternMatcherPass(pass_name="ayla")
register_muse_swiglu(_POST_GRAD_PATTERNS)
register_gemm_fx_rules(_POST_GRAD_PATTERNS)
_POST_GRAD_PASS = _POST_GRAD_PATTERNS.apply
_displaced_post_grad_pass = None
_installed_post_grad_pass = None


def _post_grad_passes(configured) -> tuple:
    if configured is None:
        return ()
    if isinstance(configured, (list, tuple)):
        return tuple(configured)
    return (configured,)


def install(*, forward_pass: bool = False) -> None:
    """Install FX rewrites, or capture a complete static graph with ``forward_pass=True``."""
    global _displaced_post_grad_pass, _installed_post_grad_pass
    from .forward_compile import capture_forward_graph

    requested = capture_forward_graph if forward_pass else _POST_GRAD_PASS
    if _installed_post_grad_pass is not None:
        if _installed_post_grad_pass is not requested:
            raise ValueError("uninstall Ayla before switching capture modes")
        return
    existing = inductor_config.post_grad_custom_post_pass
    existing_passes = _post_grad_passes(existing)
    _displaced_post_grad_pass = existing
    _installed_post_grad_pass = requested
    inductor_config.post_grad_custom_post_pass = [
        *existing_passes, requested,
    ]


def uninstall() -> None:
    """Restore the post-grad passes displaced by :func:`install`."""
    global _installed_post_grad_pass
    current = inductor_config.post_grad_custom_post_pass
    if _installed_post_grad_pass is not None and _installed_post_grad_pass in _post_grad_passes(current):
        inductor_config.post_grad_custom_post_pass = _displaced_post_grad_pass
    _installed_post_grad_pass = None
