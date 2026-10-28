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


def _post_grad_passes(configured) -> tuple:
    if configured is None:
        return ()
    if isinstance(configured, (list, tuple)):
        return tuple(configured)
    return (configured,)


def install() -> None:
    """Install Ayla's post-grad FX rewrites for ``torch.compile``."""
    global _displaced_post_grad_pass
    existing = inductor_config.post_grad_custom_post_pass
    existing_passes = _post_grad_passes(existing)
    if _POST_GRAD_PASS in existing_passes:
        return
    _displaced_post_grad_pass = existing
    inductor_config.post_grad_custom_post_pass = [
        *existing_passes, _POST_GRAD_PASS,
    ]


def uninstall() -> None:
    """Restore the post-grad passes displaced by :func:`install`."""
    current = inductor_config.post_grad_custom_post_pass
    if _POST_GRAD_PASS in _post_grad_passes(current):
        inductor_config.post_grad_custom_post_pass = _displaced_post_grad_pass
