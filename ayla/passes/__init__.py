"""IR transformation passes.

Each module defines one pass (a :class:`~ayla.pipeline.Pass` /
:class:`~ayla.lowering_pass.LoweringPass` subclass and/or a convenience
function). The pass framework itself lives in ``ayla.pipeline`` and
``ayla.lowering_pass``; the pipeline that composes these into codegen lives
in ``ayla.ptx_codegen``.
"""
