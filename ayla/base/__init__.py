"""Dialect-agnostic compiler base: the IR data model, its textual
parser/printer, and the pass framework the compiler is built on.

The GPU dialect (``ayla.gpu_dialect``), the passes (``ayla.passes``), and the
codegen orchestrator (``ayla.ptx_codegen``) all build on top of this.
"""
