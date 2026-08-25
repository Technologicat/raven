"""Adopted third-party code, maintained as Raven's own. See `README.md` in this directory.

A regular package rather than a namespace directory, so that a `tests/` subpackage inside an adopted
package resolves. Without this file, pytest's walk up from such a test stops at the adopted package and
roots there, importing it as top-level — and its `from ...common.gui import ...` then reaches above the
root. Subdirectories here need no `__init__.py` of their own; those that lack one are found as namespace
portions, as they were before.
"""
