"""Telling which records of a literature search are about what it searched for.

The library behind `raven-scopebib`, one module per stage:

  - `records`, a record as the stages see it, and the two measurements that decide when a judge must look
    past a title: whether the title says enough to judge from, and whether the abstract is complete.

It generalizes the prototype in `investigations/aokk-corpus-scope/`.
"""
