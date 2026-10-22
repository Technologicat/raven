"""Telling which records of a literature search are about what it searched for.

The library behind `raven-scopebib`, one module per stage:

  - `records`, a record as the stages see it, and the two measurements that decide when a judge must look
    past a title: whether the title says enough to judge from, and whether the abstract is complete.
  - `definition`, the definition file: the scope question, the judge's tests, the extractor's fields and
    the rules over them, read and checked.
  - `judge`, the two-pass judge: every record from its title, then the ones that cannot be settled so from
    their abstracts, and a verdict of drop, review or keep.

It generalizes the prototype in `investigations/aokk-corpus-scope/`.
"""
