"""Telling which records of a literature search are about what it searched for.

The library behind `raven-scopebib`, one module per stage:

  - `records`, a record as the stages see it, and the two measurements that decide when a judge must look
    past a title: whether the title says enough to judge from, and whether the abstract is complete.
  - `definition`, the definition file: the scope question, the judge's tests, the extractor's fields and
    the rules over them, read and checked.
  - `judge`, the two-pass judge: every record from its title, then the ones that cannot be settled so from
    their abstracts, and a verdict of drop, review or keep.
  - `extract`, what a kept record says, field by field, with the record's own words for the values that
    need them.
  - `rules`, the definition's rules over those fields: remove, hold for a person, or keep.
  - `review`, a second reader over the judge's drops, making the case that each belongs; a drop it makes a
    case for goes to a person.
  - `report`, where each record ended up, and the stage-numbered files a run leaves: the audits, the list
    of records for a person to check, and the filtered `.bib`; and `finish`, applying that person's decisions.
  - `runfile`, the run file: which model each stage asks, and under which instrument.

It generalizes the prototype in `investigations/aokk-corpus-scope/`.
"""
