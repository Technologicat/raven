"""The definition's rules over a kept record's extracted fields: remove it, hold it for a person, or keep it.

Rules are applied when the reports are written rather than asked of a model, so changing one asks nothing
again. The first rule that matches decides; a rule matches when every field it names has one of the values
it lists. A record no rule matches is kept.

One safeguard is the tool's own: a `remove` resting on a value its field says must be quoted, whose quote
is not found in the record's text, becomes a `hold`. The removal is then only as good as an answer that cannot be checked,
and a person should look.
"""

__all__ = ["KEEP", "Decision", "decide"]

from typing import NamedTuple

from . import definition as scope_definition

#: The action for a record no rule matches.
KEEP = "keep"


class Decision(NamedTuple):
    """What the rules made of one record.

    `action`: `"remove"`, `"hold"` or `KEEP`.
    `label`: the matching rule's label, or `None` where none matched.
    `note`: why the action differs from the rule's, or `""`.
    """
    action: str
    label: str | None
    note: str


def decide(extraction: dict, definition: scope_definition.Definition) -> Decision:
    """Apply `definition`'s rules to one extraction, as `raven.papers.scopebib.extract.normalize` returns it."""
    fields = extraction["fields"]
    must_quote = {field.name: field.quote for field in definition.fields}
    for rule in definition.rules:
        if all(fields.get(name) in values for name, values in rule.when.items()):
            unchecked = sorted(name for name in rule.when
                               if fields.get(name) in must_quote.get(name, ())
                               and not extraction["quotes"].get(name, {}).get("found"))
            if rule.action == "remove" and unchecked:
                return Decision("hold", rule.label, f"held rather than removed: the quote for "
                                                    f"{', '.join(unchecked)} is not in the record")
            return Decision(rule.action, rule.label, "")
    return Decision(KEEP, None, "")
