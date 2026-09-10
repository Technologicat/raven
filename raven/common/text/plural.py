"""Agreeing a noun with a count the code already has.

Where a number goes into a message, the noun beside it has a correct form available for free, and
`1 file(s)` is the tell of a line nobody re-read. This is the one-liner that idiom always is, factored
out because Raven writes it often enough — GUI captions and log lines alike, a log line being read by a
person too.

Only for a count that is *known*. A label written before anything has been counted — `"Attach file(s)"`
on a button — is telegraphic style rather than laziness, and stays as it is; where there is room, it is
better still to say `"Attach one or more files"`.
"""

__all__ = ["plural_s"]


def plural_s(count: int) -> str:
    """Return the regular English plural suffix for `count`: `""` for exactly one, `"s"` otherwise.

    Written beside the noun it agrees with, inside the same f-string:

        f"{n} node{plural_s(n)} deleted"    ->    "1 node deleted"  /  "3 nodes deleted"

    Zero takes the plural, as English does — "0 nodes deleted".

    Only the regular form. A noun that pluralizes some other way ("one entry" / "two entries", "index" /
    "indices") wants both spellings written out where it is used; and a language with more than two forms
    wants something else entirely, which is the reason this lives in one place rather than smeared across
    every f-string that needs it.
    """
    return "" if count == 1 else "s"
