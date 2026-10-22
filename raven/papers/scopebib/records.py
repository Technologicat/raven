"""A record as the scope judge sees it, and the two measurements that decide when a judge must look further.

  - **Is the title enough to judge from?** A judge asked about a title that names nothing can be confidently
    wrong, and its own confidence cannot be relied on to say so. So whether a title is too thin is
    measured from the title, by counting its content words (`title_measures`), and a thin title sends its
    record to a second look whatever the judge said.
  - **Is the abstract complete?** A publisher's preview that breaks off mid-sentence is weaker evidence than
    a whole abstract (`looks_truncated`).

The record itself (`Record`, `load_records`) carries what the stages show a model and a reader: title,
abstract and venue as readable text, and a link.
"""

__all__ = ["Record", "load_records",
           "MIN_CONTENT_WORDS", "MIN_CONTENT_DENSITY",
           "informative_words", "content_word_counts", "is_uninformative", "TitleMeasure", "title_measures",
           "looks_truncated", "describe_abstract"]

import dataclasses
import functools
import pathlib
import re
from collections.abc import Container, Iterable, Sequence
from typing import NamedTuple, Protocol

from .. import bibtex
from .. import utils as papers_utils


@dataclasses.dataclass(frozen=True)
class Record:
    """One record of a `.bib`, as the stages see it.

    `key`: the citekey.
    `title`, `abstract`, `venue`: readable text — LaTeX markup resolved, whitespace collapsed — or `""`
                                  where the record has none.
    `link`: where to read the paper, by DOI, else by `url`, else `""` (see `raven.papers.utils.paper_url`).
    """
    key: str
    title: str
    abstract: str
    venue: str
    link: str


def load_records(bib_path: str | pathlib.Path) -> list[Record]:
    """Every record of the `.bib` at `bib_path`, in file order."""
    library = bibtex.parse_file(pathlib.Path(bib_path), split_names=False)
    return [Record(key=entry.key,
                   title=bibtex.readable_text(bibtex.entry_field(entry, "title")),
                   abstract=bibtex.readable_text(bibtex.entry_field(entry, "abstract")),
                   venue=bibtex.entry_venue(entry),
                   link=papers_utils.paper_url(bibtex.entry_field(entry, "doi") or None,
                                               bibtex.entry_field(entry, "url") or None))
            for entry in library.entries]


# --------------------------------------------------------------------------------
# Is the title enough to judge from?

# When a title is too thin to judge from. Two tests, because a title can fail in two unrelated ways and
# no single number catches both.
#
# **A count catches the short and empty.** "Rethinking the paradigm" names nothing, and neither does a
# heading or a bare proper noun.
#
# **A density catches the long and padded**, which a count cannot: academic prose pads indefinitely, so
# "Tackling the possible advantages and potential challenges of a technical understanding in the present
# context" runs to fifteen words and three content words. Raising the count bound until it caught that
# was the wrong repair — measured across five corpora it flagged a quarter of one of them, because a
# *short and precise* title scores the same three or four. "Distilling the Knowledge in a Neural
# Network", "Unfitted finite element methods" and "On the instability of an axially moving elastic
# plate" all score 4, and all say exactly what they are about. Count does not separate the populations;
# the share of the title that is load-bearing does.
#
# Calibrated on five corpora in unrelated domains — education and AI, computational mechanics, applied
# mechanics, arXiv AI, hydrogen production — where these bounds flag 0.7% to 4.9%, and on a set of
# hand-written filler and genuine titles that it separates completely. Note this is *not* a classifier
# for vagueness and cannot be one: it is the backstop for the single case the model's own confidence
# cannot cover, which is being sure about an input that says nothing.
#
# Density applies at any length. A short title cannot fail it at these bounds: with at least
# `MIN_CONTENT_WORDS` content words in under eight words, its density is at least 3/7, above the bound.
# Recalibrating either bound means checking short titles again, since a raised density bound would start
# failing short precise ones.
MIN_CONTENT_WORDS = 3
MIN_CONTENT_DENSITY = 0.40


class _Pipeline(Protocol):
    """What the content-word count needs of an NLP pipeline: `raven.client.mayberemote.NLP` has it."""
    def analyze(self, text: list[str]) -> list: ...


@functools.lru_cache(maxsize=None)
def _default_stopwords() -> frozenset[str]:
    """spaCy's English stopwords and the academic ones, as the Visualizer's keyword detection uses them."""
    # Imported here, on first use: `nlptools` brings the whole ML stack with it, torch and spaCy included,
    # which loading records does not need.
    from ...common import nlptools
    from ...common.text import stopwords
    return frozenset(nlptools.default_stopwords) | stopwords.academic_stopwords


def informative_words(title: str) -> int:
    """How many words of the title carry letters — digits and punctuation say nothing about a subject."""
    return len([word for word in re.split(r"[\s\-_/]+", title) if any(c.isalpha() for c in word)])


def content_word_counts(titles: Sequence[str], nlp: _Pipeline, *,
                        stopwords: Container[str] | None = None) -> list[int]:
    """How many words of each title are neither function words nor academic filler. One `nlp` call for all.

    `nlp`: anything with an `analyze(texts) -> docs` method, normally `raven.client.mayberemote.NLP`.
    `stopwords`: lowercase words that do not count. `None` (default) is spaCy's English stopwords together
                 with `raven.common.text.stopwords.academic_stopwords`.
    """
    # Counted on *lemmas*, which is the part that cannot be skipped: the stopword lists enumerate base
    # forms, so a surface-form match lets any inflection through. Both the lemma and the surface form are
    # checked, since a list holding an irregular form directly is still a hit.
    if not titles:
        return []
    if stopwords is None:
        stopwords = _default_stopwords()
    return [len([token for token in doc
                 if token.is_alpha
                 and token.lemma_.lower() not in stopwords
                 and token.text.lower() not in stopwords])
            for doc in nlp.analyze(list(titles))]


def is_uninformative(content: int, informative: int) -> bool:
    """Whether a title with `content` content words of `informative` words is too thin to judge from."""
    return content < MIN_CONTENT_WORDS or content / informative < MIN_CONTENT_DENSITY


class TitleMeasure(NamedTuple):
    """What `title_measures` found about one title."""
    content: int
    informative: int
    uninformative: bool


def title_measures(records: Iterable[Record], nlp: _Pipeline, *,
                   stopwords: Container[str] | None = None) -> dict[str, TitleMeasure]:
    """`TitleMeasure` for each record's title, by citekey. One `nlp` call for the distinct titles.

    `nlp`, `stopwords`: as `content_word_counts`.
    """
    records = list(records)
    distinct = sorted({record.title for record in records})
    counts = dict(zip(distinct, content_word_counts(distinct, nlp, stopwords=stopwords)))
    out = {}
    for record in records:
        content, informative = counts[record.title], informative_words(record.title)
        out[record.key] = TitleMeasure(content, informative, is_uninformative(content, informative))
    return out


# --------------------------------------------------------------------------------
# Is the abstract complete?

def looks_truncated(abstract: str) -> bool:
    """Whether an abstract visibly breaks off: an ellipsis at the very end, as a publisher's teaser has.

    It cannot see a publisher who truncates silently, and no text-level rule can.
    """
    # Keyed on an ellipsis *at the end*, which is the only place one settles anything. Mid-text an ellipsis
    # is ordinary rhetoric — an elided quotation, a trailing "and so on" — so its presence there says
    # nothing about whether the text is complete.
    #
    # Length is deliberately not part of it: some abstracts are simply short, and measured on the
    # prototype's corpus the bound never fired on its own. And "ends without terminal punctuation" was
    # tried as a signal for silent truncation and rejected: it selects complete records ending in a URL, a
    # DOI or a keyword list. That gap is the judge's to report.
    return bool(abstract) and bool(re.search(r"(\.\.\.|…)\s*$", abstract.strip()))


def describe_abstract(record: Record) -> str:
    """What kind of evidence the record's abstract is: `"none"`, `"teaser"` or `"full"`."""
    if not record.abstract:
        return "none"
    return "teaser" if looks_truncated(record.abstract) else "full"
