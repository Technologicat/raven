"""Configuration for the Raven bibliography tools (`raven.papers`).

The knobs a user might reasonably want to turn. Anything a module reads and nobody outside it should touch
— a regex, an XML namespace, a progress glyph — stays in that module, in `SCREAMING_CASE`; see
`raven-style-guide.md` under *Naming*, where the casing is what says which of the two a constant is.

Used by `papers.deduplicate`, and by the arXiv tools: `papers.search`, `papers.arxiv2bib`,
`papers.download`, `papers.httpfetch`.
"""

from .. import __version__

# --------------------------------------------------------------------------------
# Talking to arXiv

# The arXiv API endpoint. One copy: `search`, `arxiv2bib` and `download` each had their own, which is
# three places to edit if arXiv ever moves it and three chances to miss one.
arxiv_api_url = "https://export.arxiv.org/api/query"

# Minimum wall time between requests, seconds. **This is arXiv's, not ours** — their terms of use require
# waiting at least three seconds between requests, so lowering it is a violation rather than a tuning
# choice. Raise it freely.
#
# Here rather than as `RateLimiter`'s default argument, where it was: a number that binds every arXiv
# caller should be visible in one place a user can find, and `ratelimit` is generic machinery that happens
# to have been written for this.
arxiv_request_delay = 3.0

# Search results per request. arXiv's own documentation suggests keeping a single request modest; this is
# the largest value that has behaved reliably in practice.
arxiv_page_size = 200

# The most results arXiv will return for one query, however you page through it. Their limit, not ours —
# a search expecting more than this needs splitting into narrower queries.
arxiv_max_results = 30_000

# Identifiers per metadata request. The API answers an `id_list` naming many papers in one response, and
# the rate limit is per *request*, so the metadata for a whole run costs ceil(N / 100) waits instead of N.
#
# 100 rather than as many as fit: a URL is not a good place to discover your limits, since an over-long
# one fails as an opaque HTTP error partway through a batch job. This keeps the query string well inside
# any sane bound while still amortizing the three-second wait over real work.
#
# Note this speeds up metadata only. Fulltext PDFs are one request each and still cost a wait apiece.
arxiv_id_batch_size = 100

# Sent as `User-Agent` on every arXiv request. arXiv asks for a descriptive agent with a contact address,
# so that they can get in touch about a misbehaving client rather than simply blocking it.
#
# **Worth changing if you are running a modified Raven**, or running it at volume: the address below
# reaches this project's maintainer, who cannot answer for what your copy did.
http_user_agent = (f"raven-papers/{__version__} "
                   f"(+https://github.com/Technologicat/raven; mailto:juha.jeronen@jamk.fi)")

# --------------------------------------------------------------------------------
# Deciding that two records are the same paper

# DOI registrants that host copies rather than publish versions of record: the preprint servers, and the
# general-purpose repositories a paper gets deposited in beside its journal. Used only to decide which copy
# of a merged pair is the base — a record whose DOI starts with one of these loses to a record whose DOI
# does not, whatever else is true of them.
#
# Deliberately a short list of registrant prefixes rather than a guess from the venue name. Demoting a real
# journal record by mistake would put the wrong DOI on a merged entry, which is the one outcome nothing
# downstream can detect. Add a prefix here if your corpus draws on a repository this does not name.
preprint_doi_prefixes = ("10.48550/",   # arXiv
                         "10.5281/",    # Zenodo
                         "10.2139/",    # SSRN
                         "10.1101/",    # bioRxiv, medRxiv
                         "10.21203/",   # Research Square
                         "10.31234/",   # PsyArXiv (OSF)
                         "10.31219/",   # OSF Preprints
                         "10.20944/")   # Preprints.org

# Titles that name a genre rather than a work. Two unrelated papers can carry one and often do — `Editorial`
# alone put a 2022 paper and a 2024 one in the same cluster — so a title normalizing to one of these carries
# no evidence of its own, and `deduplicate._title_edge_holds` requires corroboration from elsewhere.
#
# Compared against the *normalized* title, so entries here are lowercase with no spaces or punctuation.
# **This list is English.** A corpus in another language wants its own genre labels adding — `paakirjoitus`,
# `kirja-arvostelu` — and the tool cannot infer them.
#
# A curated list rather than a length threshold, which was tried first and is the wrong axis: it caught
# nothing on a 6934-record corpus that this list did not already catch, while rejecting `Reportronic` — a
# real and thoroughly distinctive title — for being eleven characters long. Distinctiveness is not length.
# `Generative AI`, which is thirteen, is the better cautionary example: longer than the title the rule
# rejected, and far likelier to head two different editorials.
generic_titles = frozenset(["editorial", "editorials", "introduction", "preface", "foreword",
                            "afterword", "correction", "corrections", "corrigendum", "erratum",
                            "errata", "retraction", "retractionnotice", "comment", "commentary",
                            "reply", "response", "discussion", "letter", "letters",
                            "lettertotheeditor", "bookreview", "bookreviews", "review", "reviews",
                            "bookprofile", "frontmatter", "backmatter", "tableofcontents", "contents",
                            "index", "abstract", "abstracts", "acknowledgements", "acknowledgments",
                            "authorindex", "subjectindex", "titlepage", "coverimage", "conclusion",
                            "conclusions", "summary", "references", "bibliography", "glossary",
                            "appendix", "notes", "news", "obituary", "announcement", "announcements",
                            "callforpapers", "aboutthisjournal", "aboutthecontributors",
                            "issueinformation", "masthead", "untitled", "notitle", "na"])

# How far two records' years may sit apart while still being one paper. One year, because a preprint and its
# published version routinely straddle a New Year, and two databases can disagree about which year an
# online-first article belongs to. Two is a different paper.
max_year_drift = 1

# Fields that say *which* item a record is, beyond its title. Consulted where the title cannot carry a
# merge on its own — a generic title like `Book Review`, or a record naming no author — so that a match on
# one of those is admitted only when nothing here contradicts it. The same person may well write two book
# reviews in a year, and those differ by DOI, by page range, or by issue.
#
# **Only a positive disagreement counts.** A field absent from either record says nothing, and requiring
# agreement would refuse nearly every genuine pair, since two databases export different subsets of the
# same record.
identifying_fields = ("doi", "pages", "volume", "number")

# Where `raven-fixbib` puts a rights notice it moves out of an abstract, and so where `deduplicate` looks
# for one. Change both together, or the notice stops being found. See `bibtex.relocate_rights_notices`.
rights_field = "copyright"

# --------------------------------------------------------------------------------
# The LLM judge (opt-in, `--judge`)

# How alike two normalized titles must be before the judge is asked about them.
#
# **The scale is `difflib.SequenceMatcher.ratio()`**: `2M/T`, where `M` is the number of matching characters
# and `T` the combined length of the two strings. Pure character overlap, 0 to 1 — no semantics, no
# embeddings, no notion of what the words mean. Two titles about the same subject in different words score
# low; two spellings of one title score high, which is the question here.
#
# Chosen to catch the near-misses an exact key cannot — a subtitle one database kept and another dropped, a
# word order fixed between the preprint and the published version — without asking about every pair of
# papers on a common subject, which in a topical corpus is most of them.
title_similarity = 0.86

# Pairs per model call. Smaller than the 40 filenames `investigations/agent-batch-classification/` used,
# because each item here carries two full records rather than one filename.
judge_batch = 12

# What the judge is asked. Yours to edit — the shape it has to keep is the JSON array of `{i, same, why}`
# objects that `deduplicate.judge_batch` reads back, and the numbered items it is given.
#
# The last paragraph is not decoration. Left to itself a model will claim to recognize a DOI or an
# identifier from memory and answer confidently from the claim; saying so plainly is what turns that into
# an admission that the records do not settle it. Same finding as
# `investigations/agent-batch-classification/`, where "an identifier is not a description" turned confident
# fabrications into correct low-confidence answers.
judge_instructions = """\
You are deduplicating a bibliography assembled from several literature databases. The same paper is \
often exported by each database that indexes it, with the fields spelled differently, so two records that \
look slightly different are frequently one paper.

For each numbered item you are given two bibliography records. Decide whether they describe THE SAME \
WORK.

Treat as the same work:
  - the same paper exported by two databases, with different capitalization, punctuation, markup or \
field coverage
  - a preprint and the published version of the same paper
  - two versions of the same book chapter or living reference work entry
  - the same paper with a subtitle present in one record and absent in the other

Treat as different works:
  - different papers by the same authors on a related topic
  - a paper and its own correction, erratum, comment or editorial
  - different chapters of the same book, or different papers in the same proceedings
  - a conference paper and a substantially different journal article, where the titles genuinely differ \
in what they claim

For each item, answer:
  "i"      the item's number, copied exactly
  "same"   true if the two records describe the same work, false otherwise
  "why"    at most fifteen words, the evidence you used

IMPORTANT: judge only from what the records actually say. Do not claim to recognize a DOI, an identifier \
or a paper from memory; you cannot, and a guess dressed as recognition is worse than saying the records \
do not settle it. If the two records do not give you enough to tell, answer false — leaving two records \
that are one paper is recoverable, and merging two different papers is not.

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

Items:
{items}
"""

# --------------------------------------------------------------------------------
# The audit trail

# How much of a field value an audit row carries. Long enough to recognize a value and see how two of them
# differ; short enough that a row holding two abstracts is still a row. The input file is where anyone reads
# one in full.
audit_value_chars = 300
