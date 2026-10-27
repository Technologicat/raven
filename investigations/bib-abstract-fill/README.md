# Filling missing and truncated abstracts from Crossref

The probe behind `raven-fillbib` (planned): how many of a `.bib`'s missing and truncated abstracts Crossref
can supply, and whether "replace only when ours is a prefix of theirs" is a safe enough rule to apply
unattended. Run 2026-10-27 on the AOKK corpus (`00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib`,
5166 records, 4784 with a DOI). AI in higher education, mostly 2023–2026.

## Scripts

| Script | What it answers |
|---|---|
| `probe_crossref.py` | For every record with a DOI and no abstract, every record whose abstract ends in an ellipsis, and a random 150 of the rest as controls: fetch the Crossref record and classify ours against theirs (equal, ours a prefix, Crossref has none, different). Caches responses in a JSONL keyed on DOI, so changing the comparison re-asks Crossref nothing. Also tallies the markup tags Crossref's abstracts carry |

```bash
python investigations/bib-abstract-fill/probe_crossref.py \
    00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib \
    --cache investigations/bib-abstract-fill/aokk_crossref.jsonl --examples 2
```

The cache is gitignored: its DOI list is the private corpus's shape.

Access: the `User-Agent` from `raven.papers.config.http_user_agent`, which carries a `mailto`, put the probe in
Crossref's polite pool (`x-api-pool: polite-single`, limit 10 requests per 1 s). 1431 requests at one every
0.35 s, sequential, about ten minutes, no 429s.

## Results

**Truncated abstracts: 432, nearly all recoverable.** 417 are SAGE, cut at about 250 characters and ending in
`...`, arriving through ProQuest. 364 passed the strict prefix test as written. Of the rest:

- 32 are **structured abstracts** where our copy fuses section headings into the text (`…surgery.Aim: This
  review…`) while Crossref's carries them as `<jats:sec><jats:title>`. They pass on a letters-only
  comparison. Fix: render a section title inline as `Title: ` when normalizing, and compare
  whitespace-insensitively around it.
- 31 **diverge at about 210 characters, at the words "Plain language summary"**. SAGE articles carry a plain
  language summary as a second abstract, and the ProQuest field holds the first ~210 characters of the
  abstract, then the summary's heading, then the summary's start, cut again. Crossref's copy has the abstract
  only. Fix: cut ours at the `Plain language summary` marker before the prefix test. Ours is then a prefix.
- 5 have no abstract in Crossref.

So with those two fixes, about 426 of 432 repairable (98%).

**Missing abstracts: 849, of which Crossref has 175 (21%).** The yield is entirely a matter of publisher and
record type:

| Publisher (as Crossref names it) | records | Crossref has an abstract |
|---|---|---|
| Springer Science and Business Media LLC (journals) | 422 | 170 |
| Springer Nature Switzerland (proceedings, book chapters) | 240 | 4 |
| Springer Nature Singapore (proceedings, book chapters) | 116 | 0 |
| SAGE Publications | 47 | 0 |
| Elsevier BV | 11 | 0 |

ACM, IEEE and Elsevier deposit nothing across the controls too (42, 5 and 5 records), matching what brief 11
measured on a different corpus in August.

**Controls: the comparison is trustworthy.** Of the 150, 61 have a Crossref abstract. 32 were equal as
written and 8 more on letters only. Of the 20 "different", 17 differ only by a **copyright line our copy
carries at the end** (`© 2025 by the authors.`, which Scopus appends); with it stripped they are equal. The
last 3 are genuine version differences — one word changed (`considered a` / `considered as a`), a LaTeX
`\textless` against a literal `<`, and one whose opening sentence was rewritten. The prefix rule refuses all
three, which is the behaviour wanted: a different version goes to a person, not into the record.

## What Crossref's abstracts carry (the JATS question)

Tags seen across the 1431 responses, by count:

| tag | count | what it is |
|---|---|---|
| `jats:p` | 839 | paragraph |
| `jats:title` | 407 | either a lone "Abstract" heading or a section heading |
| `jats:sec` | 221 | a section of a structured abstract (Background, Methods, …) |
| `jats:italic`, `jats:bold` | 196, 17 | emphasis — species names, titles, terms |
| `mml:*` | ~60 in all | MathML inside `jats:inline-formula`: `mi`, `mn`, `mo`, `mrow`, `msub`, `msup`, … |
| `jats:sup` | 5 | superscript |
| `jats:tex-math`, `jats:alternatives` | 2 each | a LaTeX rendering offered beside the MathML |
| `jats:ext-link`, `jats:xref` | 4, 1 | links |

No `jats:sub` at all here — subscripts appeared only inside MathML (`mml:msub`). An education corpus says
little about how often chemistry and maths markup turns up; a hydrogen-production corpus would be the one to
measure that on.
