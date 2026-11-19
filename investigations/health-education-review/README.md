# The health-education review: how its data was processed

The processing record of the health-education review, the third study in a series after biology and geography:
sustainability education in health education, at primary and secondary schools and in teacher education. Everything
here is what the methods section is written from, stage by stage, with the counts, the tools, the decisions and what is
still open. Opened 2026-11-17, when the questions about it first had to be answered from logs and memory.

The data stays out of git, in `00_stuff/rawdata/biology_geography_health_education/202610/` on the maintainer's machines
(*Where the files are*, below). The clustering method has its own bundle, `../llm-thematic-grouping/`.

## The stages

| Stage | Records | Tool | What happened |
|---|---|---|---|
| 1. Search | 119 | Scopus | Documents published 2000–2024, searched in English between 18 May and 13 June 2025, with a hierarchical keyword strategy (single terms up to Boolean combinations); from the manuscript |
| 2. Clean | 119 → 118 | not recorded | The record template, included by accident, removed; two records with improvised citation keys given proper ones |
| 3. Deduplicate | 118 → 117 | `raven-deduplicate` | One paper exported twice under the same key |
| 4. Fill abstracts | 117 | `raven-fillbib`, inside the screening run (`--fill`) | From Crossref, by DOI: 2 truncated abstracts completed; all 117 have an abstract; 5 differ from Crossref's text |
| 5. Screen | 117 → 73 removed, 8 to a person, 36 kept | `raven-scopebib` | LLM-assisted screening against a written definition of scope |
| 6. Hand decisions | → **40** | `raven-scopebib finish` | The authors' decisions over the tool's |
| 7. Cluster | 40 in 7 themes | `raven-importer`, `clustering_method = "llm"` | One LLM run, checked against every abstract by the authors |

The 117 is the figure the manuscript gives after its screening phase (95 articles, 20 book chapters, 2 books).

### 2–3. Cleaning and deduplication

No log of the cleaning step survives; what it did is read from the citation keys of the files before and after it. The
raw file held one record that is not a study: the template the records were filled in from, placeholders only,
included by accident and removed at cleaning; not reported in the paper (the authors, 2026-11-17). Two records had
improvised keys for want of a DOI: Cooke et al. (2011), a search result with no DOI, whose record is in QUT's
repository (<https://eprints.qut.edu.au/41660/>) with no fulltext found online; and Øvrebø (2015), keyed by its Scopus
record's URL. The
deduplication audit beside the deduplicated file was regenerated on 2026-11-17: the one present until then said nothing
was merged and did not describe the file beside it, while a fresh run of `raven-deduplicate` on the clean file gives the
deduplicated file's 117, with the merge recorded.

### 5. Screening

`raven-scopebib` run e (`scopebib-2026-10-29e/`), on 2026-11-02, after runs a–d on 2026-10-28/29 that settled the
definition. One model for every stage: Qwen 3.8 27B (`q4_k_xl`) on LM Studio. The run's own record is its `run.toml`
(seed 42, the instruments of each stage) and its two logs.

- **The definition**: the scope, *"studies on sustainability, sustainable development and education for sustainable
  development (ESD) in health education"*, with notes on what belongs, and these grounds for removal, written from the
  authors' first two manual passes (2026-10-28 and -29): not health education; sustainability not framed by sustainable
  development or the SDGs; education for health professionals; preschool or early childhood education; not education at
  an educational institution; not education at all.
- **The stages**: a judge on every record; a reviewer over the judge's removals; fields extracted from the keeps, and
  rules over them, which remove what a kept record states in its own words and hold for a person what only looks
  doubtful.
- **The outcome**: 73 removed, 8 sent to a person, 36 kept.
- **The authors' earlier decisions** (81, from runs a–d) were carried into the run; 10 of them contradicted it, and those
  were listed first for the authors to read.

### 6. Hand decisions

At `finish` (2026-11-02), the authors' decisions overrode the run in 10 places: 6 records the judge had removed were kept,
and 4 the rules had kept were dropped. Of the 8 sent to a person, 2 were kept and 6 dropped (derived: the hand-decided
sheet holds 18 decisions, 8 keeps and 10 drops, and the run's 36 keeps less 4 plus 6 make 38). So 40 studies.

### 7. Clustering

The seven themes of one run of the importer's LLM grouping, over the 40, with the topic line as its aim; decided by the
authors on 2026-11-17, after comparing it with a four-theme run without an aim and reading every abstract. The themes,
what produced them, and how this differs from the multi-run method in that bundle are in
`../llm-thematic-grouping/README.md`, *The final clustering*. The study table is `klusterit-taulukko.md` beside the run.

Clustering by embedding was tried first and found no cluster in the 40, in high dimensions or in 2D
(`../../briefs/review-automation-brief.md`, item 10).

### Checking the manuscript's reference list

`raven-checkrefs` on `04b-final.bib` against the manuscript of 2026-11-17: **all 40 studies are in its reference list**,
35 matched by DOI and 5 by title, with no near-miss to check by hand. Its 77 other entries are the remaining candidates of
the 117, which the list still carries, and six numbered section headings the parser took for entries. Whether an earlier
check was run, against the authors' list of 2026-10-26, is not recorded; this one stands in for it, on the final set. The
audit is not kept: it changes nothing, and the command repeats it.

### Corrections to the final bibliography

Two, in `04b-final.bib` only, recorded in `../llm-thematic-grouping/README.md` under *The final clustering*: a year, by
the record's DOI against Crossref (the other 37 records with a DOI agree), and an author's spelling. The files before it
keep the export's values, by decision.

## For the methods section (decided 2026-11-17)

- **The manuscript's current methods text is the geography study's, kept as a template**, so this record is what the new
  section is written from.
- **The screening is described as LLM-assisted, with human decisions**, and the selection flowchart gains its numbers.
- **The abstract fill from Crossref is reported.**
- **The embedding pipeline's description goes**, replaced by the LLM grouping of the final set.
- **A word cloud of the final 40**, and possibly one of the full 117 beside it for comparison.
- **Two-author citations take "&"** throughout the paper.
- **The same caveat about PRISMA as the AOKK review's** (the maintainer, 2026-11-19): screening and eligibility
  as one step, with full texts consulted as needed, and criteria refined during it, so exploratory rather than
  confirmatory. Written out in `METHODS-DRAFT.md`, *How the process differs from PRISMA*.
- **The run-to-run variation is reported** (2026-11-19), with the caveat that it was measured on the development
  prompt.

## Not part of this study

**`scopebib-2026-11-02-qwen3.6/`**, a run of the same screening on Qwen 3.6 35B-A3B. It was a model comparison for the
AOKK review's choice of model, which used this study's hand decisions as its labels, and was asked for to check the
maintainer's hunch about how fast the Qwen family's capability progresses (2026-11-02). Its score is in
`../aokk-corpus-scope/README.md`: against the hand decisions, the 27B agreed outright on 63, the 3.6 on 55.

## Open

The search is reported as the manuscript has it, which is all that was recorded of it: not a full PRISMA report, and
not needing one (the authors, 2026-11-17).

- **Two more exclusions, not yet identified** (2026-11-19): the coauthor excluded two studies on closer reading, and
  which two is not yet known. When it is, every count downstream of 40 changes: the flowchart, Table 5, the cluster
  table `klusterit-taulukko.md`, `04b-final.bib`, `exclusion-grounds.md` (their reasons) and `METHODS-DRAFT.md`.
  The themes stay as they are, already in the manuscript, so the clustering is not rerun (the authors, 2026-11-19).
  The manuscript snapshot of 2026-11-17 has Table 5 but predates these two; a later snapshot's Table 5, compared
  against `klusterit-taulukko.md`, would name them, if the coauthor has removed them there.
- **Raven 0.2.11 is released before the paper comes out**, so that the paper can cite a released version
  (the maintainer, 2026-11-19). Both runs used 0.2.11 in development; the citation can name the release, with the
  commits (screening `run.toml`, grouping `69da540e`) for exactness.
- **The free-form methods description**, from this record, for the authors to write the manuscript's section from; in
  English. In `METHODS-DRAFT.md` in this bundle (agreed 2026-11-18); next, from 2026-11-19.
- **The manuscript's Table 5 now holds the clusters** (2026-11-17), with the Visualizer's 0-based numbers (#0 to #6),
  checked against `klusterit-taulukko.md`: the same 40 studies in the same clusters and order. Five two-author
  citations still read "and" where the paper settled on "&", and cluster #1's topic may be missing a space
  ("school-basedhealth"), unless that is a line break the text extraction drops.
- **The fulltexts**: the set for all three reviews arrived on 2026-11-17, one folder with subfolders, in the maintainer's
  downloads (`~/Downloads/papers/biology_and_geography_education/`). It may hold papers added later that are in no
  dataset. Next, on another day: match the study lists of the three reviews against it. This is what the per-country
  table (below) and the content analysis of the review-automation brief (items 8 and 11) read from.

- **The word clouds** (above).
- **The countries and educational levels** of the 40 (the manuscript's per-country table): by hand, or by the fulltext
  tool planned in the review-automation brief (item 8).

## Where the files are

In `00_stuff/rawdata/biology_geography_health_education/202610/`, gitignored:

| File | What it is |
|---|---|
| `health_education.bib` | The raw bibliography, 119 records |
| `health_education_clean.bib` | After cleaning, 118 |
| `health_education_deduped.bib`, `..._audit.tsv` | After deduplication, 117, with the audit of the merge |
| `aim-topic.txt` | The topic line, the clustering's aim |
| `scopebib-2026-10-29e/` | The screening run: `definition.scope.toml`, `run.toml`, the logs, the sheets, `04b-final.bib` (the 40) |
| `scopebib-2026-10-29e/visualizer-llm-topic.pickle`, `import-llm-topic.log` | The chosen clustering, as a Visualizer dataset, and its import log |
| `scopebib-2026-10-29e/klusterit.md`, `klusterit-taulukko.md` | The clusters as exported from the Visualizer, and the study table made from them |
| `scopebib-2026-10-29e/llm-grouping/` | The multi-run grouping experiments (`../llm-thematic-grouping/`) |
