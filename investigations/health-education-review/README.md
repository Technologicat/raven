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
| 2. Clean | 119 → 118 | not recorded | A hand-made stub removed (placeholder title, placeholder key); two records with improvised citation keys given proper ones |
| 3. Deduplicate | 118 → 117 | `raven-deduplicate` | One paper exported twice under the same key |
| 4. Fill abstracts | 117 | `raven-fillbib`, inside the screening run (`--fill`) | From Crossref, by DOI: 2 truncated abstracts completed; all 117 have an abstract; 5 differ from Crossref's text |
| 5. Screen | 117 → 73 removed, 8 to a person, 36 kept | `raven-scopebib` | LLM-assisted screening against a written definition of scope |
| 6. Hand decisions | → **40** | `raven-scopebib finish` | The authors' decisions over the tool's |
| 7. Cluster | 40 in 7 themes | `raven-importer`, `clustering_method = "llm"` | One LLM run, checked against every abstract by the authors |

The 117 is the figure the manuscript gives after its screening phase (95 articles, 20 book chapters, 2 books).

### 2–3. Cleaning and deduplication

No log of the cleaning step survives; what it did is read from the citation keys of the files before and after it. The
raw file is not a pure export. Two records carried the placeholder key `doi-numero`: the stub the cleaning removed,
whose title is a placeholder too, and Cooke et al. (2011), which reached the final 40; both look added by hand (*Open*,
below). A third, Øvrebø (2015), was keyed by its Scopus record's URL, having no DOI, which suggests it came from the
export. The
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

## Not part of this study

**`scopebib-2026-11-02-qwen3.6/`**, a run of the same screening on Qwen 3.6 35B-A3B. It was a model comparison for the
AOKK review's choice of model, which used this study's hand decisions as its labels, and was asked for to check the
maintainer's hunch about how fast the Qwen family's capability progresses (2026-11-02). Its score is in
`../aokk-corpus-scope/README.md`: against the hand decisions, the 27B agreed outright on 63, the 3.6 on 55.

## Open

- **The search**: the query itself, and the record counts of the manuscript's identification phase.
- **The records added by hand**: where they came from, and on what grounds they were added. One of them reached the
  final 40: Cooke et al. (2011), which has no DOI. Whether Øvrebø (2015) came from the export, as its Scopus key
  suggests.
- **`raven-checkrefs` against the study list**, decided on 2026-11-02 to catch slips like those found in the two earlier
  papers' lists: whether it was run.
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
