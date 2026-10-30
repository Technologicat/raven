# Grouping a review's accepted studies with an LLM

**The question**: can a model, given a review's research questions and every accepted study's title and
abstract in one prompt, group the studies into themes and sub-themes the way the hand-made cluster tables of
the earlier studies in the series do — and how stable is that grouping from run to run?

This is item 10 of `briefs/review-automation-brief.md`, where the design decisions are logged (2026-10-29
and 2026-10-30). For the health-education study it replaces the Visualizer's HDBSCAN clusters, which found
no cluster at all in a set of forty.

## Scripts

- **`group_studies.py`** — runs the grouping several times over one `.bib`, shuffling the records per run
  (the run's number is the seed), and writes per run a table shaped like the published ones, a table of
  every placement with the model's reason, and a Raven-visualizer dataset laid out by hand: one circle per
  theme, its studies around the perimeter in sub-theme order, the circles' centres on a maximin Latin
  square as Librarian's attachment piles are. In the dataset each study is at its primary placement only,
  the Visualizer showing one point per record; the tables keep every placement.
  - Resumable: each run is one item of a `raven.librarian.sweep`, and every model call is kept whole in the
    ledger's traces; `stream.log` shows each turn as it streams.
  - Outputs go beside the `.bib` by default (`llm-grouping/`), with the private data, and not here.

## Running it

```bash
python investigations/llm-thematic-grouping/group_studies.py \
    00_stuff/rawdata/biology_geography_health_education/202610/scopebib-2026-10-28d/04b-final.bib \
    --questions 00_stuff/rawdata/biology_geography_health_education/202610/research-questions.txt \
    --runs 5 --declare "reasoning effort xhigh, thinking budget 12k"
```

`--declare` records what the backend does not report — here the chat template's reasoning effort and LM
Studio's thinking budget — in the instrument's fingerprint, so runs under other settings land in another
ledger.

## Results

None yet.
