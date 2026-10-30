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
  - Themes only, since 2026-10-30's second batch; the first evening's runs asked for sub-themes too (the
    script at `66c6cea8`). `--guidance FILE` adds an instruction of the user's own, and a reply with slips
    gets a repair turn naming each one, up to two, built on the original prompt so the backend's prefix cache
    covers it.
- **`compare_groupings.py`** — how far the runs agree, pairwise, at the theme and the sub-theme level: the
  omega index (Collins & Dent 1988) over every placement, and the adjusted Rand index on primary placements.
  The omega index is the ARI generalized to groupings where a study sits in several groups, and equals it
  exactly on disjoint ones (checked on random partitions, 2026-10-30).
  Then, per condition, a per-study view from how often each pair of studies shares a theme across the runs:
  the stable cores (together in every run), the studies that drift most, and with `--plot DIR` a heatmap
  of that co-membership ordered by average linkage. `--bib` names the studies by citation.

## Running it

One condition per combination of aim and guidance file, the same seeds in every condition. The batch queued
for the night of 2026-10-30, themes only, with the repair turn:

```bash
D=00_stuff/rawdata/biology_geography_health_education/202610
B=$D/scopebib-2026-10-28d/04b-final.bib
G=$D/guidance-programme-continuity.txt
for args in "--aim $D/aim-topic.txt --guidance $G" "--guidance $G" "--aim $D/aim-topic.txt"; do
    python investigations/llm-thematic-grouping/group_studies.py $B $args --runs 10 \
        --declare "reasoning effort xhigh, thinking budget 12k"
done
python investigations/llm-thematic-grouping/compare_groupings.py $D/scopebib-2026-10-28d/llm-grouping/*/grouping-*.jsonl \
    --bib $B --plot $D/scopebib-2026-10-28d/llm-grouping
```

The third condition is the second's control: the topic line without the instruction about programme
continuity. The comparison then also reads the first evening's ledgers, made with sub-themes and no repair
turn; read those against the new ones at the theme level only.

`--declare` records what the backend does not report — here the chat template's reasoning effort and LM
Studio's thinking budget — in the instrument's fingerprint, so runs under other settings land in another
ledger.

## Results

**Health education, run d's final set of 40, 2026-10-30.** Qwen 3.8 27B (`q4_k_xl`) on LM Studio, reasoning
effort `xhigh`, thinking budget 12k; five runs per condition, about 4 minutes a run. Outputs, with the
comparison (`comparison.txt`) and its heatmaps, in `llm-grouping/` beside the `.bib`.

- **Pilot** (the script at `c4defead`; the research questions with their file's provenance header, by
  mistake): theme-level omega 0.35, ARI 0.34; sub-theme 0.28. Five stable cores held 18 studies.
- **The three conditions**, mean omega index at the theme level (ARI within 0.02 of it throughout):

  |                    | questions | topic | no aim |
  |---|---|---|---|
  | **questions**      | 0.43 | 0.41 | 0.33 |
  | **topic**          | | 0.45 | 0.46 |
  | **no aim**         | | | 0.53 |

  Sub-themes agree at 0.27–0.36 everywhere. Stable cores: 21, 26 and 28 of the 40 studies.
- **What it suggests, from five runs a condition** (ten pairs, not independent, so a difference of about
  0.1 is a lead rather than a result):
  - **Guidance does not make the grouping more stable**; the unguided runs agree with each other most.
  - **The research questions move the grouping**, rather than only adding noise: their runs agree less
    with the unguided ones (0.33) than either condition does with itself. Read by name, their themes
    follow the questions — curriculum content, teaching and learning methods, competencies and learning
    objectives — although the prompt says not to make one theme per question. The topic line sits
    between, and adds a teacher-education theme, the line naming teacher education.
  - **The unguided themes are the most topical**: environmental education and health outcomes, community
    engagement, transdisciplinary education.
  - **Every condition has an implementation-and-sustainability-of-programmes theme**: "sustainability" in
    the sense of a programme lasting, the second sense the search word has.
  - **Placing a study twice stayed the exception**: 0 to 6 of 40 per run. A few replies left a study
    without a primary placement, placed one twice in a sub-theme, or (once) left one out.
- **The sub-theme level is unstable under every condition**, so a published table would want its sub-themes
  read and named by people, as the earlier studies' were.
