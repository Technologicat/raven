# Grouping a review's accepted studies with an LLM

**The question**: can a model, given a review's research questions and every accepted study's title and
abstract in one prompt, group the studies into themes and sub-themes the way the hand-made cluster tables of
the earlier studies in the series do — and how stable is that grouping from run to run?

This is item 10 of `briefs/review-automation-brief.md`, where the design decisions are logged (2026-10-29
and 2026-10-30). For the health-education study it replaces the Visualizer's HDBSCAN clusters, which found
no cluster at all in a set of forty.

**`METHOD.md` is the method as a whole**, inductive then deductive, and what is and is not known about it.

## Scripts

- **`group_studies.py`** — runs the grouping several times over one `.bib`, shuffling the records per run
  (the run's number is the seed), and writes per run a table shaped like the published ones, a table of
  every placement with the model's reason, and a Raven-visualizer dataset laid out by hand: one circle per
  theme, its studies around the perimeter in sub-theme order, the circles' centres on a maximin Latin
  square as Librarian's attachment piles are. In the dataset each study is at its primary placement only,
  the Visualizer showing one point per record; the tables keep every placement.
  - Resumable: each run is one item of a `raven.librarian.sweep`, and every model call is kept whole in the
    ledger's traces; `stream.log` shows each turn as it streams.
  - Outputs go beside the `.bib` by default (`llm-grouping/<condition>/`), with the private data, and not
    here: the ledger in the condition's folder, each run's tables and dataset in `<fingerprint>/` below it.
  - `--rebuild FINGERPRINT` writes those from the ledger's stored answers, asking no model — after the
    output format or the `.bib` changes, or with no backend at hand. The `.bib` only names the studies, so
    give the one the runs saw where their citations matter. Every run of 2026-10-30 was rebuilt this way into
    the per-fingerprint layout on 2026-10-31, the first evening's from the `.bak-2026-10-30` bib, and checked
    identical, table for table and dataset for dataset, to the outputs it replaced.
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

- **`build_catalogue.py`** — pools the themes of several runs into one catalogue for people to edit, by the overlap of
  their studies: each entry's support, its names, its core and occasional studies; plus the entries that nearly merged
  and a saturation check. Its JSON is what `group_studies.py --catalogue FILE` applies, placing the studies under fixed
  themes in place of finding them, with `None of these` for a study no theme fits.

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

**Second batch, the evening of 2026-10-30**: themes only, with the repair turn, ten runs a condition, the
same model and settings. Ledgers `aim-topic+guidance-programme-continuity/grouping-fa86c56b`,
`no-aim+guidance-programme-continuity/grouping-c818ef54` and `aim-topic/grouping-0af376c5` (the control);
comparison in `comparison-batch2.txt`, heatmaps in `batch2-plots/`. About 4 minutes a run.

- **The repair turn fired three times in thirty runs**, once per condition, and each cleared its slips; no
  run has a problem left. Studies placed twice: 0 to 7 of 40 a run.
- **Mean omega index at the theme level, within each condition** (ARI within 0.02):

  | condition | runs | omega |
  |---|---|---|
  | topic + continuity instruction | 10 | 0.37 |
  | no aim + continuity instruction | 10 | 0.51 |
  | topic (control) | 10 | 0.42 |
  | topic, first batch | 5 | 0.45 |
  | no aim, first batch | 5 | 0.53 |

  So with ten runs, no aim stays the most stable, and the topic line's figure holds from five runs to ten.
  The stable-core counts are not comparable across five and ten runs, a core having to hold in every run.
- **The continuity instruction mostly renames the theme rather than dissolving it.** A theme about
  implementing programmes appears in 9 of 10 runs with it and the topic line, 8 of 10 with it and no aim,
  against 10 of 10 without it; under the instruction it is called "implementation" and often drops
  "sustainability" from its name. The nine studies that sat in it in at least 7 of 10 control runs share a
  theme at a mean co-membership of 0.68 with the instruction against 0.85 without (topic line), and 0.82
  against 0.96 (no aim). **So they group on a topic of their own, implementing programmes in schools**, which
  the word's second sense only labelled; and the instruction costs stability (0.42 to 0.37 with the topic
  line) for a partial loosening of that group.

**The final set, 2026-11-02**: run e's `04b-final.bib`, 40 studies, the topic line, themes only, ten runs; ledger
`scopebib-2026-10-29e/llm-grouping/aim-topic/grouping-87e42fa2`. Theme-level omega 0.40, ARI 0.44, against 0.42
and 0.43 for the topic line on run d's set. Four to seven themes a run. The repair turn fired six times in ten
runs, against three in thirty the evening before, each time clearing its slips; why more often is not known.
Seven stable cores hold 23 of the 40, the largest two of five studies: one on sustaining school programmes, one
on the conceptual and policy link between health and sustainability. Half the runs split sustaining a programme
from implementing it as two themes. Studies placed twice: 0 to 13 of 40 a run.

**Applying a draft catalogue, 2026-11-02**: the final set, the topic line, ten runs, under the unedited 10-entry catalogue
(cut 0.7); ledger `aim-topic+catalogue-draft-cut0.7/grouping-ddf46036`. **ARI on primary placements rises from 0.44 to 0.76**;
omega from 0.40 to 0.57. Fixing the themes does what it was expected to, for each study's main place. Omega stays lower
because the second placements do not settle: 9 to 30 of the 40 studies are placed twice in a run, against 0 to 13 when the
model finds its own themes, despite the same instruction to treat that as the exception. No run used `None of these`, and no
repair turn fired. Primary placements per theme average 2 to 6 of the 40, the ten together taking 40. Stable cores read high here, at
38 of 40, being chained through those many second placements.

**The 7-entry draft (cut 0.8), same settings**; ledger `aim-topic+catalogue-draft-cut0.8/grouping-33e8affd`: ARI on primary
placements 0.82, omega 0.64, against 0.76 and 0.57 with 10 entries. Fewer themes agree more, which is partly what fewer themes
do, and is no argument for fewer: the 7 lack the community and non-Western entries of the 10. 10 to 24 of the 40 placed twice in
the runs listed; `None of these` again unused; three repair turns. Between the two catalogues the runs agree at ARI 0.55, so the
catalogue a study is sorted under decides much of where it lands.
