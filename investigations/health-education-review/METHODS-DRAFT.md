# The health-education review: a description of the methods, for the manuscript

*A free-form account of how the data was processed, from the search to the seven themes, for the authors to write
the manuscript's methods section from. It is not the section itself. It says what was done, with the numbers, in
an order that follows the flowchart. Wording, length and what to leave out are the authors' call. Started
2026-11-19 from the processing record in `README.md` beside it, which holds the provenance of every figure here.*

*Places where the record runs out and the authors know the answer are marked **[Question]**.*

***Pending:** the coauthor has since excluded two more studies on closer reading (reported 2026-11-19), which two
not yet known. The counts below are the 40 as screened, and change once those two are identified.*

## Overview

Processing ran in seven steps:

1. The records were retrieved from Scopus.
2. Two technical steps prepared them: deduplication, and completing truncated abstracts from Crossref.
3. They were screened for relevance with a large language model (LLM) against a written definition of the review's
   scope. Every borderline case went to the authors, and the authors' decisions were final.
4. The 40 included studies were grouped into themes by the LLM, in a single run.
5. The authors checked that grouping against every abstract and adopted it unchanged.

All of the LLM work was done locally, on the research group's own hardware: no record was sent to an external
service. The software was the open-source Raven toolkit (<https://github.com/Technologicat/raven>), version
0.2.11 (development at the time of the runs; 0.2.11 is to be released before the paper, and cited), running the model Qwen 3.8 27B, quantized to 4 bits (the `q4_k_xl` GGUF build), served by
LM Studio.

## Numbers for the flowchart

| Stage | Records |
|---|---|
| Identified in Scopus | 118 |
| Duplicates removed | 1 |
| Screened (titles and abstracts) | 117 (95 articles, 20 book chapters, 2 books) |
| Excluded at screening | 77 |
| Included, and grouped into themes | 40 |

How the 77 exclusions break down between the LLM and the authors is in *Screening* below. The flowchart could show
it as one box or as two.

The raw export held 119 entries. One was an empty record template, included by accident and removed before
processing; by the authors' decision (2026-11-17) it is not reported, so the flowchart starts at 118.

## Search

As the manuscript already describes it: Scopus, documents published 2000–2024, in English, searched between 18 May
and 13 June 2025, with a hierarchical keyword strategy running from single terms up to Boolean combinations.

## How the process differs from PRISMA

The flowchart borrows PRISMA's shape, but the process differs from what PRISMA assumes, in the same ways as the
group's ongoing AOKK review, which uses the same tools and approach (discussed 2026-11-19):

- **Screening and eligibility were one step, not two.** Where an abstract did not settle a record, the authors
  looked at its full text as needed.
- **PRISMA assumes** that screening is a basic mechanical filter, which is what was feasible when PRISMA 2020 was
  written; that every full text is retrieved and examined; and that full texts are retrieved only once screening is
  complete.
- **The screening criteria were not fixed in advance.** The definition of scope was written from the authors' first
  two passes by hand and refined over four trial runs, within that combined step. So the approach is exploratory
  rather than confirmatory.
- **The criteria may have been shaped by decisions made against a full text**, which the section should say.

## Preparing the records

**Deduplication.** One paper had been exported twice. It was merged with Raven's deduplication tool, which matches
records on DOI and on normalized title, leaving 117.

**Completing abstracts.** Some databases export a truncated abstract. Each record with a DOI was compared against
Crossref's copy of the abstract, and an abstract was completed from Crossref only where the record's text was a
prefix of Crossref's. That completed two abstracts. Afterwards all 117 records had an abstract.

## Screening

**The definition of scope.** The scope was written down before screening: *studies on sustainability, sustainable
development and education for sustainable development (ESD) in health education*. It came with notes on what
belongs, and with six grounds for exclusion:

1. Not about health education.
2. Sustainability not framed by sustainable development or the Sustainable Development Goals (SDGs).
3. Education for health professionals.
4. Preschool or early-childhood education.
5. Not education at an educational institution.
6. Not about education at all.

The grounds were derived from the authors' own first two passes over the records by hand (28 and 29 October 2026).
Four trial runs of the screening on the same records (28–29 October) were used to refine the wording before the run
reported here (2 November 2026).

**The screening procedure.** Three stages, each asked of the LLM, with a person deciding wherever the stages did not
agree or were unsure:

- **The judge** read every record and answered, for each ground, whether that ground positively applies, with a
  confidence level. It read the title and venue first, and the abstract for any record the title could not settle.
  A record was proposed for exclusion only when a ground applied with high confidence, or with medium confidence
  after reading the abstract. Any weaker finding sent the record to a person. A record where no ground applied was
  kept.
- **The reviewer**, an independent second pass, read each record the judge had proposed to exclude. It was asked
  the opposite question: is there a credible case that the record belongs in scope? It was not told that the record
  had been judged, nor shown the grounds. A record it made a case for went to a person instead of being excluded.
  As a control on the reviewer's discrimination, 40 kept records were mixed in, unlabelled.
- **Extraction and rules.** For each kept record, the LLM extracted structured fields from the record's own text:
  the setting, the learners, whether there is health content, the sense in which "sustainability" is used, and the
  health topic. Where a field had to be supported by a quotation, the quotation was checked against the record's
  text. Fixed rules over these fields then either excluded a record, using its own words as the evidence, or held it
  for a person. A rule resting on a quotation that could not be found in the text held the record rather than
  excluding it.

**What the screening proposed**, over the 117: 73 to exclude (72 by the judge, 1 by the rules), 36 to include, and
8 for a person to decide.

**The authors' decisions.** The authors read the records the screening sent to them, and their decisions overrode
the tool's in either direction:

- Of the 8 sent to them, they included 2 and excluded 6.
- They included 6 that the judge had proposed to exclude.
- They excluded 4 that the screening had proposed to include.

That gives 40 included studies: 32 of the tool's 36, plus the 6 reinstated, plus the 2 decided by hand. The 77
exclusions are the 67 proposed exclusions the authors let stand, the 4 they overruled, and the 6 they decided by
hand.

The authors had also decided 81 distinct records by hand during the trial runs. Those decisions were carried into the
reported run; where one contradicted the run (10 records), it was put in front of the authors first.

So the authors read 18 records: the 8 the screening sent to them, and the 10 where the run contradicted an earlier
hand decision. The other 67 proposed exclusions were not read by a person. Of those, 66 were the judge's, and the
reviewer had found no case for including any of them. The remaining one was excluded by a rule, resting on a
quotation from the record that had been checked against its text.

**For the limitations, or the discussion:** of the 6 exclusions the authors reversed, 4 had been proposed on the
SDG-framing ground at high confidence. That ground is the one where the model and the authors' reading of the scope
diverged most.

**Exclusion grounds in the flowchart.** If the flowchart should give reasons for exclusion, the counts by ground
for the screening's 67 standing exclusions are in `exclusion-grounds.md` beside the run (generated 2026-11-19). It
has two tables: one counting a record under every ground found for it, and one counting each record once, by its
combination of grounds. The 10 exclusions the authors made themselves have no recorded reason. They are listed
there with an empty column, for the authors to fill in if the manuscript keeps the breakdown.

## Thematic grouping

**Earlier attempts.** The group's earlier reviews grouped their studies by clustering text embeddings, the method
the manuscript currently describes. On these 40 studies that method found no clusters, either in the
high-dimensional embedding space or in a two-dimensional projection, so it was replaced.

**The method used.** All 40 studies were given to the LLM in a single prompt, each with its authors, year, title
and abstract, together with the review's aim: *"Sustainability education in health education, at primary and
secondary schools and in teacher education."* The model was asked to:

- group the studies into themes by what they are about, choosing the themes and their number itself;
- treat the aim as context, without making one theme per part of it;
- place every study at least once, under a second theme only by exception;
- give a one-sentence reason for each placement;
- mark one placement of each study as its primary one.

The answer came back as structured data (JSON) and was checked mechanically: every study placed, each with one
primary placement. Had it fallen short, the tool would have asked the model again with the problems listed; this
answer was complete at the first attempt.
It took one run of about four minutes, on 11 November 2026, and produced 7 themes, with every study under exactly
one of them:

1. Conceptual foundations for integrating health and sustainability in education
2. Implementation and sustainability of school-based health and sustainability programs
3. Curriculum content, policy frameworks, and whole-school approaches
4. Teacher education and higher education for health and sustainability
5. Program evaluations and empirical interventions in school settings
6. Health education in diverse, global, and under-resourced contexts
7. Pedagogical strategies, instructional methods, and technology in health education

(Table 5 numbers them #0 to #6, in this order.)

**The authors' check.** The authors compared this grouping with an alternative run without the aim, which gave four
broader themes. They read every abstract against the themes, and adopted the seven-theme grouping as the model
produced it, with no study moved and no theme renamed (17 November 2026). So the grouping is LLM-generated and
fully human-verified: every study's placement was checked, not a sample.

***Pending:** the paper uses one prompt throughout (the authors, 2026-11-19), so the measurement is being repeated
with the adopted run's own prompt, ten runs from 2026-11-19. The figures below, from the development prompt, are
then kept for reference only and replaced here.*

**On repeatability.** An LLM grouping varies from run to run, and the group measured how much, on these same 40
studies, while developing the method (2 November 2026; the measurements are in `../llm-thematic-grouping/`):

- **Ten independent runs** with the same model, the same aim and the same 40 studies produced four to seven themes
  each.
- **Agreement between runs**, averaged over all pairs: an adjusted Rand index of 0.44 and an omega index of 0.40.
  Both are 0 for chance agreement and 1 for identical groupings. The adjusted Rand index compares each study's
  primary placement (Hubert & Arabie, 1985); the omega index generalizes it to groupings where a study may sit
  under two themes (Collins & Dent, 1988).
- **A stable core**: 23 of the 40 studies fell into seven groups that stayed together in every one of the ten runs.

So the themes are a plausible reading of the material rather than the only one, and the authors' check is what
makes this one the result. The figures describe the method, not the adopted run, and one difference should be
stated if they are reported: the ten runs used the development version of the prompt, and the adopted run the
version built into the software. They ask for the same thing, themes with a primary placement and a reason for
each, but the development version framed the collection as a systematic review, showed each record's keywords as
well, and asked for the themes "the way a review presents its accepted studies as a table of thematic clusters".
Single runs suggest that such wording moves how many themes come out, so the figures are indicative of the adopted
prompt's variation rather than a measurement of it.

References, to check before citing: Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of
Classification*, 2, 193–218. Collins, L. M., & Dent, C. W. (1988). Omega: A general formulation of the Rand index of
cluster recovery suitable for non-disjoint solutions. *Multivariate Behavioral Research*, 23(2), 231–242.

## Still to come

- **Word clouds**: one of the 40 included studies, and possibly one of all 117 screened records beside it for
  comparison. Not yet made. They are made from the titles and abstracts, with stop words and publisher
  boilerplate (copyright statements and the like) removed.
- **The per-country and educational-level table**, which needs the full texts (`README.md`, *Open*).

## Details the manuscript probably doesn't need

These are kept here so that a reviewer's question can be answered:

- **Corrections to the final bibliography.** One year was corrected against Crossref, and one author's spelling. The
  grouping is unaffected.
- **Run settings.** Seed 42 for the reviewer's control sample; the instruments (the exact prompts) of each screening stage are
  fingerprinted in the run's `run.toml`. The grouping ran from Raven at commit `69da540e`.
- **Reference check.** All 40 included studies are in the manuscript's reference list (35 matched by DOI, 5 by
  title), checked with Raven's reference checker on 17 November 2026.
- **The Qwen 3.6 run** in the data folder belongs to another study's model comparison, not to this review.
