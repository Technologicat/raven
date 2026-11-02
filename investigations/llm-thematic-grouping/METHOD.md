# Grouping a review's accepted studies with an LLM: the method

How to get from a review's accepted studies to a table of themes, with an LLM doing the reading and people making
the decisions. For a set small enough that every abstract fits in one prompt, where clustering by embedding finds
little. The scripts are in this bundle, and `README.md` holds the measurements.

**Status, 2026-11-02**: steps 1 and 2 have run on one set. Step 4 is built and its first measurement is queued. Steps
3 and 5, and the check against published tables, are not done yet. So this describes the method as designed, and the
sentences under *What was measured* are the only ones resting on data.

## The problem it answers

Run a model several times over the same studies and it returns different groupings, each with its own strengths: a
complete list of themes may need several of them (observed by the study's authors, reading the runs, 2026-11-02).
Two simpler answers fall short of that:

- **Taking one run**, the "best" one, loses what the others found. There is also nothing to maximize: no grouping is
  known to be true, and the run agreeing most with the rest is the most typical one, which is a different thing.
- **Averaging whole groupings**, consensus clustering (Strehl & Ghosh 2002; Fred & Jain 2005), outvotes a theme found
  in only a few runs, and that may be the theme worth having.

So what is combined here is the theme. The method then follows the two moves of qualitative content analysis
([Elo & Kyngäs 2008](https://doi.org/10.1111/j.1365-2648.2007.04569.x)): first derive the categories from the
material, inductively, and then apply them, deductively.

## The steps

### 1. Find themes, several times (inductive; the model)

`group_studies.py`. One prompt holds every study's citation, title, keywords and abstract, and asks for themes by what
the studies are about, a reason per placement, and one placement per study marked primary. A study may sit under
two themes, as in hand-made tables.

- **The order of the studies is shuffled per run**, the run's number being the seed, so that order cannot pass for
  stability. What remains between runs may be sampling noise, or the problem admitting several good groupings.
- **The guidance is an optional aim**, a line saying what the review is about. Research questions proved a poor aim:
  the themes came to follow the questions, against an instruction not to (*What was measured*).
- **A reply is checked, and slips are sent back**: a study placed nowhere, an item number naming no study, a study
  without exactly one primary placement. Each is named in a repair turn asking for the whole corrected answer, up to
  two turns, and all of it is recorded with the run.

### 2. Pool the themes into a catalogue (mechanical)

`build_catalogue.py`. Every theme of every run becomes a point; two themes are close when their studies overlap
(Jaccard), and average linkage groups them into entries. Each entry states:

- **In how many runs it was found.** This is information rather than a verdict, and low support does not remove an entry.
- **What the runs called it**, the most typical name first.
- **Its studies**, core (in at least half of the entry's themes) or occasional, with how often.

The page also lists the entries that nearly merged, since the cut-off is a judgement no calculation settles, and how
many new entries each further run added. That is a saturation check, after
[Guest, Bunce & Johnson 2006](https://doi.org/10.1177/1525822X05279903): when further runs stop adding entries, more
runs are unlikely to find more themes.

This is also what people read, one page in place of one grouping per run.

### 3. People edit the catalogue (the decisions)

The authors keep, merge, split and rename entries, and decide what the themes are called. A theme found in few runs
stays if the review needs it. Every name is theirs in the end, as in hand-made tables.

### 4. Apply the fixed themes (deductive; the model)

`group_studies.py --catalogue`. The model places every study under the authors' themes, over several runs, or under
"None of these", which shows what the catalogue lacks. A study the runs place inconsistently goes on a short list for
people to settle.

### 5. People settle what remains

The table is the catalogue, with each study placed by agreement across the runs or by the authors. The per-placement
reasons are there to check any placement against.

## How agreement is measured

`compare_groupings.py`, indifferent to what a group is called:

- **The omega index** ([Collins & Dent 1988](https://doi.org/10.1207/s15327906mbr2302_6)), the adjusted Rand index
  generalized to groupings where a study may sit in several groups. It equals the ARI on disjoint groupings (checked).
- **The adjusted Rand index on primary placements**
  ([Hubert & Arabie 1985](https://doi.org/10.1007/BF01908075)), the familiar number.
- **Per study**: which studies share a theme in every run (stable cores), and which drift.

Agreement says whether a grouping repeats, and says nothing of whether it is good. That takes people, or tables people
made.

## What was measured

On one health-education set of 40 studies, with one model; `README.md` has the detail.

- **Run-to-run agreement is moderate**: omega 0.35 to 0.53 at the theme level across the conditions tried, from five to ten
  runs each, so a difference of about 0.1 is a lead rather than a result.
- **The guidance moves the grouping.** Research questions pull the themes toward themselves despite an instruction
  against it. With no aim the runs agree most, and name the most topical themes.
- **Sub-themes agree less than themes**, at 0.27 to 0.36, so this method asks for themes only.
- **A "do not make this a theme" instruction mostly renamed the theme**, and cost agreement.
- **Saturation looks fast on this set**: over random orders of ten runs, the first added 5.8 catalogue entries, the second
  1.0, and the tenth 0.2. How many entries there are depends on the cut-off: 17 at 0.5, 10 at 0.7, 7 at 0.8.

## What is not known yet

- **Whether fixing the themes raises agreement.** Expected, and step 4's first measurement is queued, on draft catalogues
  no person has edited.
- **Whether the themes match what people found**, which is the question that matters. The earlier studies in the
  series have published tables, recovered as test sets; they have not been run yet.
- **How it scales.** Those test sets are three to four times the size of this one.
- **How much depends on the model.** Everything above is one model.

## References

- Collins, L. M. & Dent, C. W. (1988). Omega: A general formulation of the Rand index of cluster recovery suitable for
  non-disjoint solutions. *Multivariate Behavioral Research*, 23(2).
- Elo, S. & Kyngäs, H. (2008). The qualitative content analysis process. *Journal of Advanced Nursing*, 62(1).
- Fred, A. L. N. & Jain, A. K. (2005). Combining multiple clusterings using evidence accumulation. *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 27(6). <https://doi.org/10.1109/TPAMI.2005.113>
- Guest, G., Bunce, A. & Johnson, L. (2006). How many interviews are enough? *Field Methods*, 18(1).
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2.
- Strehl, A. & Ghosh, J. (2002). Cluster ensembles: A knowledge reuse framework for combining multiple partitions.
  *Journal of Machine Learning Research*, 3, 583–617. <https://www.jmlr.org/papers/v3/strehl02a.html>
