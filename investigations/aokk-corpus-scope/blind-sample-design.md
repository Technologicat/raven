# The blind sample: a test of the revised definition against people

*Draft, 2026-11-04. For the maintainer's review, then the team's agreement; nothing here has run.*

## Why

The team's labels on the held-for-review tier cover one hard tier of records, ones a judge had kept. They say nothing
about what the tool removes, and the revised definition was written after reading them, so its score on them is partly
fitted. A fresh sample, rated by people who do not see the tool's answers, is the test that can fail.

The two questions it answers, in order of what a mistake costs:

1. **How much in-scope work does the tool remove?** A wrong removal leaves nothing behind to notice it by.
2. **How much out-of-scope work does it keep?** A wrong keep costs a reader a line.

## The design

**When:** after the team has answered the scope questions, since they rate against that scope, and it is what the definition
says. Before the full run, whose definition this tests.

**The pool:** the confidence probe's seeded sample of 300 sifted records, minus any the team has already rated (6 of them, at
last count). It is a random sample of the corpus, so a rate measured in it carries over.

**The tool's answers:** the 3.8 27B over the pool under the revised definition, about an hour by the probe's timings, with all of
it counting toward the full run.

**The strata**, by the tool's outcome, about 60 records in all, as asked of the team:

| Stratum | Records | Why this many |
|---|---|---|
| Removed, split evenly across the tests that decided it | 30 | The costliest mistake, and the largest group, so the most; a clean 30 bounds it at about 10% |
| Kept, split across `study_kind`, `agent_kind` and `focus` as far as they go | 20 | The tags are what the team decides on; this shows what each holds |
| Sent to a person | 10 | A check that holding is worth a person's time |

Draws are seeded and recorded. Each stratum's share of the pool is kept, so that rates can be weighted back to the corpus:
oversampling removals changes the precision of the estimate, and must not change what it estimates.

**Double rating:** about 15 of the 60 go to two reviewers (or, failing a second one, to the same reviewer again after enough time for
a fresh look, which measures consistency rather than agreement). Two people's agreement is the yardstick for the tool's: a tool agreeing
with people about as often as people agree with each other has reached what labels can show.

**What reviewers see:** a shuffled list with title, abstract, venue and link, and no trace of the stratum or the tool's answer. The
columns are the ones they added themselves, in place of `mark`: decision, whether the abstract was enough, and a short reason.

**What they do:** decide from the title and abstract, and say so where that is not enough. So this measures people and tool on the
same evidence. The earlier round was deeper than a sample at this scale needs.

## What it can and cannot say

Sixty records give wide intervals, and that should be said with the result rather than found out afterwards. The bound applies to the
group rated and not to the whole sample: with no wrong removal among 30, the one-sided 95% upper bound on the removal error rate is
about 10%. A bound of 5% on removals alone takes 60 of them, about 90 records in all with the other groups. So:

- **It can show a problem**: several in-scope removals would be a clear signal to change the definition before the full run.
- **It cannot certify the tool**: no result here means "the tool removes nothing in scope". Reporting should give the bound.
- **A second round is cheap to add**, the design being the same with more draws, if the first leaves the answer open. It needs only
  more removals rated.

Labels from people's reading remain a reference and not ground truth: the double-rated records say how far.

## Open

- **Who rates, and how many records each**: for the team to say, in answer to the question list's section 4.
- **Whether 60 is the right size**: more narrows the bound, at the team's time. It is a judgement for them to make, knowing the bound above.
- **Double rating**: left out of the first request to keep it small, to raise when they agree to the rest.
- **Mixed tags**: a stratum smaller than its share of draws takes what there is, and the rest go to the other tags.

## To build, when agreed

One script here, about the size of `rerun_extract_on.py`: run the tool over the pool, draw the strata, write the shuffled sheet and
a separate gitignored key, and score returned decisions against the key with the weights. `score_against_decisions.py` covers the
unweighted half.
