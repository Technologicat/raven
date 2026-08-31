# Brief: sorting the AOKK corpus into in-scope and out-of-scope

**Filed 2026-08-31.** A batch LLM pass over `00_stuff/datasets/AOKK/multisource.pickle` (5007 records)
that flags records the literature search pulled in but that are not about the topic, for Juha to review.

Origin: noticed while looking at high-D clustering results on this corpus. Not scheduled; this exists so
it is not re-derived.

## The question the corpus is supposed to answer

`00_stuff/rawdata/AOKK/search-phrase.txt` holds the boolean query that produced the corpus: four blocks
AND-ed together — an AI-agent term, a collaborative/self-regulated learning term, `student*`, and a
higher-education term. In a form a model can be asked to match against:

> Studies on different aspects of the use of AI agents in higher education.

`00_stuff/rawdata/AOKK/research-questions.txt` narrows it further — pedagogical functions of AI agents
across the phases of data-informed decision-making, and the AI literacy that working with them requires
— but **the brief's target is the broad question above**, since that is what the search actually asked
and what a false positive fails.

## The problem is real, and it has two distinct shapes

Both confirmed against the corpus on 2026-08-31.

- **A term that matches outside the intended sense.** `"conversational agent"` catches *Controlled Yet
  Natural: A Hybrid BDI-LLM Conversational Agent for Child Helpline Training* — real research, nothing
  to do with higher education. It clears all four blocks because the other three are broad enough
  (`student*`, `universit*`, `collaborat*`) that almost any paper mentions them somewhere.
- **A term of art the query did not mean.** `"learning assistant"` matches 40 records, and the phrase
  means two things: an AI tool (*FretMate: ChatGPT-Powered Adaptive Guitar Learning Assistant*), and the
  established STEM-education term for a **human undergraduate Learning Assistant** (*Impacts of Learning
  Assistants on Student Belonging and Confidence*, *Experiential Learning through a Peer Learning
  Assistant Model in STEM*). The second sense has no AI in it at all.

The second shape is the one that argues for doing this with a model rather than with more search terms:
no exclusion list distinguishes the two senses of "learning assistant", but a reader distinguishes them
instantly from the title alone.

**They also cluster.** The high-D clustering run on 2026-08-31 put twelve of the human-Learning-Assistant
papers in a cluster of their own, so a per-cluster pass is a plausible cheaper first cut — worth trying
before the per-record pass, and worth *not* relying on, since it only finds the false positives that
happen to be numerous enough to form a cluster. The singleton helpline paper would not be found this way.

## Approach

Modelled on `investigations/agent-batch-classification/classify_papers.py`, which did the same shape of
job over ~1600 arXiv papers, and lands as its own investigation bundle beside it.

1. **Pass 1, titles only, in batches.** Cheap, and enough for the clear cases in both shapes above.
   Ask for a verdict plus a confidence, in JSON, as that script does.
2. **Pass 2, one record at a time, title *and* abstract.** For everything pass 1 was not sure about.
   83% of this corpus has an abstract, so the second pass has real extra evidence to work with — which
   is not true of every corpus and is worth stating as a precondition.

Output is a review list, not a deletion: the deliverable is "here are the N records I think are out of
scope, with a one-line reason each", and the corpus is not modified.

### The escalation rule has a known failure mode — do not reuse it unexamined

`investigations/agent-batch-classification` found that **the model was most confident about the inputs
carrying the least information**. Its escalation was driven by the model's own confidence, so exactly the
cases that most needed a second look were the ones the rule never re-examined.

That finding transfers directly, because the design above escalates on confidence too. Titles here vary a
lot in how much they say, and a short generic title is precisely where a title-only verdict is worth
least and may be delivered most confidently. So pass 2 should take **everything the model is unsure
about, plus everything whose title is short or generic regardless of confidence** — the second criterion
measured from the title, not asked of the model. Better still, calibrate first: hand-check a sample of
pass-1's high-confidence verdicts before trusting the rule at all.

## Open

- **Whether to judge against the broad question or the research questions.** This brief says the broad
  one. The narrower reading would flag many more records, and "not about data-informed decision-making"
  is a different claim from "not in scope" — worth deciding before the run, since it changes what the
  output means.
- **What to do with the answer.** A filtered `.bib`, a flag on the dataset records, or just a list. The
  cheapest useful thing is the list.
