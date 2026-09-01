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

**They also cluster, and a cheap screen finds them.** Measured 2026-09-01 on the agglomerative
clustering (cut at 100, min size 5) with LLM keywords, `investigations/highdim-clustering/`:

- **The human-Learning-Assistant group is a cluster of its own, 58 records**, keyworded *Learning
  Assistants, Peer Learning, Collaborative Learning, Active Learning, Higher Education, Pedagogical
  Training* — **with no AI term in it at all**, in a corpus about AI agents in higher education. The
  earlier HDBSCAN run found twelve of these and left the other 46 unclustered, which is one more
  instance of the coverage argument that decided the algorithm.
- **The AI/human sense split is handled correctly**: *"Generative-AI, a Learning Assistant?"* sits in an
  AI cluster, and a separate cluster covers AI teaching assistants. The clustering separates the two
  senses of the phrase that the boolean query cannot.
- **Screening on "no AI term among the keywords" puts 4 clusters of 83 in front of a reader**, covering
  107 records (2.1%): the 58 above, 23 on participatory design and HCI, 17 on computer-supported
  collaborative learning, and 9 on digital literacy and social equity. Judging four clusters is a
  different proposition from judging 5007 records.

So run the screen first. **It is a first cut and not a substitute**: it can only find false positives
numerous enough to form their own cluster, and the singleton helpline paper sits inside a large AI
cluster where nothing about its keywords gives it away.

Note the screen is currently a script over the investigation's output. The keyword dialog proposed in
`briefs/visualizer-keyword-pools-brief.md` would make it an ordinary GUI operation — select the clusters
carrying each AI term, invert, look at what is left — which is a better home for it than a script.

## Approach

Modelled on `investigations/agent-batch-classification/classify_papers.py`, which did the same shape of
job over ~1600 arXiv papers, and lands as its own investigation bundle beside it.

1. **Pass 1, titles only, in batches.** Cheap, and enough for the clear cases in both shapes above.
   Ask for a verdict plus a confidence, in JSON, as that script does.
2. **Pass 2, one record at a time, title *and* abstract.** For everything pass 1 was not sure about.
   83% of this corpus has an abstract, so the second pass has real extra evidence to work with — which
   is not true of every corpus and is worth stating as a precondition.

**Two outputs, and the source corpus is modified by neither** (decided 2026-09-01):

- **A filtered `.bib`** holding the records judged in scope, so it imports straight into Visualizer and
  the corpus can be looked at without the false positives in it.
- **A list of what was dropped, and why** — one line of reason per record. This is the reviewable half:
  a filtered file alone gives no way to tell a good cut from a bad one, and the reasons are what make
  the run auditable after the fact rather than only during it.

### The escalation rule has a known failure mode — do not reuse it unexamined

`investigations/agent-batch-classification` found that **the model was most confident about the inputs
carrying the least information**. Its escalation was driven by the model's own confidence, so exactly the
cases that most needed a second look were the ones the rule never re-examined.

**But it transfers less severely than it first appears, and the reason is worth knowing before copying
that script's structure.** The flaw's bite depends on how much the inputs vary in informativeness, and
*filenames* vary enormously — `2301.12345.pdf` carries nothing at all, so there the badly-served subset
was large. Titles do not. Measured on this corpus, 2026-08-31: **median 13 words, 5th percentile 7, and
only 33 records (0.7%) under five words**.

So the mitigation is cheaper here than a heuristic. Those 33 are a hand-checkable list, and they are
genuinely the uninformative ones — *Editorial*, *Afterword*, *Machine culture*, *Generative AI*. Send
every record under about five words to pass 2 regardless of what the model says about its own
confidence, and the known failure mode is closed without tuning anything. Calibrating first is still
worth it: hand-check a sample of pass-1's high-confidence verdicts before trusting the rule.

(The wider lesson from that run stands even though this corpus dodges it: an escalation rule driven by
the model's own confidence is blind exactly where the input is thin, so it needs a second criterion
measured from the input rather than asked of the model.)

## Settled 2026-09-01

- **Judge against the broad question**, as this brief proposed: *studies on different aspects of the use
  of AI agents in higher education*. Juha's call, and the reason matters for what comes after — **the
  research questions get their own, more detailed passes later**, and those want the corpus ingested into
  Librarian first. So this run is the coarse cut that makes the corpus worth ingesting, and "not about
  data-informed decision-making" stays a separate question asked separately, rather than being folded in
  here where it would look like the same verdict.
- **Both outputs**: the filtered `.bib` and the dropped-with-reasons list. See *Approach* above.
