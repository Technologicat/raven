# How Raven names a cluster

*The rules as built, written 2026-11-26 and reviewed by the maintainer on 2026-11-27; to be checked once more against
the next re-import of the test corpora.*

The rules that turn a group of records, such as a cluster of the Visualizer's map, into the words shown for it: the
phrases quoted from the records, the themes an LLM names, and how the two are shown. Each rule is given with the
failure it prevents, since that is what tells a reader whether a case in front of them is the exception the rule
was written for. The code is `raven.common.keyphrases`, for the phrases, and `raven.visualizer.importer` and
`raven.visualizer.entry_renderer` for the LLM's part and the display. The measurements behind the rules are in the
repository's `investigations/cluster-keywords/`.

- [Two lists, two jobs](#two-lists-two-jobs)
- [Phrases: what counts as one](#phrases-what-counts-as-one)
- [Phrases: when two spellings are one term](#phrases-when-two-spellings-are-one-term)
- [Phrases: capitals](#phrases-capitals)
- [Phrases: ranking, and the label](#phrases-ranking-and-the-label)
- [Themes: the LLM's keywords](#themes-the-llms-keywords)
- [The display](#the-display)
- [Known limitations](#known-limitations)
- [Considered and rejected](#considered-and-rejected)
- [References](#references)


## Two lists, two jobs

A cluster has two lists of keywords. **Themes** are the LLM's keywords: what the cluster's papers share, named in
words that may appear in none of them. **Quoted** are phrases taken from the papers, ranked by how particular they are
to the cluster: what the papers actually say, and the words to search by.

- **Neither list can do the other's job.** Phrases can only quote, so an abstract theme that no paper names in so many
  words is beyond them; an LLM abstracts, and in doing so leaves out the specific names a reader searches for. A
  single merged list of six had no room for both, which is why there are two lists.
- **The Themes are the list read closely, and the Quoted the list consulted for specifics.** So a weak Themes list
  costs a cluster its gist, while a weak Quoted list costs a few search hooks, and effort goes where that says.


## Phrases: what counts as one

`raven.common.keyphrases.extract`, per record, on a spaCy parse of its title and abstract.

- **A candidate is a noun phrase with its head at the end, and every head-final part of it** ("large language
  model", "language model", "model"), up to four words. The long form rarely repeats across records where the
  shorter ones do, so which length is worth showing is left to the ranking.
- **Determiners, pronouns, numbers and the like are stripped from the ends, and so is a leading stopword.** A
  stopword at the end stays when a noun qualifies it ("tool use") and goes when an adjective does ("remarkable
  ability").
  - **Except a word joined to the next by a hyphen**, which is the first half of a word rather than a word: stripping
    "test" from "test-time compute" (a feature of reasoning LLMs) left "time compute".
- **Punctuation inside a noun chunk, other than a hyphen, splits it in two, and the punctuation goes**: the chunk
  "large language models (LLMs" gives "large language models" and "LLMs".
- **A part starting with a stopword or with no letter in it is skipped**: "and photo-fermentation", "2.0 license".
- **A possessive stays on its word**: "prisoner's dilemma". Dropping it made "prisoner dilemma", a phrase nobody
  writes.
- **The head is lemmatized, with two exceptions the small English pipeline gets wrong**: "data" (which it makes
  "datum") and any word in "-ics", which is either a singular field ("physics", made "physic") or a plural
  ("characteristics"); the corpus-wide step below decides which.
- **A title in Title Case or ALL CAPS is lowercased before parsing**, since a tagger reads a capitalized word as a
  proper noun and leaves it unlemmatized.
  - **In a Title Case title, a word with a capital past its first letter keeps its capitals** ("CO2", "TiO2",
    "LoRA"). A two-letter symbol such as "Co" (cobalt) looks like any other Title Case word, so it is lowercased,
    and only the abstract says it was a symbol.
  - **ALL CAPS gives no such hint**, so it is lowercased throughout.
- **The text is NFKC-normalized before parsing**, so a phrase written "TiO₂" is extracted as "TiO2", and counts
  together with one written that way. The records themselves keep their text as written.
- **Each phrase is also kept as written**, its capitals and its punctuation as the paper has them, for the rules below
  that need to know how a word was spelled. A capital that only starts a sentence is not counted as written.

**Abbreviations** are found where a term is followed by its abbreviation in parentheses, "self-regulated learning
(SRL)", by the algorithm of Schwartz and Hearst (2003): the abbreviation is one token of 2–10 characters, and the long
form is the shortest run of the words before the parenthesis in which every character of the abbreviation occurs in
order, its first starting a word.

- **The abbreviation must have two capitals or a digit** ("SRL", "ChatGPT", "TiO2"). Without that, "the gas evolved at
  the anode (oxygen)" defines the word "oxygen".
- **The long form is the text as written**, commas on their words: "verification, validation, and uncertainty
  quantification".


## Phrases: when two spellings are one term

`raven.common.keyphrases.normalize`, over the whole corpus. Each fold happens only where the corpus contains both
forms, which is what keeps a word that merely looks like a variant ("process", which ends in "s") as it is.

- **A plural onto its singular.**
- **A final word in "-ics" onto "-ic", where the "-ic" form heads a phrase somewhere in the corpus** ("a key
  characteristic"), and never for a field of study ("kinetics", "dynamics", "mechanics", "economics" and the rest
  of a short list).
  - **Heading a phrase, rather than merely appearing as a noun**: a stray noun tag on "mechanic" or "robotic" was
    enough to fold "mechanics" and "robotics", which mean something else.
  - **A list for the fields of study, because counting cannot tell them from plurals.** A rule by counts would fold
    "X-ics" when "X-ic" is common enough as a head. But a corpus has its slips, such as "a first-order kinetic", and
    in one corpus they made "kinetic" about as common beside "kinetics" as "characteristic" is beside
    "characteristics". So no threshold separates the two, and the fields are listed by hand. The list is a judgement,
    not a measurement.
- **An abbreviation with its long form, word by word, where the word is written as the abbreviation is.** So "SRL" is
  "self-regulated learning", and "Co" stays cobalt in a corpus that defines "CO" as carbon monoxide. Matched without
  case, element symbols merged into whatever the corpus defined: cobalt into carbon monoxide, iron into faradaic
  efficiency.
  - **A definition made by two records or more applies corpus-wide; one made by a single record stays in it**, so a
    paper calling its system CARE does not take over the word "care" everywhere else. Since the match is by spelling,
    a lowercase "care" is never taken over either way; see *Known limitations* for an uppercase one.
  - **An abbreviation displays as its short form when most records using it leave it unexplained**, its readers
    evidently knowing it ("ChatGPT"), and spelled out otherwise, with the abbreviation after it as the papers write
    it ("fast weight programmer (FWP)"), which tells a reader the papers may use either.
    - **The line is a share, not a rule of always or never**: an abbreviation is spelled out when fewer than half of
      the records using it leave it unexplained, which is to say when most of the papers using it also define it.
      `clusters_keyword_short_form_share`, at 0.5, sets the line.
- **Spellings of one term, onto the one most records use**: a hyphen, a space or nothing between two words
  ("retrieval-augmented", "retrieval augmented"; "bit-width", "bitwidth"), a possessive ("prisoner's dilemma",
  "prisoner dilemma"), and a British and an American spelling ("modelling", "modeling"; "fibre", "fiber").
  - **A word written with a capital keeps its spelling apart**, so "Co catalyst" (cobalt) does not fold into
    "cocatalyst".
  - **The British rules are regular endings only**: "-isation", "-ise", "-yse", "-elling", "-our", and "-re" after a
    consonant. "metre" and "meter" fold too, a price paid knowingly, being rare; so do "re-cover" and "recover".


## Phrases: capitals

A term is stored lowercase and displayed with capitals as the corpus writes them (`casing_table`, `recase`).

- **Each record votes for the spelling it mostly uses, and the most votes win, lowercase on a tie.** So "Solomonoff"
  and "LoRA" keep their capitals, and a word some papers capitalize for emphasis does not. The corpus itself says which
  spelling is right, which works because the corpora are large.
- **Votes are counted in the abstract only.** A title in Title Case is lowercased before parsing, so its proper names
  would vote lowercase, and they used to outvote the abstracts.
- **Every occurrence of a word counts, whatever its part of speech there**, once the word appears as a noun or
  adjective somewhere. Some papers capitalize the second part of a compound, "AI-Based", which the tagger then reads as
  a proper noun; counting nouns and adjectives alone gave that spelling every vote over the same word as a verb
  ("based on"), and the display came out "simulation-Based". A compound whose parts are both proper names, such as
  "Timor-Leste", keeps its capitals by the same vote.
- **The LLM's keywords are recased the same way**, and two of the LLM's habits are undone:
  - **Title Case throughout**, as in "Foundation Models", which comes out "foundation models".
  - **A capital on the first letter only**, as at the start of a sentence: "A posteriori error estimation" comes out
    "a posteriori error estimation".

  In both, a word the corpus capitalizes keeps its capital. The second rule is the brittle one: a proper name that the
  corpus never writes, and so has no vote for, loses its capital.


## Phrases: ranking, and the label

- **Class-based TF-IDF over the clusters, on record counts** (Grootendorst 2022), so that a phrase's score also gives
  the share of the cluster's records that use it.
- **The first phrase is the cluster's best**, saying what it is about; in a corpus on one subject this is often a
  phrase other clusters share, and that is allowed there.
- **The rest say how the cluster stands out**: the ranking with the corpus-wide phrases removed, ordered once by rank
  and once by similarity to the cluster's centroid in the embedding space (after KeyBERT; Grootendorst 2020), the two
  orders fused by reciprocal rank (Cormack et al. 2009).
  - **Two orders, because each fails alone, in opposite places**: the ranking fills an abstract cluster's label with
    the ordinary words most of it uses, and the similarity reaches for narrow terms in a technical one. A phrase high
    in both survives the fusion.
- **A phrase is corpus-wide when it is in at least a tenth of the clusters, and in no fewer than five** where there
  are five. With a handful of clusters the floor of five dominates, so in a small corpus only a phrase in about half
  the clusters counts; see the display's ordering below for what covers the rest, and *Known limitations*.
- **Two phrases nested in each other count as one**, and the longer replaces the shorter when it covers at least half
  as much of the cluster: every record with "latin hypercube sampling" also has "sampling", so ranking alone would
  keep the bare word and lose the name.


## Themes: the LLM's keywords

`raven.visualizer.importer`, with `clusters_keyword_method` "llm" or "hybrid".

- **The model reads the cluster's titles and abstracts, a sample of them in a large cluster** spread from its centre
  outwards, **and, after them, the cluster's phrases with the share of records using each**, as evidence of what the
  whole cluster contains. The prompt says the phrases are data, not suggestions, so that it does not copy them.
- **It is asked for up to fifteen keywords.** The models tested tend to give the general keywords first and the more
  specific ones after them, and fifteen was enough to catch both. The label shows six, and the dataset keeps them all.
- **The format rules come last in the prompt**, after the papers and the evidence, where a reader setting up the task
  expects them. One instruction, stated once.
- **A reply written as a Markdown list is read as one**, with a warning in the log, and so is one that states its
  theme first and then lists the keywords under a heading ("Keywords:", "Suggested keywords:"). Rejecting them
  recorded good answers as no theme at all.
- **A reply that is no list at all is rejected and asked again**, up to three times. Now and then a model answers with
  a bulleted summary of the papers instead of the keywords asked for; its items are sentences, ending in a full stop,
  and no heading names keywords, which is how it is told apart.
- **"keyword extraction failed" is a valid answer**, recorded as no theme and not retried, since asking again only
  pressures the model to invent one.
- **A failed call to the LLM backend is not an answer.** The importer waits for the backend, 5, 15 and 45 seconds
  between probes, and asks again; if it stays down, the import stops and says so.
- **A second LLM pass folds variants across clusters** (an acronym in one, its expansion in another). It is asked for a
  mapping, so that only replacements by keywords the first pass actually produced are kept.
- **Answers are saved by prompt**, so a re-import asks the model only what it has not already answered.


## The display

`raven.visualizer.entry_renderer.format_cluster_keywords`, for a dataset that has both lists.

- **Themes: the LLM's keywords without those it gave too many clusters**, by the same corpus-wide test as the
  phrases, **with the terms other clusters share moved after the cluster's own, except the first** (`shared_last`).
  The first stays where the LLM put it, so every cluster keeps its lead theme; the others make room for what tells
  this cluster apart. Ordering purely by rarity was tested, and found to drop each cluster's core theme.
- **Quoted: the phrases, without one that repeats a theme**, compared in both their stored and displayed forms, so an
  abbreviation counts as a repeat of its long form. Only exact repeats are left out, not phrases containing a theme:
  a one-word theme would otherwise take every phrase with that word in it.
- **Each list is shown where it has something, and left out where it is empty.** A small cluster may have no phrase
  left that a theme does not say, and a cluster the LLM found no theme in has no Themes. With neither, the cluster's
  heading stands alone.
- **A tooltip over more than two clusters shows one list per cluster**, as two crowd the titles off it: the Themes
  where the cluster has them, and the Quoted list where it has not.
- **Fewer than six is fine.** A small cluster's phrases thin out into generic words; nothing reasonable fixes that,
  and padding would only add noise.

A dataset with one list only, phrases or the LLM's, shows it as a single label.


## Known limitations

- **The corpus-wide floor of five suits a large corpus, not a small one.** In a corpus of ten clusters a phrase has to
  be in half of them to count as corpus-wide. It depends on the number of clusters rather than on the corpus's size,
  too: once the clustering's granularity is tunable, a large corpus clustered coarsely, at a low "conceptual zoom
  level", has few clusters as well. To be tuned for few clusters, with two cautions from the rejection below: measured
  across corpora rather than fitted to one, and with an eye on the Themes list, which the same filter decides, so
  that a lower floor also removes more shared themes outright.
- **Two records defining one abbreviation make it corpus-wide**, so two papers on a system called CARE would expand
  every uppercase "CARE" in the corpus, including one meaning something else. A lowercase "care" is safe, the match
  being by spelling.
- **Themes are the LLM's own spelling of an abbreviation** ("VVUQ"), even where the corpus defines it.


## Considered and rejected

- **One merged list**, the LLM's keywords leading and the phrases filling up: six slots could not hold both jobs.
- **Phrases as whole sentences**, and terms embedded by their sentences rather than on their own (2026-11-24; the
  reasons are in `investigations/cluster-keywords/README.md`).
- **A word-count limit on a bulleted reply's items**: a number with nothing behind it, where the header and the full
  stops already tell a summary from a list.
- **Lowering the corpus-wide floor as the remedy for repeated themes** (2026-11-26, from a small corpus where a few
  themes repeated across its clusters): a single number, easy to fit to the corpus in front of us, and it removes
  shared themes outright where ordering only moves them down, so ordering was chosen (`shared_last`). This rejects the
  floor as that remedy, not its calibration, which *Known limitations* leaves open.


## References

Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and
individual rank learning methods. In *Proceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval*, pages 758–759. https://doi.org/10.1145/1571941.1572114

Grootendorst, M. (2020). KeyBERT: Minimal keyword extraction with BERT. Zenodo.
https://doi.org/10.5281/zenodo.4461265

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
https://doi.org/10.48550/arXiv.2203.05794

Schwartz, A. S. and Hearst, M. A. (2003). A simple algorithm for identifying abbreviation definitions in biomedical
text. In *Pacific Symposium on Biocomputing*, volume 8, pages 451–462. https://doi.org/10.1142/9789812776303_0042
