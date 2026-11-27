# How Raven names a cluster

*Draft, 2026-11-26: the rules as built, to be checked against the next re-import of the test corpora.*

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
- [Considered and rejected](#considered-and-rejected)


## Two lists, two jobs

A cluster shows two lines. **Themes** are the LLM's keywords: what the cluster's papers share, named in words that
may appear in none of them. **Quoted** are phrases taken from the papers, ranked by how particular they are to the
cluster: what the papers actually say, and the words to search by.

- **Neither list can do the other's job.** Phrases can only quote, so an abstract theme that no paper names in so many
  words is beyond them; an LLM abstracts, and in doing so leaves out the specific names a reader searches for. A
  single merged list of six had no room for both, which is why there are two lines.
- **The Themes are the line read closely, and the Quoted the line consulted for specifics.** So a weak Themes line
  costs a cluster its gist, while a weak Quoted line costs a few search hooks, and effort goes where that says.


## Phrases: what counts as one

`extract`, per record, on a spaCy parse of its title and abstract.

- **A candidate is a noun phrase with its head at the end, and every head-final part of it** ("large language
  model", "language model", "model"), up to four words. The long form rarely repeats across records where the
  shorter ones do, so which length is worth showing is left to the ranking.
- **Determiners, pronouns, numbers and the like are stripped from the ends, and so is a leading stopword.** A
  stopword at the end stays when a noun qualifies it ("tool use") and goes when an adjective does ("remarkable
  ability").
  - **Except a word joined to the next by a hyphen**, which is the first half of a word rather than a word: stripping
    "test" from "test-time compute" left "time compute".
- **Punctuation inside a noun chunk, other than a hyphen, separates two phrases**, the head's side kept: "large
  language models (LLMs" is two.
- **A part starting with a stopword or with no letter in it is skipped**: "and photo-fermentation", "2.0 license".
- **A possessive stays on its word**: "prisoner's dilemma". Dropping it made "prisoner dilemma", a phrase nobody
  writes.
- **The head is lemmatized, with two exceptions the small English pipeline gets wrong**: "data" (which it makes
  "datum") and any word in "-ics", which is either a singular field ("physics", made "physic") or a plural
  ("characteristics"); the corpus-wide step below decides which.
- **The title is sentence-cased before parsing** when it is in Title Case or ALL CAPS, since a tagger reads a
  capitalized word as a proper noun and leaves it unlemmatized. In a Title Case title, a word with a capital past its
  first letter keeps it ("CO2", "TiO2", "LoRA"); ALL CAPS gives no such hint and is lowered throughout.
- **Text is NFKC-normalized first**, so "TiO₂" and "TiO2" are one spelling.
- **Each phrase is also kept as written**, its capitals and its punctuation as the paper has them, for the rules below
  that need to know how a word was spelled. A capital that only starts a sentence is not counted as written.

**Abbreviations** are found by the algorithm of Schwartz and Hearst (2003): in "long form (SF)", the short form is a
token of 2–10 characters, and the long form the shortest run of words before it in which every character of the
short form occurs in order.

- **The short form must have two capitals or a digit** ("SRL", "ChatGPT", "TiO2"). Without that, "the gas evolved at
  the anode (oxygen)" defines the word "oxygen".
- **The long form is the text as written**, commas on their words: "verification, validation, and uncertainty
  quantification". Rebuilt from tokens, it read "verification , validation , ...".


## Phrases: when two spellings are one term

`normalize`, over the whole corpus. Each fold happens only where the corpus contains both forms, which is what keeps
a word that merely looks like a variant ("process", which ends in "s") as it is.

- **A plural onto its singular.**
- **A final word in "-ics" onto "-ic", where the "-ic" form heads a phrase somewhere in the corpus** ("a key
  characteristic"), and never for a field of study ("kinetics", "dynamics", "mechanics", "economics" and the rest
  of a short list).
  - **Heading a phrase, rather than merely appearing as a noun**: a stray noun tag on "mechanic" or "robotic" was
    enough to fold "mechanics" and "robotics", which mean something else.
  - **A list for the fields, because no count tells them apart**: a field's "-ic" form is genuinely written as a head
    ("first-order kinetic", a corpus's own slip), as often in proportion as a real plural's is. The list is a
    judgement, not a measurement.
- **An abbreviation with its long form, word by word, where the word is written as the short form is.** So "SRL" is
  "self-regulated learning", and "Co" stays cobalt in a corpus that defines "CO" as carbon monoxide. Matched without
  case, element symbols merged into whatever the corpus defined: cobalt into carbon monoxide, iron into faradaic
  efficiency.
  - **A definition made by two records or more applies corpus-wide; one made by a single record stays in it**, so a
    paper calling its system CARE does not take over the word "care" everywhere else.
  - **An abbreviation displays as its short form when most records using it leave it unexplained**, its readers
    evidently knowing it ("ChatGPT"), and spelled out otherwise ("fast weight programmer" for "FWP").
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
  and "LoRA" keep their capitals, and a word some papers capitalize for emphasis does not.
- **Votes are counted in the abstract only.** A title is sentence-cased before parsing, which lowercases every
  proper name in a Title Case title and outvoted the abstracts.
- **Every occurrence of a word counts, whatever its part of speech there**, once the word appears as a noun or
  adjective somewhere. Counting nouns and adjectives alone handed the vote to a capitalized compound part, tagged a
  proper noun ("AI-Based"), over the same word as a verb ("based on"), and the display came out "simulation-Based".
- **The LLM's keywords are recased the same way**, and two of its habits are undone: a keyword in Title Case
  throughout, and a keyword capitalized only on its first letter, as a sentence starts ("A posteriori error
  estimation"). A word the corpus capitalizes keeps its capital either way. This last rule is the brittle one: a
  proper name the casing table missed loses its capital.


## Phrases: ranking, and the label

- **Class-based TF-IDF over the clusters, on record counts** (Grootendorst 2022, BERTopic), so that a phrase's score
  also gives the share of the cluster's records that use it.
- **The first phrase is the cluster's best**, saying what it is about; in a corpus on one subject this is often a
  phrase other clusters share, and that is allowed there.
- **The rest say how the cluster stands out**: the ranking with the corpus-wide phrases removed, ordered once by rank
  and once by similarity to the cluster's centroid in the embedding space (after KeyBERT, Grootendorst 2020), the two
  orders fused by reciprocal rank (Cormack, Clarke and Buettcher 2009).
  - **Two orders, because each fails alone, in opposite places**: the ranking fills an abstract cluster's label with
    the ordinary words most of it uses, and the similarity reaches for narrow terms in a technical one. A phrase high
    in both survives the fusion.
- **A phrase is corpus-wide when it is in at least a tenth of the clusters, and in no fewer than five** where there
  are five. With a handful of clusters the floor of five dominates, so in a small corpus only a phrase in about half
  the clusters counts; see the display's ordering below for what covers the rest.
- **Two phrases nested in each other count as one**, and the longer replaces the shorter when it covers at least half
  as much of the cluster: every record with "latin hypercube sampling" also has "sampling", so ranking alone would
  keep the bare word and lose the name.


## Themes: the LLM's keywords

`raven.visualizer.importer`, with `clusters_keyword_method` "llm" or "hybrid".

- **The model reads the cluster's titles and abstracts, a sample of them in a large cluster** spread from its centre
  outwards, **and, after them, the cluster's phrases with the share of records using each**, as evidence of what the
  whole cluster contains. The prompt says the phrases are data, not suggestions, so that it does not copy them.
- **It is asked for up to fifteen keywords.** The label shows six and the dataset keeps them all; a model asked for
  more adds its most specific ones after the general.
- **The format rules come last in the prompt**, after the papers and the evidence, where a reader setting up the task
  expects them. One instruction, stated once.
- **A reply written as a Markdown list is read as one**, with a warning in the log. A good answer formatted as bullets
  was rejected three times over and recorded as no theme at all. A bulleted *summary* is still rejected: its items
  are sentences, ending in a full stop.
- **So is a reply that states its theme first and then lists the keywords under a heading** ("Keywords:",
  "Suggested keywords:"), on the heading's line or below it. Prose with no such heading is still rejected.
- **"keyword extraction failed" is a valid answer**, recorded as no theme and not retried, since asking again only
  pressures the model to invent one. A reply that is no list at all is retried, up to three times.
- **A second pass folds variants across clusters** (an acronym in one, its expansion in another), asked for as a
  mapping, so that only replacements by keywords the first pass actually produced are kept.
- **Answers are saved by prompt**, so a re-import asks the model only what it has not already answered.


## The display

`raven.visualizer.entry_renderer.format_cluster_keywords`.

- **Themes: the LLM's keywords without those it gave too many clusters**, by the same corpus-wide test as the
  phrases, **with the terms other clusters share moved after the cluster's own, except the first** (`shared_last`).
  The first stays where the LLM put it, so every cluster keeps its lead theme; the others make room for what tells
  this cluster apart. Ordering purely by rarity dropped each cluster's core theme instead.
- **Quoted: the phrases, without one that repeats a theme**, compared in both their stored and displayed forms, so an
  abbreviation counts as a repeat of its long form. Only exact repeats are left out, not phrases containing a theme:
  a one-word theme would otherwise take every phrase with that word in it.
- **An empty Quoted line is left out**, as a small cluster may have no phrase left that a theme does not say.
- **A cluster the LLM found no theme in shows its Quoted line alone**, not the label "unknown topic".
- **A tooltip over more than two clusters shows the Themes alone**, as two lines per cluster crowd the titles off it.
- **Fewer than six is fine.** A small cluster's phrases thin out into generic words; nothing reasonable fixes that,
  and padding would only add noise.


## Considered and rejected

- **One merged list**, the LLM's keywords leading and the phrases filling up: six slots could not hold both jobs.
- **Phrases as whole sentences**, and terms embedded by their sentences rather than on their own (2026-11-24; the
  reasons are in `investigations/cluster-keywords/README.md`).
- **A word-count limit on a bulleted reply's items**: a number with nothing behind it, where the header and the full
  stops already tell a summary from a list.
- **Lowering the corpus-wide floor for small corpora**: a single number, easy to fit to the corpus in front of us,
  and it removes shared themes outright where ordering only moves them down.
