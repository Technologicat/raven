# What spaCy's lemmatizer makes of a glossary's terms

**Question:** the lorebook is to match a term on its lemma as well as its surface form, so that inflected uses
fire it (decided 2026-11-16, in `briefs/librarian-extension/done/05_librarian-lorebook-brief.md`). Before building
that: what does the lemmatizer do with *coinages* — words it has never seen — and with hyphenated and
possessive terms? Does lemma matching add real matches, and does it add wrong ones?

**Apparatus:** `probe_lemmas.py`, run against the substrate-independent glossary (90 entries) and the documents
that use its terms (`~/.claude/CLAUDE.md` and substrate-independent's Markdown, 9 files), with the pipeline Raven
configures (`en_core_web_sm` 3.8.0, spaCy 3.8.18), loaded in-process — the same pipeline the server loads.
`results-en_core_web_sm.txt` is its output from 2026-11-17.

## Findings (English, 2026-11-17)

- **Regular inflections of coinages lemmatize correctly**: *fups → fup*, *duolects → duolect*, *prequisites*,
  *cognized/cognizing → cognize*, *fracters*, *sporks*, *botwalls*, *questinos*, *jazzed → jazz*.
- **Some are wrong, and in this sample every error is a miss rather than a wrong match**: *fupped → "fuppe"*,
  *coaxolotls* left unlemmatized, and **"hydra taxes" → "hydra taxis"** — the lemmatizer picks the wrong
  plural of *tax*. With lemma-*or*-surface matching each of these simply does not fire, as it would not have
  without lemmas.
- **spaCy's tokens are not the matcher's words.** An infix hyphen splits a word in three (`co - pter`,
  `aria - worthy`, `hedge - hog`), and a possessive in two (`chekhov` + `'s`); a suspended hyphen is kept
  (`co-`). So lemma matching cannot run on spaCy's tokens directly: the matcher's own segmentation has to stay,
  with each word's lemma built from the spaCy tokens inside its span — the last part lemmatized, English
  inflecting a compound's head (*co-pters → co-pter*, *hedge-hogs → hedge-hog*), and a trailing possessive
  dropped (*duolect's → duolect*).
- **Lemmatizing lowercased text loses smart case.** `hybridir.keyword_tokens` lowercases first, so on that path
  `EQ` would match "eq". An exact key word has to match as written; only a forgiving one may also match through
  its lemma.
- **A few titles lemmatize oddly**: *Ding agent → "de agent"*, *Barthes mode → "barthe mode"*, *AI heating →
  "ai heat"*. On the title and the text alike, so the matcher compares like with like; noted in case one fires
  where it should not.
- **In real English prose the gain is small**: across the 9 documents, lemmas add one match the words miss
  ("confabulations"). These terms are seldom inflected in English, so the payoff is expected in Finnish, which
  this probe cannot yet test — only `en_core_web_sm` is installed.

## Voikko on Finnish (2026-11-19)

`probe_voikko.py` asks Voikko's analyser (`libvoikko1` 4.3.2, `voikko-fi` 2.5, the system packages) through
`ctypes`, so it needs no Python binding. `results-voikko.txt` is its output.

- **Perfect on Finnish words, however agglutinated**: *juoksentelisinkohan → juoksennella*, *kissoillammekin →
  kissa*, *verottajalta → verottaja*, and compounds of dictionary words (*kaksikielisyysongelmassa*).
- **Nothing at all for a word outside its dictionary** — no analysis, and the spellchecker rejects it: the
  Finnish-shaped coinages (*duolektissa*, *hydraveron*, *hydra* being unknown to it) and the glossary's English
  terms inflected as Finnish speakers inflect loans (*duolectissa*, *co-pterin*, *sporkilla*, *botwallin*).
- **Its wordlist is incomplete, as wordlists are**: *hydra*, a word in ordinary Finnish use, is unknown to it
  on its own, capitalized or not. So a word it does not know is not thereby a coinage.
- So it cannot lemmatize a lorebook's terms, which are coinages by nature, and it has no generation call to
  produce their forms either.

## How many unrelated words the generated stems catch (2026-11-20)

**Question:** with Finnish switched on, the lorebook asks the LLM for stems, and a stem fires its term on any
word it begins. So how many words that are *not* forms of the term does each generated stem catch — in Finnish,
and in English, since the matcher does not know a message's language? The stem guard in
`lorebook.clean_expansion` (all but the last two letters of the word, and at least four) is to be judged by
this, and a looser one considered (the brief's option (b), for Finnish verb stem changes).

**Apparatus:** `measure_stem_false_positives.py` reads an expansion ledger (`raven.librarian.lorebook_expansion`),
applies the guard to every Finnish stem, and counts the words of a word list each would match, other than the
term's own forms. Word lists: `/usr/share/dict/words` and WordNet's `index.*` for English; for Finnish,
`data/nykysuomensanalista2024.txt` (below).

**Voikko could not be the Finnish word list**, as planned: `voikko-fi` ships only its compiled analyser
(`mor.vfst`), which answers about a word it is given and cannot list its vocabulary.

### The Finnish word list

`data/nykysuomensanalista2024.txt` is **Nykysuomen sanalista 2024**, by Kotimaisten kielten keskus (Kotus),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), downloaded 2026-11-20 from
<https://kaino.kotus.fi/lataa/nykysuomensanalista2024.txt> (the file dated 11.4.2025 on
[its page](https://kotus.fi/sanakirjat/kielitoimiston-sanakirja/nykysuomen-sana-aineistot/nykysuomen-sanalista/)).
Unmodified. Tab-separated, UTF-8: headword, homonym number, word class, inflection type; about 105,000
headwords, base forms only.

### Findings (2026-11-20, Qwen Flash Next via gufo, `{"English": "forms", "Finnish": "stems"}`)

The ledger measured: all 91 glossary terms answered in 8 min 17 s, no failures (English alone, an earlier run:
2 min 26 s). `results-stems-english-lists.txt` and `results-stems-finnish-list.txt` are the probe's output.

- **63 stems pass the guard.** 23 of them catch an English word that is not one of their term's forms, 11 a
  Finnish headword. The worst: `hydr` 127 English words (*hydrangea*, *hydrant*, *hydrate*) and 17 Finnish
  (*hydraulinen*); `mode` and `flat` 47 English each (*model*, *flatbed*); `anim` 37 (*animal*); `lint` 40 Finnish
  (*lintu*, a bird). **Every bad catch comes from a stem of four or five letters.**
- **The counts overstate the harm, in two ways.** They are per *word*, while a multi-word term fires only with its
  other words beside it — *mode* catching *model* matters in "little prince model", which nobody writes — so the
  exposure is the single-word terms. And a Finnish compound beginning with the term is often a true match
  (*jazz* → *jazzmusiikki*, *agent* → *agentti*), which the probe counts as false.
- **So the data argues against loosening the guard** (option (b) admits more short stems) and, if anything, for
  a higher floor. **The floor was raised to five** the same day, which drops the four-letter stems (*hydr*,
  *mode*, *flat*, *anim*, *lint*, *jazz*), the worst of the catches; *space* (19 English words) and *latin* (13)
  remain.
- **The model also translates, unasked**: *rent* → *vuokran*, *vuokrassa*, …; *landmine* → *maamiina*, … — each
  word translated, then inflected, under its forms.
- **Its Finnish has the errors the maintainer found**: *hydryt*, and wrong vowel harmony in *hydrissä*,
  *hydristä*, *hydrillä*. A form nobody writes costs nothing; the stems are where an error matters.
- **The English forms are mostly right, with one bad one in two runs**: *fup* → *fapped*, *fapping* in the
  first run (a mangled form, and a vulgar word), *fups*, *fup's* in the second.

### Measured again (2026-11-23), after the floor of five and the prompt forbidding translations

The same probe on a ledger asked that morning, same model and languages; `results-stems-*-2026-11-23.txt`.
81 of the 91 terms are in it: one batch of ten was lost to a reply that looped on one Finnish form and never
closed, which is what `lorebook_expansion.salvage` and the retries now handle.

| | stems kept | catch an English word | catch a Finnish headword |
|---|---|---|---|
| 2026-11-20 (floor four, translations allowed) | 63 | 23 | 11 |
| 2026-11-23 (floor five, no translations) | 51 | 7 | 12 |

- **English improved most**: the four-letter stems are gone. The worst left is `prose`, 21 words (*prosecute*,
  *proselyte*), which matters only beside *basilisks*.
- **The Finnish catches are mostly genuine loan compounds now**: *agentti*, *profiili*, *gratiini*, *jukeboksi*,
  *designer*. The real false positives are `prose` → *prosentti*, *proseduuri* and their compounds, 25.
- **A stem can run past its word into another one**: `prosess`, given for *prose*, is the Finnish for *process*
  (*prose* is *proosa*). The guard passes it, since it begins with the whole of *prose*. A limit on how far a
  stem may extend past its word would catch it, but loans genuinely extend (*Gratin* → `gratiin`, *jukebox* →
  `jukeboks`), and no clean bound is in sight; recorded rather than guarded.
