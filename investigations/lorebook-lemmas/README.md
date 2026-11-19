# What spaCy's lemmatizer makes of a glossary's terms

**Question:** the lorebook is to match a term on its lemma as well as its surface form, so that inflected uses
fire it (decided 2026-11-16, in `briefs/librarian-extension/05_librarian-lorebook-brief.md`). Before building
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
