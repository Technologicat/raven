"""What spaCy's lemmatizer makes of a glossary's terms: the coinages, the hyphenated ones, and their inflected forms.

Run from the repository root:

    python investigations/lorebook-lemmas/probe_lemmas.py [GLOSSARY] [CORPUS_FILE ...]

GLOSSARY defaults to the substrate-independent glossary, and the corpus to the documents that use its terms
(`~/.claude/CLAUDE.md` and substrate-independent's Markdown). Prints three parts:

  A. every title, as spaCy tokenizes and lemmatizes it, flagging the ones that do not come back word for word;
  B. a set of inflected forms of the coinages, with the lemma each gets;
  C. in the corpus, the occurrences whose lemmas match a title's while the words as written do not: what
     matching on lemmas would add to matching on words.

The pipeline is `raven.librarian.config.spacy_model`, loaded in-process, which is the one the server loads.
"""

import pathlib
import re
import sys

import spacy

from raven.librarian import config as librarian_config
from raven.librarian import lorebook

home = pathlib.Path.home()
glossary_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else home / "Documents/koodit/substrate-independent/glossary.md"
if len(sys.argv) > 2:
    corpus_paths = [pathlib.Path(arg) for arg in sys.argv[2:]]
else:
    corpus_paths = [home / ".claude/CLAUDE.md"] + [path for path in (home / "Documents/koodit/substrate-independent").rglob("*.md")
                                                   if path.name != "glossary.md" and "00_stuff" not in path.parts]

nlp = spacy.load(librarian_config.spacy_model)
print(f"pipeline: {nlp.meta['name']} {nlp.meta['version']} on spaCy {spacy.__version__}\n")

entries = lorebook.parse(glossary_path.read_text(encoding="utf-8"), name="glossary", source=glossary_path)

def analyse(text: str) -> list[tuple[str, str]]:
    """(token, lemma) pairs, of the lowercased text, as `hybridir.keyword_tokens` sees it."""
    return [(token.text, token.lemma_) for token in nlp(text.lower())]

print("A. Titles whose tokens or lemmas are not their words")
print("   (the matcher's words on the left; spaCy's token/lemma pairs on the right)\n")
plain = 0
for entry in entries:
    words = [match.group().lower() for match in re.finditer(r"\w+(?:['-]\w+)*(?:-(?!\w))?", entry.title)]
    pairs = analyse(entry.title)
    if [text for text, _ in pairs] == words and all(text == lemma for text, lemma in pairs):
        plain += 1
        continue
    print(f"   {entry.title!r:45s} {words}  ->  {pairs}")
print(f"\n   {plain} of {len(entries)} titles come back word for word, each word its own lemma.\n")

print("B. Inflected forms of the coinages")
forms = ["fup", "fups", "fupped", "fuppable", "duolect", "duolects", "duolect's",
         "co-pter", "co-pters", "the co- prefix", "prequisite", "prequisites", "cognize", "cognized", "cognizing",
         "coaxolotl", "coaxolotls", "fracter", "fracters", "hydra tax", "hydra taxes", "hedge-hog", "hedge-hogs",
         "spork", "sporks", "botwall", "botwalls", "endognosis", "questino", "questinos", "aria-worthy design",
         "aria-worthy designs", "maybe-nature", "jazz", "jazzed", "loophole", "loopholes"]
for form in forms:
    print(f"   {form!r:24s} -> {analyse(form)}")
print()

print("C. In the corpus: occurrences lemmas would match and words do not")
title_lemmas = {entry.id: [lemma for _, lemma in analyse(entry.title) if any(c.isalnum() for c in lemma)] for entry in entries}
title_lemmas = {entry_id: lemmas for entry_id, lemmas in title_lemmas.items() if lemmas}
added: dict[str, set[str]] = {}
for path in corpus_paths:
    for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace")):
        by_words = lorebook.find_matches({entry.id: entry for entry in entries}, paragraph)
        doc = nlp(paragraph.lower())
        tokens = [(token.text, token.lemma_) for token in doc]
        lemmas = [lemma for _, lemma in tokens]
        for entry_id, wanted in title_lemmas.items():
            if entry_id in by_words:
                continue
            n = len(wanted)
            for start in range(len(lemmas) - n + 1):
                if sorted(lemmas[start:start + n]) == sorted(wanted):
                    added.setdefault(entry_id, set()).add(" ".join(text for text, _ in tokens[start:start + n]))
for entry_id, wordings in sorted(added.items()):
    print(f"   {entry_id:32s} {sorted(wordings)}")
print(f"\n   {len(added)} entries gain at least one match in {len(corpus_paths)} documents.")
