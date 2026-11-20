"""How many unrelated words each of the lorebook's generated stems would catch.

A stem fires its term on any word it begins, followed by letters (`raven.librarian.lorebook`), so a stem the
model got too short, or that happens to begin common words, fires the term where it should not. This counts,
for every stem in an expansion ledger, the words of a word list it would match. The word lists are data
neither the model nor we wrote, which is what makes the count a measurement rather than another example.

Run from the repository root:

    python investigations/lorebook-lemmas/measure_stem_false_positives.py [LEDGER.jsonl] [--wordlist PATH ...]

Without a ledger, takes the newest one under `~/.config/raven/librarian/lorebook_expansion/` that holds
Finnish answers. The word lists default to `/usr/share/dict/words` and WordNet's `index.*` files, both
English: Finnish stems apply to English text too, the matcher not knowing a message's language. Pass a
Finnish list (one word per line, e.g. Kotus's Nykysuomen sanalista) with `--wordlist` to count in Finnish.

Words the stem's own term gave as forms are not counted, nor the term word itself; everything else a stem
matches is counted as a word it should not have, which overcounts where a list holds a genuine inflected
form the model did not list. So the count is an upper bound, and the words are printed for reading.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from raven.librarian import lorebook  # noqa: E402 -- after the path setup, for running from the repository root

LEDGER_DIR = pathlib.Path.home() / ".config/raven/librarian/lorebook_expansion"
DEFAULT_LISTS = [pathlib.Path("/usr/share/dict/words"),
                 *sorted(pathlib.Path("/usr/share/wordnet").glob("index.*"))]

def read_wordlist(path: pathlib.Path) -> set[str]:
    words = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(" "):  # WordNet's index files open with an indented licence
            continue
        # WordNet's index: the lemma, then its data; Kotus's list: tab-separated, the headword first.
        word = line.split()[0] if path.parent.name == "wordnet" else line.split("\t")[0].strip()
        if path.parent.name == "wordnet" and "_" in word:
            continue  # a WordNet collocation: several words, which no single text word is
        words.add(word.lower())
    return words

def finnish_ledgers() -> list[pathlib.Path]:
    found = []
    for path in sorted(LEDGER_DIR.glob("expansions-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith("-traces.jsonl"):
            continue
        if '"Finnish"' in path.read_text(encoding="utf-8"):
            found.append(path)
    return found

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ledger", nargs="?", type=pathlib.Path)
    parser.add_argument("--wordlist", action="append", type=pathlib.Path)
    args = parser.parse_args()

    maybe_ledger = args.ledger or next(iter(finnish_ledgers()), None)
    if maybe_ledger is None:
        sys.exit(f"no ledger with Finnish answers under {LEDGER_DIR}")
    lists = args.wordlist or DEFAULT_LISTS
    vocabulary: set[str] = set()
    for path in lists:
        vocabulary |= read_wordlist(path)
    # A control: the instrument has to be able to see a match, or a count of zero means nothing.
    assert any(word.startswith("hydr") for word in vocabulary), "the word lists hold no word starting 'hydr'"
    print(f"ledger: {maybe_ledger.name}; word lists: {', '.join(str(p) for p in lists)} ({len(vocabulary)} words)")

    answers = {}
    for line in maybe_ledger.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            answers[record["id"]] = record["answer"]

    rows = []
    for term, by_word in answers.items():
        for word, by_language in by_word.items():
            given = by_language.get("Finnish", {})
            own = {form.lower() for language in by_language.values() for form in language.get("forms", [])}
            own.add(word.lower())
            expansion = lorebook.clean_expansion(word, given.get("forms", []), given.get("stems", []))
            for stem in expansion.stems:
                # The stem's own rule from `lorebook._accepts`, lowercase: the stem, then letters or nothing.
                caught = sorted(w for w in vocabulary
                                if w.startswith(stem.lower()) and (w == stem.lower() or w[len(stem):].isalpha())
                                and w not in own)
                rows.append((len(caught), term, word, stem, caught))
    rows.sort(reverse=True)
    stems = len(rows)
    catching = sum(1 for row in rows if row[0])
    print(f"{stems} stems kept by the guard; {catching} catch at least one listed word that is not their own form")
    for count, term, word, stem, caught in rows:
        if count:
            shown = ", ".join(caught[:12]) + (f", … (+{count - 12})" if count > 12 else "")
            print(f"  {count:5d}  {stem!r:18} ({word} in {term!r}): {shown}")

if __name__ == "__main__":
    main()
