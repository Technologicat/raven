# Tokenizer migration: how long re-tokenizing a RAG index takes

**The question:** v0.2.10 changes the keyword-search tokenizer (`hybridir.keyword_tokens`), and an index
built by the old one is re-tokenized when it is first opened. How long does that take, and does each chunk
get its own tokens back?

Measured 2026-10-01 on the maintainer's development machine, with Raven-server running on the same machine
and doing the tokenizing (spaCy `en_core_web_sm`, spaCy 3.8.16).

## Scripts

| script | what it answers |
|---|---|
| `migrate_one.py <index dir>` | Opens one index, which migrates it, and prints the time taken and the rate per chunk. Opens it with `HybridIR` directly, so nothing reconciles it against a documents directory. |
| `migrate_all.sh` | Backs up each of the eight indexes under `~/.config/raven/librarian/`, then runs `migrate_one.py` on each, smallest first, logging to `migrate.log` beside the backups. |
| `check_alignment.py <index dir>` | Whether every chunk got its own tokens back: one token list per chunk, and most of each chunk's tokens found in that chunk's own text. Runs the same test on tokens shifted by one chunk as its negative control. |

## Batching per document against batching across documents

The first version of the migration tokenized one document at a time, in batches of up to 64 chunks. On
`rag_index_hydrogen_photocat`, an index of single BibTeX records averaging 2.8 chunks per document, that sent about
three chunks per request to Raven-server:

| batching | time | per chunk |
|---|---|---|
| per document | 219.7 s | 31.4 ms |
| across documents | 47.9 s | 6.8 ms |

So the request round trip, not the tokenizing, was most of the cost. The shipped migration batches across
documents. Ordinary indexing still batches per document; that is `TODO_DEFERRED.md`, "Indexing pays two
server round trips per document".

## The migration of all eight indexes

| index | chunks | time | per chunk |
|---|---|---|---|
| `rag_index_tmp` | 62 | 0.5 s | 8.9 ms |
| `rag_index_banichuk` | 542 | 2.0 s | 3.7 ms |
| `rag_index_arxiv` | 2 596 | 17.9 s | 6.9 ms |
| `rag_index_fiction` | 2 977 | 23.6 s | 7.9 ms |
| `rag_index_eccomas2024` | 6 607 | 42.2 s | 6.4 ms |
| `rag_index_hydrogen_photocat` | 6 995 | 46.8 s | 6.7 ms |
| `rag_index_hydrogen` | 31 600 | 214.2 s | 6.8 ms |
| `rag_index_arxiv_fulltext` | 159 383 | 1157.6 s | 7.3 ms |

26 minutes in all, nothing failed, and every index came out at tokenizer version 2 with one token list per
chunk in every document. The times include opening the index, which is negligible beside the tokenizing for
all but the smallest. The backups took 3.4 GB.

## Alignment

On `rag_index_hydrogen_photocat` after the migration, `check_alignment.py` found no document with a
mismatched token-list count, and 0 of 850 sampled chunks whose tokens mostly did not occur in their own
text. With the tokens shifted by one chunk, it flagged 779 of 850, so the check can tell an aligned index
from a misaligned one.
