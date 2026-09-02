"""Paper management tools — arXiv search/download, bibliography converters, corpus hygiene.

Consolidates arXiv API support (search, download, ID parsing) and format
converters (WoS, CSV, PDF, BibTeX burst) that feed Raven's research pipeline.

Three of the tools operate on a bibliography you already have, and they ask
different questions of it. `fixbib` asks whether a record can be *read*, and
repairs the ones a parser refuses. `deduplicate` asks whether two records name
one study, and merges them. `siftbib` asks whether a record can be *used* — a
record with nothing to screen on is not off topic, it is unusable — and removes
the ones that cannot.

## A tool that removes records writes an audit; a tool that repairs them need not

The asymmetry is about what a reader can recover from the output alone, not
about how important the tools are.

A repair is visible: `fixbib` writes a new file beside the old one, so what it
changed is a diff away, and an audit would restate what the two files already
say between them. It reports to the console and offers `--list`, which is the
right amount.

A removal is not. Once a record is gone from the output, nothing in that output
says it ever existed, what it was, or why it went — and both `deduplicate` and
`siftbib` remove records. So each writes a TSV naming every record it took out
and the reason, with a header stamping the tool version, the inputs, and the
test applied. That header is what makes the file citable: a method section has
to say what came out of the corpus and on what grounds, and "the script said so"
is not a method section.

Both write it by default, and both take `--no-audit` to decline it.

The two differ in *command-line shape* rather than in this, and that difference
tracks their arity. `deduplicate` reads several files as one corpus, so it has a
single output and a single audit and names each by path (`-o`, `-a`). `fixbib`
and `siftbib` process each input separately, so they have as many outputs as
inputs and name them by suffix; where those files *land* is one `--out-dir` on
`siftbib`, covering the sifted bibliography and its audit together. Hence `-a`
is a path in `deduplicate` and does not exist in `siftbib` — one short flag
meaning two different kinds of thing across siblings is worse than no short
flag.
"""
