#!/bin/bash
for INFILE in "$@" ; do
    DIR=$(dirname "$INFILE")
    BASE=$(basename "$INFILE" .txt)
    OUTFILE="$DIR/$BASE.bib"
    python -m raven.papers.wos2bib "$INFILE" >"$OUTFILE"
done

