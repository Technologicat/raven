#!/bin/bash
for INFILE in $@ ; do
    DIR=$(dirname "$INFILE")
    BASE=$(basename "$INFILE" .txt)
    OUTFILE="$DIR/$BASE.bib"
    python -m raven.import_wos "$INFILE" >"$OUTFILE"
done

