#!/bin/bash
# Back up each RAG index, then migrate it to the current tokenizer by opening it. Smallest first, so a
# failure shows early. Each index is opened directly with `migrate_one.py`, never through `raven-indexer`,
# which would reconcile the index against whatever documents directory it was given.
set -u
HERE=$(dirname "$(readlink -f "$0")")
BASE=$HOME/.config/raven/librarian
BACKUP=$BASE/rag_index_backup_$(date +%F)
LOG=$BACKUP/migrate.log
mkdir -p "$BACKUP"
echo "start $(date -Is)" >> "$LOG"
for d in rag_index_tmp rag_index_banichuk rag_index_arxiv rag_index_fiction rag_index_eccomas2024 rag_index_hydrogen_photocat rag_index_hydrogen rag_index_arxiv_fulltext; do
    echo "=== $d: backing up $(date -Is)" >> "$LOG"
    cp -a "$BASE/$d" "$BACKUP/$d" || { echo "BACKUP FAILED for $d, skipping it" >> "$LOG"; continue; }
    echo "=== $d: migrating $(date -Is)" >> "$LOG"
    python "$HERE/migrate_one.py" "$BASE/$d" >> "$LOG" 2>&1 || echo "MIGRATION FAILED for $d (exit $?)" >> "$LOG"
done
echo "done $(date -Is)" >> "$LOG"
