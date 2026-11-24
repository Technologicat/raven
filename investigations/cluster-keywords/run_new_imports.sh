#!/bin/bash
# Import the test corpora again after the keyword changes of 2026-11-24 (hyphen and joined spellings, the
# corpus's capitals, the hybrid method without the LLM's corpus-wide terms): each corpus with LLM keywords given
# the phrase evidence, and with hybrid keywords, the evidence on in both.
#
# Resumable at the granularity of one import: a dataset that already exists is skipped, so stopping this (Ctrl+C,
# or killing it) loses only the import in progress, and running it again carries on. Extraction is cached per
# BibTeX file as soon as it is done, so even an interrupted import keeps that.
#
# The order is by priority, smallest and most read first, so a run cut short still leaves the useful part:
# arXiv, the two education corpora, AOKK, ECCOMAS, hydrogen.
#
# Before the first run, move the previous evidence datasets aside, or they are taken as done; see the bundle's
# README. Datasets and logs go to data/ (gitignored). Run from the repository root:
#
#     bash investigations/cluster-keywords/run_new_imports.sh
set -u
OUT=investigations/cluster-keywords/data
mkdir -p "$OUT"

run() {  # name, method (llm/hybrid), then the BibTeX inputs
    local name=$1 method=$2; shift 2
    if [ -f "$OUT/$name.pickle" ]; then
        echo "$(date +%T) $name: already imported, skipping"
        return
    fi
    echo "$(date +%T) $name: importing ($method, evidence on)"
    if python investigations/cluster-keywords/import_with_evidence.py --method "$method" on \
            --clustering agglomerative --llm required --log "$OUT/$name.log" "$OUT/$name.pickle" "$@" \
            > /dev/null 2>&1; then
        echo "$(date +%T) $name: done"
    else
        echo "$(date +%T) $name: FAILED, see $OUT/$name.log"
    fi
}

both() {  # corpus name, then the BibTeX inputs
    local name=$1; shift
    run "$name-evidence" llm "$@"
    run "$name-hybrid" hybrid "$@"
}

R=00_stuff/rawdata
H=$R/100000_most_relevant_refs_of_hydrogen_productionzip
both arxiv     00_stuff/datasets/ai_papers/ai_papers_202611.bib
both biology   $R/biology_geography_health_education/202508/biology.bib
both geography $R/biology_geography_health_education/202512/geography.bib
both aokk      $R/AOKK/scopebib-2026-11-full/run-without-retracted/kept-and-held.bib
both eccomas   00_stuff/datasets/ECCOMAS2024/success_final_manually_fixed_and_added_missing_abstracts.bib
both hydrogen  $H/savedrecs.bib $H/savedrecs-2.bib $H/savedrecs-3.bib
