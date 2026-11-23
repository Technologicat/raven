#!/bin/bash
# Import the test corpora with LLM cluster keywords given the phrase evidence, for comparison with the
# imports `run_llm_imports.sh` made without it.
#
# arXiv runs twice: with the evidence, and again without it. The LLM samples, so two runs of one prompt
# differ already; the two without-evidence runs of arXiv measure that floor, against which the evidence's
# effect is read.
#
# One corpus at a time; a dataset that already exists is skipped, so running this again resumes. Datasets
# and logs go to data/ (gitignored). Run from the repository root:
#
#     bash investigations/cluster-keywords/run_llm_evidence_imports.sh
set -u
OUT=investigations/cluster-keywords/data
mkdir -p "$OUT"

run() {  # name, evidence (on/off), then the BibTeX inputs
    local name=$1 evidence=$2; shift 2
    if [ -f "$OUT/$name.pickle" ]; then
        echo "$(date +%T) $name: already imported, skipping"
        return
    fi
    echo "$(date +%T) $name: importing (evidence $evidence)"
    if python investigations/cluster-keywords/import_with_evidence.py "$evidence" \
            --clustering agglomerative --llm required --log "$OUT/$name.log" "$OUT/$name.pickle" "$@" \
            > /dev/null 2>&1; then
        echo "$(date +%T) $name: done"
    else
        echo "$(date +%T) $name: FAILED, see $OUT/$name.log"
    fi
}

R=00_stuff/rawdata
ARXIV=00_stuff/datasets/ai_papers/ai_papers_202611.bib
run arxiv-evidence    on  $ARXIV
run arxiv-repeat      off $ARXIV
run biology-evidence  on  $R/biology_geography_health_education/202508/biology.bib
run geography-evidence on $R/biology_geography_health_education/202512/geography.bib
run eccomas-evidence  on  00_stuff/datasets/ECCOMAS2024/success_final_manually_fixed_and_added_missing_abstracts.bib
run aokk-evidence     on  $R/AOKK/scopebib-2026-11-full/run-without-retracted/kept-and-held.bib
run hydrogen-evidence on  $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs.bib \
                          $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs-2.bib \
                          $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs-3.bib
