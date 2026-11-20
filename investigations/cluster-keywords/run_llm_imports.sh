#!/bin/bash
# Import each test corpus with LLM cluster keywords, for comparing them with the classical keyworder.
#
# One corpus at a time, smallest first. A corpus whose dataset already exists is skipped, so running this
# again resumes after the last one finished. Datasets and logs go to data/ (gitignored: they hold corpus
# text). The LLM backend and keyword method come from the machine's configuration, which must say
# `clusters_keyword_method = "llm"`; `--llm required` stops a run whose backend does not answer.
#
# Run from the repository root:  bash investigations/cluster-keywords/run_llm_imports.sh
set -u
OUT=investigations/cluster-keywords/data
mkdir -p "$OUT"

run() {  # name, then the BibTeX inputs
    local name=$1; shift
    if [ -f "$OUT/$name.pickle" ]; then
        echo "$(date +%T) $name: already imported, skipping"
        return
    fi
    echo "$(date +%T) $name: importing"
    if raven-importer --clustering agglomerative --llm required --log "$OUT/$name.log" "$OUT/$name.pickle" "$@" \
            > /dev/null 2>&1; then
        echo "$(date +%T) $name: done"
    else
        echo "$(date +%T) $name: FAILED, see $OUT/$name.log"
    fi
}

R=00_stuff/rawdata
run biology   $R/biology_geography_health_education/202508/biology.bib
run geography $R/biology_geography_health_education/202512/geography.bib
run eccomas   00_stuff/datasets/ECCOMAS2024/success_final_manually_fixed_and_added_missing_abstracts.bib
run aokk      $R/AOKK/scopebib-2026-11-full/run-without-retracted/kept-and-held.bib
run hydrogen  $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs.bib \
              $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs-2.bib \
              $R/100000_most_relevant_refs_of_hydrogen_productionzip/savedrecs-3.bib
