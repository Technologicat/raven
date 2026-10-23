#!/usr/bin/env python
"""How the agglomerative tree relates to a human grouping, on a corpus of a hundred or two records.

The rework cuts the tree at 100, which on a corpus of ~150 leaves about 1.5 records per cluster. This
sweeps the cut level and scores each against a human-checked grouping, at both of the levels such a
grouping has — its clusters, and the named items within them — and also scores the shipped 2D labelling
the human grouping was made from, when that dataset is given.

The human grouping is a reference to compare against, not a target: there is no correct grouping, so a
cut level that maximizes agreement with it describes that grouping rather than the corpus.

Two reference formats, read by file extension:

  `.txt`  the Visualizer-export layout: `#N` opens a cluster, unindented lines without a year are item
          headings, and records are `Author (year): Title`. Matched to the dataset by title.
  `.tsv`  `cluster<TAB>item<TAB>first-author surname<TAB>year`, for a reference that names records
          only by citation, as a published table does. Matched by surname and year; a citation matching
          several records takes them all, which is harmless when they share a cluster and is reported
          when they do not.

A record the reference places in two clusters is dropped from the scoring, being in neither.

    python investigations/highdim-clustering/small_corpora.py --vectors V.npz --dataset D.pickle \\
        --reference R.txt [--baseline-dataset OLD.pickle]
"""

import argparse
import collections
import re
import sys
import unicodedata

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import clusterlab


def fold(text):
    """Casefold and strip diacritics and punctuation, for matching names and titles written by hand."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def load_entries(pickle_path):
    """`(title, first-author surname, year, cluster_id)` per entry, in dataset order."""
    import pickle
    with open(pickle_path, "rb") as f:
        dataset = pickle.load(f)
    out = []
    for entry in dataset["vis_data"]:
        names = entry.bibtex_author
        surname = " ".join(names[0].von + names[0].last) if names else ""
        out.append((entry.title, surname, str(entry.year), int(entry.cluster_id)))
    return out


def load_reference(path):
    """Returns a `list` of `(key, cluster, item)`, `key` being `("title", t)` or `("cite", surname, year)`."""
    rows = []
    if path.endswith(".tsv"):
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cluster, item, surname, year = line.rstrip("\n").split("\t")
                rows.append((("cite", fold(surname), year), int(cluster), item))
        return rows
    cluster, item = None, None
    record = re.compile(r"^(.*?)\s\((\d{4})\):\s(.+)$")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            if m := re.match(r"^#(\d+)$", line):
                cluster, item = int(m.group(1)), None
            elif m := record.match(line):
                rows.append((("title", fold(m.group(3))), cluster, item))
            else:
                item = line
    return rows


def match(entries, reference):
    """Map dataset indices to `(cluster, item)`. Returns `(mapping, unmatched_rows, conflicted_indices)`."""
    by_title = collections.defaultdict(list)
    by_cite = collections.defaultdict(list)
    for idx, (title, surname, year, _cid) in enumerate(entries):
        by_title[fold(title)].append(idx)
        by_cite[(fold(surname), year)].append(idx)
        # A hand-written citation may give only the last word of a compound surname.
        by_cite[(fold(surname).split(" ")[-1], year)].append(idx)
    assigned = collections.defaultdict(set)
    unmatched = []
    for key, cluster, item in reference:
        hits = by_title[key[1]] if key[0] == "title" else by_cite[(key[1], key[2])]
        if not hits:
            unmatched.append(key)
        for idx in set(hits):
            assigned[idx].add((cluster, item))
    mapping, conflicted = {}, []
    for idx, places in assigned.items():
        if len({c for c, _ in places}) > 1:
            conflicted.append(idx)
        else:
            mapping[idx] = sorted(places, key=str)[0]
    return mapping, unmatched, conflicted


def score(predicted, mapping):
    """ARI and AMI against the reference's clusters and items, over the matched records.

    Predicted outliers (-1) each count as a cluster of their own, so declining to place a record is
    scored as disagreeing about it rather than ignored.
    """
    idxs = sorted(mapping)
    pred = [int(predicted[i]) if predicted[i] >= 0 else -1000 - i for i in idxs]
    top = [mapping[i][0] for i in idxs]
    sub = [f"{mapping[i][0]}/{mapping[i][1]}" for i in idxs]
    return (adjusted_rand_score(top, pred), adjusted_mutual_info_score(top, pred),
            adjusted_rand_score(sub, pred), adjusted_mutual_info_score(sub, pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--dataset", required=True, help="dataset aligned with --vectors, for titles and citations")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline-dataset", help="the dataset the reference was made from, to score its 2D labels")
    parser.add_argument("--cuts", default="2-40,50,60,80,100")
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    original = clusterlab.normalize(vectors)
    fit_vectors = clusterlab.center(original)
    entries = load_entries(args.dataset)
    if len(entries) != len(original):
        sys.exit(f"dataset has {len(entries)} entries, vectors {len(original)}: not aligned")
    reference = load_reference(args.reference)
    mapping, unmatched, conflicted = match(entries, reference)
    ref_clusters = collections.Counter(c for c, _ in mapping.values())
    ref_items = collections.Counter(mapping.values())
    print(f"{len(original)} records, embedded by {model_name}, cache dtype {vectors.dtype}")
    print(f"reference: {len(reference)} rows; {len(mapping)} records matched into {len(ref_clusters)} clusters "
          f"and {len(ref_items)} items; {len(unmatched)} rows unmatched, {len(conflicted)} records in two clusters")
    for key in unmatched:
        print(f"    unmatched: {key}")
    print(f"reference cluster sizes: {sorted(ref_clusters.values(), reverse=True)}")
    print(f"reference item sizes:    {sorted(ref_items.values(), reverse=True)}\n")

    if args.baseline_dataset:
        base = load_entries(args.baseline_dataset)
        base_mapping, base_unmatched, _ = match(base, reference)
        base_labels = np.array([cid for *_, cid in base])
        ari, ami, ari_s, ami_s = score(base_labels, base_mapping)
        print(f"shipped 2D labels of {args.baseline_dataset}: {len(set(base_labels) - {-1})} clusters, "
              f"{int(np.sum(base_labels == -1))} outliers of {len(base)}; {len(base_mapping)} matched")
        print(f"    against clusters ARI {ari:.3f} AMI {ami:.3f}; against items ARI {ari_s:.3f} AMI {ami_s:.3f}\n")

    cuts = []
    for part in args.cuts.split(","):
        lo, _, hi = part.partition("-")
        cuts.extend(range(int(lo), int(hi or lo) + 1))
    print("cut   raw  >=2  kept  coverage  floor    clusters: ARI   AMI     items: ARI   AMI    sizes kept")
    for k in cuts:
        if k >= len(original):
            break
        raw = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(fit_vectors)
        sizes = np.bincount(raw)
        labels, floor = clusterlab.filter_by_cohesion(original, raw)
        ari, ami, ari_s, ami_s = score(labels, mapping)
        kept_sizes = sorted(np.bincount(labels[labels >= 0]).tolist(), reverse=True)
        print(f"{k:3d}  {k:4d} {int(np.sum(sizes >= 2)):4d} {labels.max() + 1:5d}   {np.mean(labels >= 0):6.1%}  "
              f"{floor:.3f}      {ari:+.3f} {ami:+.3f}        {ari_s:+.3f} {ami_s:+.3f}   {kept_sizes[:12]}")


if __name__ == "__main__":
    main()
