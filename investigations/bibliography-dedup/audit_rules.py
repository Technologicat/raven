"""List what each deduplication rule actually did to a bibliography, grouped so it can be read.

    python audit_rules.py path/to/search.bib [--limit 20]

This is the instrument that found both false merges in `raven-deduplicate`, and neither was visible in a
cluster count. The habit it exists to support: *a corpus tells you what a rule catches and nothing about
what else it catches*, so the check that works is not a number but a list of what a rule did, grouped by
whatever the rule keys on, read by a person.

Each section below is one place a rule can be wrong, and each is small enough to read end to end — the two
real defects were three rows in a list of thirty-six and one row in a list of one. Run it after changing
any rule in `raven.papers.deduplicate` and read the sections that rule touches.

Reads through `raven-fixbib`'s repair, exactly as the tool does, so the counts here are the tool's counts.
Writes nothing.
"""

import argparse
import collections
import logging
import pathlib
import sys

from raven.common import text as textutil
from raven.papers import deduplicate as dd


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe(record: dd.Record) -> str:
    return (f"    {record.key:42} {record.year}  {record.doi}\n"
            f"        {record.display_title[:96]!r}")


def report_shape(records: list[dd.Record], clusters: list[dd.Cluster]) -> None:
    """What the run did, in the aggregate. The numbers that are *not* the check, but frame it."""
    section("Shape of the run")
    merged = [cluster for cluster in clusters if len(cluster) > 1]
    print(f"  {len(records)} records -> {len(clusters)} clusters "
          f"({sum(len(c) - 1 for c in merged)} merged away from {len(merged)})")
    for rule, count in collections.Counter("+".join(c.rules) for c in merged).most_common():
        print(f"    matched by {rule:20} {count}")
    for size, count in sorted(collections.Counter(len(c) for c in merged).items()):
        print(f"    clusters of {size}: {count}")

    no_doi = sum(1 for r in records if r.doi is None)
    bad_doi = sum(1 for r in records if r.doi is None and r.field("doi"))
    print(f"    without a usable DOI: {no_doi} ({bad_doi} of them have a `doi` field that is not one)")
    print(f"    without an author:    {sum(1 for r in records if not r.surname)}")


def report_authorless(clusters: list[dd.Cluster], limit: int) -> None:
    """Merges where no record names an author — where the title carried the whole decision.

    The section that found the serial-heading bug. `II Political Science: Method and Theory` heads an item
    in every issue of its journal: authorless, same year, an ordinary title by every other test, and four
    issues merged into one record. Of 36 authorless merges in the corpus this was built on, the three whose
    DOIs disagreed were the three that were wrong.
    """
    section("Merges with no author anywhere — the title decided these alone")
    interesting = [c for c in clusters
                   if len(c) > 1 and not any(r.surname for r in c.records)]
    conflicted = [c for c in interesting
                  if len({r.doi for r in c.records if r.doi}) > 1]
    print(f"  {len(interesting)} authorless merges, {len(conflicted)} of which disagree about a DOI")
    print("  (a DOI disagreement here should be impossible — `_title_edge_holds` refuses it)")
    for cluster in interesting[:limit]:
        flag = "  <-- DOIs DISAGREE" if cluster in conflicted else ""
        print(f"  size {len(cluster)} by {'+'.join(cluster.rules)}{flag}")
        for record in cluster.records:
            print(describe(record))
    if len(interesting) > limit:
        print(f"  ... and {len(interesting) - limit} more; raise --limit to see them")


def report_refused_edges(records: list[dd.Record], limit: int) -> None:
    """Title matches the guards turned down — read these to see what the guards cost.

    Every refusal is a merge that did not happen, so a wrong one is a duplicate left in the output. Cheaper
    to check than a false merge, and this is the only place they are visible at all.
    """
    section("Title matches refused by the guards")
    groups = collections.defaultdict(list)
    for record in records:
        if record.title is not None:
            groups[record.title].append(record)

    refused = []
    for title, members in groups.items():
        for position, a in enumerate(members):
            for b in members[position + 1:]:
                if not dd._title_edge_holds(a, b):
                    refused.append((title, a, b))
    generic = [row for row in refused if dd.is_generic_title(row[0])]
    print(f"  {len(refused)} refused pairs, {len(generic)} on a generic title, "
          f"{len(refused) - len(generic)} on an ordinary one")
    for title, a, b in refused[:limit]:
        kind = "generic" if dd.is_generic_title(title) else "ordinary"
        print(f"  [{kind}] {a.key} vs {b.key}")
        print(describe(a))
        print(describe(b))
    if len(refused) > limit:
        print(f"  ... and {len(refused) - limit} more; raise --limit to see them")


def report_doi_conflicts(clusters: list[dd.Cluster], limit: int) -> None:
    """Merges whose records do not agree on a DOI — every one of these deserves an eye.

    Most are ordinary (a preprint beside its published version, a hyphen typed as an en-dash), and the
    ones that are not are the merges most likely to be wrong, because a title match made them over a
    contradiction the records were able to state.
    """
    section("Merges whose records disagree about the DOI")
    conflicts = dd.conflicting_clusters(clusters)
    settled = sum(1 for c in conflicts
                  for other in c.records[1:] if dd.settled_by_rule(c.records[0], other))
    print(f"  {len(conflicts)} clusters; {settled} pairs are Springer chapter versions, settled by rule")
    for cluster in conflicts[:limit]:
        print(f"  size {len(cluster)} by {'+'.join(cluster.rules)}")
        for record in cluster.records:
            print(describe(record))
    if len(conflicts) > limit:
        print(f"  ... and {len(conflicts) - limit} more; raise --limit to see them")


def report_nothing_disappears(clusters: list[dd.Cluster]) -> None:
    """The promise, checked exhaustively rather than on a fixture.

    Every field value in every merged-away record must be in the output or in the audit row. The one
    exemption is an abstract differing from the kept one only by its rights notice; that is counted
    separately rather than waved through, so a change to `_best_abstract` shows up here as a number moving.
    """
    section("Invariant: nothing disappears without a trace")
    exempt, missing, examples = 0, collections.Counter(), {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        entry, row = dd.merge_cluster(cluster)
        kept = {field.key: field.value for field in entry.fields}
        recorded = " | ".join(row.differences)
        for record in cluster.records:
            for field in record.entry.fields:
                value = dd._field_value(record.entry, field.key)
                if value is None or kept.get(field.key) == value:
                    continue
                if f"{field.key}: kept" in recorded:
                    continue
                if (field.key == "abstract"
                        and textutil.strip_boilerplate(value)
                        == textutil.strip_boilerplate(kept.get("abstract", ""))):
                    exempt += 1
                    continue
                missing[field.key] += 1
                examples.setdefault(field.key, (record.key, value[:110]))

    print(f"  abstracts differing only by a rights notice (exempt by design): {exempt}")
    print(f"  values neither kept nor recorded for any other reason: {sum(missing.values())}")
    for key, count in missing.most_common(10):
        print(f"    {key}: {count}   e.g. {examples[key]}")
    if not missing:
        print("  -> the invariant holds over this corpus")


def report_boilerplate_cuts(records: list[dd.Record], limit: int) -> None:
    """The largest things `strip_boilerplate` removed, longest first.

    Reading this list is what caught the stripper eating an abstract that *discussed* copyright. A
    collapse statistic cannot show it: the numbers look excellent either way, because a wrong cut and a
    right one are both cuts.
    """
    section("Largest boilerplate cuts — read the top of this list")
    cuts = []
    for record in records:
        raw = record.field("abstract")
        if not raw:
            continue
        stripped = textutil.strip_boilerplate(raw)
        if len(raw) != len(stripped):
            cuts.append((len(raw) - len(stripped), record, raw[len(stripped):]))
    cuts.sort(key=lambda row: -row[0])
    total = sum(row[0] for row in cuts)
    print(f"  {len(cuts)} abstracts cut, {total} characters in total")
    for size, record, removed in cuts[:limit]:
        print(f"  -{size:5}  {record.key}")
        print(f"        {' '.join(removed.split())[:150]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bibfile", help="the .bib to audit")
    parser.add_argument("--limit", type=int, default=12,
                        help="rows per section (default 12; every section is meant to be read)")
    opts = parser.parse_args()

    # This tool reports the same records itself, with more context than the parser's own warning carries.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    path = pathlib.Path(opts.bibfile).expanduser().resolve()
    records, unreadable = dd.read_records(path.read_text(encoding="utf-8"))
    if not records:
        print(f"{path}: nothing readable here.", file=sys.stderr)
        return 1
    clusters = dd.cluster_records(records)

    print(f"{path.name}: {len(records)} records read, {len(unreadable)} unreadable")
    for report in unreadable:
        print(f"    {report.describe()[:150]}")

    report_shape(records, clusters)
    report_authorless(clusters, opts.limit)
    report_refused_edges(records, opts.limit)
    report_doi_conflicts(clusters, opts.limit)
    report_nothing_disappears(clusters)
    report_boilerplate_cuts(records, opts.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
