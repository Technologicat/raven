# Sketch: offline and remote processing at import

**Status: a discussion sketch, not an implementation brief.** Written 2026-10-01 from the maintainer's notes.
Decided: the direction — expensive per-document work moves to import time, runs offline, and may run remotely
when the data allows it. Open: nearly every mechanism.

## Why import time after all

Brief 12 records import-time summarization as measured and rejected: hours to days for 10k abstracts, so
summaries were to be derived lazily, on retrieval. **That decision is revisited** (maintainer, 2026-10-01).
Raven will need offline processing at import at some point, because the computationally expensive things are
what give the user the most leverage over their data — and the summary mipmap chain (brief 12) is the first
consumer that makes the case concrete.

## Offline, from whose viewpoint

**Offline from an app's viewpoint**: the job outlives the app that started it. That pairs with moving the
document DB behind Raven-server (brief 13, *Where the database lives*) — once the server owns the DB, the
server can own the jobs that fill it, and no app has to stay open for them.

**That is not offline enough on its own.** Raven-server may be local, and then it shuts down when the apps
do. A job that runs *in* the server stops with it, whichever LLM backend it talks to. So the fully offline
case is a job that runs somewhere else entirely and is collected later.

## Remote processing

Not all data is private, and for the public part, cloud services can do the heavy lifting. Two kinds:

- **Ordinary cloud backends** — an API the server calls while it runs.
- **Batch providers** — a Slurm queue, such as CSC's in Finnish academia. Submit a job with a script,
  download the results later. This is the fully offline case above, and a natural fit for generating the
  mipmap chains of a large corpus. The submission uploads the full texts, preferably already extracted to text
  to save bandwidth.

## What that needs

- **Several backends at once, local and cloud.** Today Librarian has exactly one LLM backend. The per-document
  LLM pass (`briefs/per-document-llm-pass-brief.md`) is where a job meets a backend, so it is probably the
  layer that would take more than one.
- **A privacy tag on every record, set at import by where it came from.** Two magic import directories: one
  for public items, where cloud services are allowed, and one for private items, where only local services
  are. The record carries the distinction from then on.
- **A service router** that picks the backend per job, from the tags of the data being sent and from the
  usual things — latency, bandwidth, cost. **One private record in the batch routes the whole batch to local
  services only.** That rule is the decided part; everything else about the router is open.

## Open questions

1. **Where the tag lives** — on the record in the unified DB (brief 13) is the obvious answer, which makes this
   another reason the two land together.
2. **What a batch job looks like** — the script, the upload format, how results come back and are matched to
   records, and what happens to a partial result. The per-document pass's resumable ledger is probably the
   right shape for the matching.
3. **How a user sees a job that runs for a day elsewhere.** `TODO_DEFERRED.md`'s *Web status panel* item is
   the same question for a local job.
4. **Whether derived artifacts inherit the tag.** A summary of a private document is as private as the
   document; a router that checks only the source's tag would get this right, but only if every artifact
   remembers its source.
