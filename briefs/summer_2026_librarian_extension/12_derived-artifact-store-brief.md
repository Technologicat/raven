# Brief: one mechanism for derived artifacts, two stores

**What:** unify how Raven stores things it *computed from* a source artifact — extracted text, OCR text,
rescaled images, burst `.bib` records, and eventually embeddings — behind one keying and regeneration
mechanism, with separate stores for the chat and document-DB lifecycles. Today the same class of object has
three different storage answers, arrived at independently and each defensible on its own.

**Why now:** images are about to become valid DB documents, which means the DB needs stored OCR text and a
rescaled image — i.e. it needs exactly the thing `sidecarstore` already does for chat attachments. That is
the moment the question stops being cosmetic. This is core logic and the longest-lived part of the codebase,
so it is worth designing rather than discovering.

**Not a rewrite.** Most of the mechanism exists. `sidecarstore` already holds the shared mechanics — URL
scheme, provenance skeleton, byte ingestion, GC content-walk — with `imagestore` and `textfilestore` as two
specializations mirroring each other at three public operations each, and the GC mark phase already documented
as composable by set union over per-kind reference interpreters. What this brief adds is a third
specialization, a second reference-map provider, and a version stamp. The claim is *not* "it's just the orphan
manager again"; it is that the orphan manager was written to be extended in this direction and hasn't been yet.

**Scope discipline.** This brief covers keying, storage, regeneration and GC. It does **not** cover scopes,
tags, or the unified document DB — none of which it depends on, which is the point. It can be built while
those are still open.

## The finding that motivates it

Verified by reading the tree, not inferred. Three storage strategies for one class of object:

| Artifact | Where it lives today |
|---|---|
| Chat attachment text (PDF, docx, HTML) | **Not persisted.** `sidecarstore.sidecar_to_text` extracts on demand, memoized on the content-addressed filename — so once per process, not per wire build. |
| DB document text | **Persisted inline** in hybridir's datastore. `_read` calls the `docextract.extract_text` callback at ingest and hands the result to `add(document_id, path, text)`. |
| Attachment bytes (image, PDF, office) | **Content-addressed sidecar.** |

All three are the same kind of object: derived from a source artifact, expensive, content-determined, not
authoritative, regenerable. The divergence is historical, not designed.

**And a fourth is arriving before this brief does.** Webfetch sidecarring (`TODO_DEFERRED.md:517`) ships in
v0.2.8, ahead of this design. It follows the chat-attachment pattern, which is the *correct* pattern for it, so
it does not make things worse — but it is a fourth call site that will want to become a registered producer,
and it is listed in the migration work below rather than treated as a constraint on it.

## What hybridir actually stores

Worth stating precisely, because the obvious reading is wrong in a way that matters for the design:

- `add()` persists the **full plaintext** of the document.
- `_prepare_document_for_indexing` then derives **chunks**, each carrying its own `text`, `chunk_id` and
  `offset`. With `extra=0.4` overlap, the chunk set alone is ~1.4× the document.
- Both are kept.

So the datastore holds roughly 2.4× the source text per document — and the module docstring already concedes
this at line 8: *rather memory-hungry, because we keep a second copy of chunks/tokens/embeddings*. D1 is
therefore acting on a view the module already holds about itself, not imposing an outside one.

**The chunks are not the redundant part** — they *are* the search results, and they need to live somewhere
regardless. The full inline copy is the redundant one.

This also corrects an argument made earlier in design discussion: the BM25 index rebuild does **not** need the
full text in bulk, because it works off the tokenized chunks already in the datastore. The full copy is needed
for re-chunking on a full rebuild (rare, already expensive) and for **arbitrary-span fetch** — both servable
from a sidecar, the latter *better* from a sidecar.

**Span fetch is the consumer that matters, and it is about to arrive.** Chunk offsets are expressed in the
full text's coordinate system, so the full text is not merely one way of reading the document — it *is* the
coordinate space the offsets refer to. Reconstructing an arbitrary span from chunks means finding the covering
set, de-overlapping at `extra=0.4`, and splicing: a jigsaw where a seek would do. And the planned
look-around-the-hit and locate-the-truncated-middle features are exactly arbitrary-span fetches, so this copy
is about to gain a caller rather than lose one.

So **D1 is not "delete the redundant copy" but "move it somewhere that serves span fetch well"** — which a
content-addressed sidecar with seekable random access does better than the datastore.

## The key shape

One key covers every case:

    (source content hash, derivation kind, derivation version)

- **Source content hash** — of the source artifact's bytes. Already how `sidecarstore` names files.
- **Derivation kind** — `extracted_text`, `ocr_text`, `thumbnail`, `page_image`, `embedding`, …
- **Derivation version** — **a hash of the producer's identity and parameters**, not a hand-bumped integer.
  For text extraction the params are trivial and it degenerates to a constant. For embeddings it must cover
  `(model_name, chunk_size, overlap)`, because re-chunking invalidates embeddings just as surely as a model
  swap does — an integer version would let a chunk-size change slip through silently.

**The version stamp is the part that pays later and is nearly free now.** `extract_keywords` already does this
locally (`nlp_cache_version = 1`); generalizing it means the Nomic embedding switch invalidates
embedding-derived artifacts and leaves OCR text and extracted PDF text alone. Today each cache invents its own
versioning or has none, which is what turns a model swap into a full reprocess.

**Derivations compose, and this is what makes the registry worth having.** `PDF → page image → OCR text` is
two registered producers with the intermediate cached like any other artifact. That dissolves the two
"off-diagonal import cells" (*get PDF page range as images*, *get text off an image*) into ordinary
registrations rather than new import paths. Adding a format becomes registering a producer function.

**Acquisition is not derivation.** Downloading a document from a URL is a *write path into the corpus*, not a
derived artifact, and `corpus-interrogation-sketch.md` already flags it as wanting deliberate treatment (token,
size cap, extension filter, off by default) rather than inheriting the inference API's trusted-network posture.
Out of scope here; noted so it doesn't get absorbed by accident.

## Multimodal: one source, several producers

The image case is the motivating one for this brief, and it is worth being precise about it, because **OCR
alone is not sufficient and neither is description alone**:

- **OCR text** — exact text *in* the image. Figures usually do have text: axis labels, legends, equation
  symbols, units. This is what keyword search wants, and it is exactly what a description would paraphrase
  away.
- **Description** — the semantics OCR cannot reach. **Needs no additional model if the backend is a VLM**,
  which brief 03's content-parts work already supports. Cheapest possible producer: a model that is already
  loaded.
- **Vision embedding** — `nomic-embed-vision` is **aligned to the same latent space as `nomic-embed-text`**
  (Nussbaum et al., *Nomic Embed Vision*, arXiv 2406.18587). So image and text documents embed into **one**
  collection and rank against each other natively; a text query matches a figure with no text produced. This
  is a side benefit of the Nomic switch, which was decided for text-quality reasons.
  - **Conditional on brief 11's model choice, and that choice is now a fork.** The alignment is to
    `nomic-embed-text-v1.5`, not to `v2-moe` — and v2-moe is the multilingual one (~100 languages,
    cross-lingual retrieval, Matryoshka 768→256). If no v2-aligned vision encoder exists, then *shared
    image-text space* and *cross-lingual text retrieval* are mutually exclusive, and picking v2-moe means
    images reach text queries through the description pivot rather than directly. See the brief 11 amendment.
    **Nothing in this brief breaks either way** — the registry stores whatever the producers emit — but the
    section above assumes v1.5, so read it with that caveat.
- **Thumbnail** — display, and cheap.

So one source type has **four producers** with different costs and independent version stamps. That is a
stronger argument for the registry than the text case, which is one-producer-one-source and does not exercise
it.

**Audio splits by whether its content is linguistic, and the two halves want different treatment.**

- **Speech → text, and that is the whole answer.** Whisper transcript, then processed as an ordinary document.
  Nobody searches recorded speech by voice timbre or speaker emotion; the words are the payload, so the
  transcript is lossless for the purpose. (`raven-transcribe` for podcasts has been floated.)
- **Music → needs its own embedding.** The semantics were never text, so a generated description is a lossy
  proxy for what an audio embedding captures directly, and *"something adventurous"* is a similarity query, not
  a keyword one. Nomic ships no audio model, so this space is **not** aligned with the text space.
  - **But it also wants a description alongside**, not instead. The embedding serves *retrieval*; the
    description serves the *conversation*, because the embedding is opaque to the model — *"what's this album
    like?"* and *"why did you pick that one?"* need text. Same coexistence as OCR and description for images.
  - **Encoder choice is measure-before-committing.** CLAP-family and music-specific encoders (MERT and
    relatives) are the obvious candidates, but quality on abstract descriptors like *adventurous* is genuinely
    uncertain — CLAP's training skews toward sound events, and the music-audio embedding field moves faster
    than text embedding.
  - **Lyrics are a third channel, and a composed derivation**: `audio → vocal stem → transcript`, exactly the
    same shape as `PDF → page image → OCR text`, with the stem cached like any other intermediate. Whisper on a
    full mix is mediocre (backing instruments, sustained vowels, melisma, harmony); source separation first,
    then transcription, is markedly better. **In practice ASR is the primary route, not a fallback** — embedded
    `LYRICS` tags and `.lrc` sidecars exist in the formats but are essentially never present in real
    collections.
  - **And this is the one channel the lazy-derivation trick does not rescue.** Summaries can be derived on
    retrieval because retrieval narrows first; lyrics search is *recall over the whole library*, since *"the one
    that goes something-something"* cannot find what was never indexed. So it is a full-library batch —
    thousands of tracks × (separation + ASR), both GPU. Overnight-job shape, same as the 12k-abstract problem
    but without the escape hatch.
  - So a music scope wants **three channels answering three different questions**: the audio embedding for mood
    (*"something adventurous"*), the description for conversation, and lyrics for exact-phrase recall (*"the one
    that goes something-something"* — BM25's home ground, not the embedding's).
  - **Lyrics are copyrighted**, and Raven is public. Indexing them locally for search is one thing; having the
    model reproduce them into output is another — arguably fair use, but cheap to be careful about. Lyrics
    chunks should be retrieval-only, not injected verbatim. Note that the extraction machinery is the
    *conservative* option here rather than the risky one: deriving from a copy the user already owns is a
    cleaner provenance story than scraping a lyrics site, which is what the alternative would have been.

**The unaligned space is less of a fusion problem than it first looks, though the scope boundary is a
tendency, not a guarantee.** A text query matching a figure *inside a paper* is a real query — same document,
same scope, which is exactly why the shared Nomic space matters there. Cross-modality queries are rarer, so
modality-specific spaces are usually fine when the scopes are modality-specific in practice.

**But they are not impossible, and an earlier draft of this brief wrongly asserted they were.** Counterexample
(Juha, 2026-08-01): *"find the hydrogen paper that best matches the mood of this track"*, with a FLAC attached.
Well-posed, and there is no reason to rule out its relatives *a priori*.

**And RRF would not have answered it anyway.** RRF fuses rank lists over *the same candidate set*; here the
probe is in audio space and the candidates are in text space, so there is no second rank list to fuse. Nothing
ranks hydrogen papers by an audio vector.

**The mechanism that does work is text as pivot**: audio → description (*restless, exploratory, unresolved*) →
text embedding → papers. Lossy, but it needs nothing new — and it gives the description channel a **third**
consumer, which makes it structural rather than a convenience:

1. retrieval within the modality (alongside the embedding),
2. the conversation (the embedding is opaque to the model),
3. **the bridge between unaligned spaces.**

**The three may not want the same text**, though one artifact probably serves all three well enough for v1:
conversation wants prose a human reads comfortably, the pivot wants dense descriptor terms that embed well, and
retrieval-within-modality wants whatever the encoder likes. This is a `(kind, params)` distinction the registry
can already express, so it costs nothing to leave the door open — and something to assume it shut.

The scaling argument is the reason to prefer this over trained bridges: pairwise alignment costs n² bridges,
while every modality already needs a text route for the corpus to work at all — so the pivot is free, n routes
instead of n².

So: **cross-space *ranking* is deferred and unsolved; cross-space *querying* is available today via the pivot**,
at whatever fidelity the description model has. Measure-first territory, same as the encoder pick.

Which fixes the framing for the vision doc: not *"Raven maps things that carry text"* but **"things that can
be made to carry text"** — with images as the modality that also has a direct route, and music as the one that
needs a route of its own.

**Deferred, not decided: true cross-space ranking**, if the pivot's fidelity turns out not to be enough. That
would mean either a trained alignment (an n² cost, and out of scope for a local tool) or one collection per
space with a fusion layer above — noting that RRF as it stands is not that layer, for the reason above.

## Two stores, one mechanism

**The lifecycles genuinely differ.** Chat sidecars are GC'd by reference from chattree nodes; DB sidecars would
be GC'd by reference from the document index. A collector that has to understand both graphs at once is where
this kind of code goes bad — and `PersistentForest._referenced_sidecars` plus the cleanup dialog are already
nontrivial against one graph.

So: **one implementation, two reference-map providers, two stores.** The mark phase is already documented as
composable by set union over per-kind reference interpreters, so this is the extension it was written for.

**Placement policy stays per-store rather than unified.** The chat store is content-addressed because dedup
matters and nobody browses it. `foo.bib.d/smith2024.bib` is legible in a file manager, and DB documents *are*
browsed. That difference looks essential rather than accidental. Unify the derivation registry; leave placement
pluggable.

Concretely for burst `.bib`: **files are named by BibTeX slug, not by hash.** Slugs are unique within a `.bib`,
so per-directory uniqueness holds, and the result is readable. Needs light filename sanitization. (Content
hashing survives as a *dedup check*; it is hash-as-filename and hash-as-primary-document-ID that were rejected
— document IDs stay path-relative, and a user who reorganizes folders pays a re-index, which is the same deal
rsync gives them.)

### Bursting is not sidecarring, and this section previously read as though it were (2026-08-07)

The paragraph above is about the **burst** of a multi-record `.bib`, and it does not generalize. Two different
mechanisms were sharing one heading:

- **Burst output** — one source file explodes into per-record files that *are* the indexed documents. One
  `.d/` directory per multi-record `.bib`, named for it, sitting beside it, browsable, legible names.
  `foo.bib.d/smith2024.bib` belongs here and nowhere else.
- **Derived-text sidecar** — the extracted plaintext of an indexed document, whatever format it arrived in.
  Content-addressed, seekable, nobody browses it; its consumer is arbitrary-span fetch (D1). A PDF's or a
  `.docx`'s extracted text has no business in anything called `.bib.d/`, and the burst records need
  sidecars of their own like every other document.

So bursting is a *source transformation* whose outputs get indexed, and sidecarring is a *cache* keyed to a
document that is already indexed. Deciding placement for one says nothing about the other.

### Where the two sidecar stores live (decided 2026-08-07)

**Chat store: keep the derivation, fix the name.** The directory is computed as
`datastore_file.with_suffix(".images")`, which is why it is `data.images/`. That derivation is load-bearing
rather than incidental — it is what keeps two datastores' sidecars apart, and therefore what keeps the GC
correct, since a prune against one datastore must not delete files the other still references. A fixed
shared directory would break that quietly.

So change the suffix and not the scheme: **`<datastore>.sidecars/`**. No `chat_` prefix is wanted; the
datastore's own name is already in there. Only `images` was inaccurate, and the code has said so for a while
(`chattree.py`: *"named `<datastore>.images/` — from when images were the only kind of attachment"*).

**And rename the datastore itself: `data.json` -> `chat.json`**, which makes the sidecar directory
`chat.sidecars/`. `data.json` says nothing about what is in it. Both renames need a migration on load;
`chattree._upgrade` is the precedent, and `librarian_config.llm_datastore_file` is the single place the name
is configured.

**Document store: one active slot, mirroring the corpus directories.** `document_sidecars` is a symlink to
the slot in use, with the real directories beside it — `document_sidecars_eccomas2024`,
`document_sidecars_hydrogen`, and so on — exactly as `documents` and `rag_index` already work.

This sidesteps a GC question rather than answering it. Content-addressed derived text would dedup across
corpora, which is a real benefit for a machine holding several that overlap; but then "is this file
orphaned?" stops being a question any one index can answer and becomes a union query over every index that
exists. With one active slot at a time there is no cross-corpus case yet, so the per-slot arrangement is
correct *and* cheap, and the union-aware version is not written speculatively.

**Once scopes land (brief 13), `document_sidecars` becomes cross-corpus naturally** — a scope-aware index
knows which corpora reference what, which is the union query, supplied rather than reconstructed.

## Synopsis: the author's compression, and when to generate one

An abstract is to a fulltext what a figure caption is to a figure: an author-written compression. That makes
`synopsis` a producer with an unusual property — **it is often already present in the source**, so the producer
is really *extract-or-generate*.

- **Prefer the author's compression when one exists.** If a document has an abstract, that *is* the synopsis;
  do not spend a VLM/LLM pass regenerating it. (This is the "check metadata first" rule, which was wrong for
  lyrics — those are essentially never tagged — and right here.)
- **Fiction is where generation earns its keep.** A synopsis compresses a novella far harder than an abstract
  compresses a paper, so coverage mode gains more. But fiction rarely fits in context, so it needs rolling or
  hierarchical summarization — **a genuinely different producer shape** from abstract-length work, and worth
  treating as a separate producer rather than the same one with a bigger input. (The FiO add-on stories are
  probably short enough to dodge this; a novel is not.)
- **Spoiler policy is a `params` distinction, not a separate kind.** A synopsis that reveals the ending and one
  that does not are different artifacts serving different consumers — *"which one is this?"* wants full
  information, a recommendation context does not. Same shape as the three description consumers above, and the
  key already expresses it.
- **Relation to the retrieval work**: this is the lazy-on-retrieval summary from the coverage-mode discussion,
  reaching the registry as an ordinary producer. Cost is bounded by what gets looked at rather than by corpus
  size, which is what makes it viable at all — import-time summarization of 10k abstracts was measured as
  hours-to-days and rejected.

## Deferred: measuring what the pivot costs

Not part of this brief's implementation, but it determines how much weight the description pivot can carry, so
it is recorded here rather than lost. **Images are the only modality where both routes exist**, which makes
them the measurable analogue for the audio case.

**Three routes over one corpus**: direct vision embedding; VLM description then text embedding; **author
caption** then text embedding. The caption is a free control condition — a human describer, effectively an
upper bound on the description channel.

- If VLM ≈ caption, the describer is not the bottleneck and the loss is intrinsic to verbalization.
- If VLM ≪ caption, the bottleneck is description *quality* — which predicts audio does **worse**, since music
  description models are far less mature than VLMs.

That distinction is what transfers to audio; the raw number will not.

**Kendall tau between routes needs no annotation**: for each query, rank the set by route A and by route B and
compare rank correlation. That measures how much the pivot changes what would be retrieved, with zero
labelling. Annotate only the queries where routes disagree most.

**Vary description length** as a second diagnostic: steep quality gains with richer descriptions means the
describer is the bottleneck; a fast plateau means verbalizability itself is.

**The document-level version runs on the corpus that already exists**: abstract-embedding vs. chunked-fulltext
embedding across the 12k papers, no annotation needed for the tau comparison. It bears directly on coverage
mode — if abstracts track fulltexts closely, generated synopses have a known-good target; if they diverge, it
says what the summary route discards before anything is built on it.

**Where the analogy strains**: the describable fraction of query-relevant semantics differs by modality. Image
queries mostly concern objects, scene and layout, all readily verbalizable. For music, mood verbalizes
acceptably but timbre, groove and production notoriously do not — and those are much of what *"adventurous"*
is doing. So the image measurement is likely an **optimistic bound**, and the caption control is what says how
optimistic.

**Scope**: needs none of this brief — a vision encoder, a VLM, and a figure set with captions is the whole
setup. Weekend-scale, like the persona/quantization pilot.

## Decisions this brief proposes

**D1 — the sidecar is authoritative for derived text; hybridir's inline full copy moves there rather than
being kept.** Chat becomes look-up-and-compute-on-miss (still lazy, but persistent across processes instead of
per-process). Chunks stay in the datastore, and span fetch reads the sidecar.

Three consumers already agree and hybridir is the outlier, which is why this is evidence rather than
preference: `textfilestore`'s docstring argues no-text-inline explicitly (*the datastore stays small even for a
large PDF*), and the deferred webfetch item makes the identical argument at `TODO_DEFERRED.md:534` (*a smaller
datastore, since the JSON keeps a `sidecar:` reference rather than the text*).

Implementation may land in two steps — demote first (sidecar authoritative, inline copy retained and
regenerable), delete second — but **demotion is a stepping stone, not the destination.** Scalability wants the
copy gone.

**D2 — burst `.bib` regeneration is selective, and this is load-bearing rather than an optimization.** Hash
each record's normalized text at burst time; rewrite only what changed. If the burst touches every file,
mtimes bump and the entire downstream per-file cache invalidates — the hash comparison would have been built
and then thrown away one layer down.

Note this hashing is **internal to the burst step** and never leaks downstream, which is what keeps it
compatible with having rejected the container/locator design. Deleted records leave orphans in `.bib.d/`,
which is the sidecar orphan problem again — including its recover-or-delete affordances.

**D3 — burst output is visible.** `foo.bib.d/` as a sibling directory, original untouched, shown in the UI,
one-click undo. This is the answer to *"auto-burst on ingest? no, too much magic"*: the objection is to things
appearing invisibly, not to things appearing.

## Open — needs a decision before implementation

- **O1. Version bump behaviour: lazy re-derive on next access, or eager sweep?** Lazy is simpler and spreads
  cost; eager gives a predictable "reprocessing, N remaining" and avoids a first-query latency cliff. Possibly
  per-kind: lazy for cheap derivations, eager for embeddings.
- **O2. Migration for existing hybridir datastores.** Regenerate from source files (correct, slow, and requires
  the sources still be where they were) or migrate in place (fast, but writes a migration path that is dead
  code within a release). Leaning migrate-in-place for text, given that sources moving is exactly the case that
  breaks regeneration.
- **O3. Migration for existing chat sidecars, and for webfetch.** Chat sidecars: probably nothing to do —
  bytes are already content-addressed; only the derived-text layer is new. Confirm. **Webfetch**: shipped in
  v0.2.8 ahead of this brief, so its extraction call site needs extracting into a named producer and
  registering. Expected to be small, but look at what actually landed rather than assuming.
- **O4. Does the DB store get its own directory, or a namespace within the existing sidecar directory?**
  Interacts with the attachment-browser item, which computes directory sizes.
- **O5. Embeddings: fold into the registry?** Easier than it first looks — **Raven computes them, and chroma
  is a pure index.** `hybridir` states this at line 275 (*we compute vector embeddings manually, on Raven's
  side*), and line 303 notes chroma only builds the HNSW because datastore documents arrive pre-embedded. They
  are already persisted Raven-side as a separate compressed `.npz`, and `self.embedding_model_name` is already
  known at save time — just not used as an invalidation key. So this is the *best-behaved* case, not the
  hardest: adding the version stamp is most of the work, and chroma keeps its current role as a rebuildable
  index. Remaining question is whether the vectors move into the store proper or stay in the npz with only the
  key managed.
- **O6. The npz round-trip is positionally coupled.** `_save_datastore` pops embeddings in dict-insertion order
  and `np.savez_compressed`es them as `*args`; `_load_datastore` re-attaches them with
  `zip(documents.values(), arrs.values())`. It works, and this is a fragility rather than a bug — but
  correctness rests on an invariant held across two files with nothing checking it, and a mismatch would attach
  embeddings to the wrong documents *silently*. A keyed registry fixes it as a side effect. Worth deciding
  whether to fix it here or separately.

## Sequencing

**How much of this to build now is an open question, and the brief is deliberately larger than the first
increment.** A plausible split, to be decided rather than assumed:

- **Core** — the key shape with producer-param versioning, the two stores with their reference-map providers,
  and D1's move of the full text to a sidecar. This is what images-in-DB actually needs.
- **Producers, incrementally** — OCR, VLM description, vision embedding, thumbnail. Each is a registration
  against the core, so they can land one at a time.
- **Later briefs** — audio (speech and music), lyrics, the synopsis producer, and the pivot-fidelity
  measurement. All of these constrain the key shape, which is why they are written down here; none of them
  need to be built for the core to be useful.

Then the ordering constraints:

- **This brief is v0.2.9 work and does not gate v0.2.8.** The release cuts once brief 09, the webfetch
  attachment work, and the queued UX fixes are in.
- **Webfetch sidecarring lands *first*, ahead of this brief, and that is fine.** It was planned and scoped
  before this design existed, and the chat-attachment pattern it follows is already the correct pattern for it
  — so it will arrive as another consumer of `sidecarstore`/`textfilestore`, not as a fourth invented scheme.
  Retrofitting it is *extract the inline extraction call into a named producer and register it*: a refactor,
  not a redesign. Deliberately **not** worth renegotiating in-flight.
  - Consequence: **add webfetch to the migration list** alongside the existing consumers (O2/O3). It is a
    fourth call site, not a special case.
  - Silver lining: a fourth consumer diverging independently *while the unification was being designed* is
    concrete evidence for the pattern this brief fixes, rather than a projected one.
- **This brief does block images-in-DB**, which in turn gates several things in the corpus track.
- **It does not depend on scopes or the unified DB**, so it can be built in parallel with that design work
  rather than after it.
