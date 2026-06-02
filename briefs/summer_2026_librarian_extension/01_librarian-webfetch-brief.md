# Brief: `webfetch` native tool (complements `websearch`)

**Why now:** the lack of `webfetch` is a daily blocker — you can't currently use Librarian to
discuss things published on the web (LW posts on AI tech, blog posts, arXiv abstracts the
HTML way, etc.) without it. `websearch` returns links; `webfetch` retrieves the chosen one and
hands the model clean content. The two are a natural pair.

**Scope:** small. A Raven-native tool sitting alongside `websearch` in `raven.server`, exposed
to Librarian via the existing hardcoded `tool_entrypoints` mechanism. Not MCP (MCP is for
*external* tools). Orthogonal to the MCP client brief; can be done before, after, or in parallel.

---

## 1. Two-tier fetch — cheap path first, Selenium fallback

Mirroring the pattern used elsewhere in this work (token counting, continue stopgap):

- **Tier 1 (default):** `requests` GET + a readability extractor — recommend
  [`trafilatura`](https://trafilatura.readthedocs.io/) — which handles boilerplate removal and
  returns clean text or markdown. Fast, no browser process, suitable for the overwhelming
  majority of pages.
- **Tier 2 (fallback):** Selenium (already a Raven dependency) for pages that need JS rendering.
  Slow (spawns a browser), so used only when Tier 1 fails to extract.

**Detection is by result length, not by framework-sniffing.** Trying to detect JS-rendered apps
(SPAs — Single-Page Applications, the React/Vue/Next/Svelte class of site where the initial HTML
is a near-empty shell and content arrives via JS) upfront via `<div id="root">` markers,
`__NEXT_DATA__` presence, or script-to-text ratios is fragile — you'll forever be patching it for
new frameworks. Instead: run Tier 1; if the extractor yields below a usefulness threshold (e.g.
~300 chars of main text, tunable in config), escalate to Tier 2. False-positive cost is one
needless Selenium spawn (just slow). False-negative cost is a thin result, which the model can
flag as incomplete. Bias the threshold low so genuinely short pages don't over-escalate.

**When even Tier 2 yields below threshold**, set a `spaSuspected: true` flag on the result
(structured field alongside the extracted text). Lifted from `webbrain/web-tools`'s sensible
default — lets the model give up gracefully on a genuinely-uncrackable page (heavy SPA with
client-only hydration, login wall, captcha) instead of retrying or hallucinating around the
thin content. **The result also carries a canonical user-facing string** the model is meant to
copy: `"This site doesn't render its content as static HTML and can't be fetched as text."`
This is deliberate prompt engineering at the tool-design level — without a pre-templated
phrase, models fabricate variations on the same explanation, some of which are confusing or
inaccurate; with one, they overwhelmingly copy the phrase verbatim and user-facing messaging
stays consistent across sessions and models.

This canonical-phrase pattern applies to every error / limit case in this tool. See §2 for
the security-side phrases.

Feed the LLM the extracted clean text/markdown, never raw rendered HTML — wasteful and noisy.

**Pre-fetch URL rewriting / special-case extractors (v0)**, applied before the two-tier fetch
runs. A small URL-pattern matching layer routes known-difficult sites to better extraction
paths. The wins from existing community plugins (`killerrr777/web-search-nub235`,
`npacker/web-tools`) collected and parked here for clean v0 inclusion:

- **arXiv** (`arxiv.org/abs/X` / `arxiv.org/pdf/X`): rewrite to the HTML form
  (`arxiv.org/html/X`) when available — gives clean text instead of PDF or abstract-only page.
  Fall back to abstract page if HTML form 404s (older papers).
- **Reddit** (`reddit.com/...`, `www.reddit.com/...`): rewrite to `old.reddit.com` for cleaner
  extraction — the new design is JS-heavy and trafilatura struggles; the old design is plain
  HTML that extracts cleanly via Tier 1.
- **YouTube** (`youtube.com/watch?v=...`, `youtu.be/...`): extract video ID, fetch transcript
  via `youtube-transcript-api` (small Python dep, MIT licensed). Return transcript text instead
  of the page itself, since the page has near-zero text content beyond the player.

Pattern matching is small (~30 lines), per-pattern extractors are isolated (each is its own
function returning text/markdown). Easy to extend with new patterns later; absent patterns
fall through to the standard two-tier fetch unchanged.

---

## 2. Security — domain allowlist and content normalization

Three layers, all small.

### SSRF defense — private-network blocking (always on, v0)

**SSRF** (Server-Side Request Forgery) is the attack class where a network-capable server-side
component is tricked into making requests to resources the attacker can't reach directly. For
`webfetch`, the "server-side component" is Librarian itself — the AI agent decides URLs to
fetch on the user's behalf. Without defense, a model prompted with `http://192.168.1.1` would
happily request the user's router admin page; on cloud-hosted machines the canonical attack
hits `169.254.169.254` (AWS/GCP/Azure metadata service) and exfiltrates IAM credentials.

For Juha's case (local desktop, no cloud-metadata creds bound to 169.254), the cloud-metadata
risk is moot — but the broader class still applies on a personal machine: local network devices
(routers, NAS, IoT) often have HTTP admin pages on `192.168.x.x` with default credentials;
local services (databases, dev servers) listen on `127.0.0.1`; `file://` URLs can read local
files; on a work-from-VPN-connected machine, internal corporate services are reachable on
private addresses. The defense is cheap and universally applicable.

**v0 implementation**, applied unconditionally *before* the allowlist check (so even auto-
allowed user-pasted URLs go through):

- Resolve the URL's host. If the resolved IP falls in any of the following blocklists, refuse:
  - **IPv4 private**: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
  - **IPv4 loopback**: 127.0.0.0/8
  - **IPv4 link-local**: 169.254.0.0/16 (the cloud-metadata range)
  - **IPv4 unspecified / multicast / broadcast**: 0.0.0.0/8, 224.0.0.0/4, 255.255.255.255
  - **IPv6**: `::1` (loopback), `fc00::/7` (ULA), `fe80::/10` (link-local)
- Refuse any non-HTTP(S) scheme (`file://`, `ftp://`, `gopher://`, etc.).
- **Return a clear error result with a canonical user-facing string** the model copies. Mapped
  by refusal reason:
  - Private-IP block: `"This URL points to a private-network address (<host>). Fetching it
    would risk accessing local services unintentionally. Blocked for safety."`
  - Non-HTTP scheme: `"Only HTTP and HTTPS URLs can be fetched. <scheme> URLs are not
    supported."`
  - DNS resolution failure: `"Could not resolve the host name <host>. The site may be down,
    or the URL may be wrong."`
  Same prompt-engineering rationale as the `spaSuspected` phrase: pre-templated strings keep
  user-facing explanations consistent across sessions and prevent the model from improvising
  around what is essentially a fixed condition.

**Opt-out for power users**: a config flag `webfetch_allow_private_networks` (default `False`,
named explicitly so the user knows what they're disabling). Useful for legitimate cases —
fetching from a local Hindsight HTTP server, for instance, would land on `127.0.0.1` and be
blocked otherwise. **DNS rebinding defense is harder and out of scope for v0** (would require
resolving and comparing addresses at fetch time vs. connect time); the above static blocks
catch the overwhelming majority of accidental and ambient cases without adding that complexity.

### Domain allowlist (access control)

Off by default (`None` = unrestricted); when set, a config-driven allowlist gates which domains
`webfetch` will hit. Wildcards via simple prefix match (`*.example.com`). Applied **after**
SSRF blocking (so private-address blocks fire regardless of allowlist status).

Refusal carries the canonical string: `"The host <host> is not on the configured allowlist.
The user can add it to the webfetch_allowlist setting if you should be able to access this
site."` — same pattern as the SSRF and SPA phrases, deliberate consistency for the model to
copy.

**Suggested default allowlist for the median scientific user**, intended as the *baseline* the
user extends with field-specific entries:

```python
webfetch_allowlist = [
    # Citation / metadata
    "doi.org",
    "api.crossref.org",

    # Preprints and open peer review
    "*.arxiv.org",
    "*.biorxiv.org",
    "*.medrxiv.org",
    "openreview.net",

    # Major publishers / journals
    "*.nature.com",
    "*.science.org",
    "*.pnas.org",
    "*.plos.org",
    "*.cell.com",
    "*.springer.com",
    "link.springer.com",

    # Search / discovery
    "scholar.google.com",
    "*.semanticscholar.org",
    "researchgate.net",  # paywalled-ish without login, but a common publication venue
    "www.researchgate.net",

    # Biomedical
    "*.ncbi.nlm.nih.gov",

    # Code / models / data
    "github.com",
    "raw.githubusercontent.com",
    "gist.github.com",
    "huggingface.co",

    # General reference
    "*.wikipedia.org",
    "*.wikimedia.org",
]
```

Excluded from the default but reasonable user additions, by field: AI alignment
(`lesswrong.com`, `transformer-circuits.pub`, `alignmentforum.org`), tech press (`venturebeat.com`,
`zdnet.com`), and so on. Anything paywalled (most Nature/Science/Cell content) still gets fetched
when listed — the extractor returns the paywall page and the model can report the access state to
the user, which is the correct behaviour.

**Auto-allow URLs from the current user turn.** When the user pastes a URL into their *latest
user-role message*, treat the host(s) named there as temporarily allowlisted *for that turn only*.
Rationale: the allowlist's job is to constrain *the AI's initiative* (the model can't wander to
arbitrary sites it picks). A URL the user explicitly typed isn't the model's initiative; it's the
user's, and friction there is pure annoyance. Implementation: extract URLs from the latest
user-role message, take their hosts, union with the configured allowlist for the duration of that
generation. Restrict to the *current* turn — don't accumulate across history — so a URL embedded
in retrieved-document content from earlier turns doesn't permanently widen the trust surface
(prompt-injection guard: trust user input one hop, never transitively).

**Crucially: tool-result content does NOT receive auto-allow.** URLs appearing inside a
tool-result message (role=`tool`) — including the output of webfetch itself, MCP tools, websearch
results, retrieved DB chunks — go through the normal allowlist check. Auto-allow scopes to
*user-typed intent*, not content discovered by tools. This is the same one-hop trust boundary
applied at the message-role level: a user-pasted URL is the user's intent; a URL the model found
inside a fetched page is not.

**Bridging the workflow gap — "send to AI" affordance in the chat renderer.** Without this, the
auto-allow workflow has a hole: when the model presents search results, the user can't easily
get a URL back into a user-role message (the markdown renderer's only URL action today is
click-to-open, no copy/select). Fix: add a small inline icon next to each rendered URL in the
chat-history widget. Clicking it *appends the URL to the user's pending message draft* in the
input box (the user can then add context — "look at this one and discuss" — and send normally).
The resulting user-role message goes through the existing auto-allow logic with no special-
casing; the click is the trust-conferring action, structurally identical to the user typing or
pasting the URL themselves. This belongs to GUI work (Librarian's chat-history renderer), but
listed here because without it the webfetch workflow it's meant to support is dead-ended at the
UI layer. CC should implement both pieces in the same pass.

**Power-user escape hatch — `webfetch_trust_search_results` (default `False`, comment it
DANGEROUS).** Some users will want zero-friction "AI searches and follows links" flows and
accept the attack surface. When `True`, auto-allow extends to URLs found in websearch
tool-result content. This is a real prompt-injection vector — a poisoned SERP snippet could
embed a URL crafted to trigger the model into fetching it, which then injects further
instructions — so it's off by default, named explicitly, and the config comment names the risk
unambiguously. Don't make this easy to enable casually.

**Reloadable on the fly — planned, but not v0.** Restarting Librarian for an allowlist tweak is a
15-30 s round trip and a real iteration tax when the model hits a restriction the user wants to
relax mid-conversation. The v1 goal: an explicit "reload allowlist" action (button or
slash-command) that re-reads the config without restarting. v0 ships startup-only; v1 adds reload.

Companion `webfetch_blocklist` for the inverse case is not needed in v0 — allowlist covers the
security-minded user; blocklist can be added later if a use case shows up.

### Defensive content normalization (v0)

Before returning extracted content to the model, normalize it to strip known prompt-injection
vectors. Cheap, deterministic, mandatory for v0:

- **Strip Unicode tag characters** (U+E0000 – U+E007F). This is the ASCII-smuggler attack vector:
  invisible-to-humans glyphs that the LLM tokenizes normally, used to smuggle hidden instructions
  into seemingly innocent text. There are PoCs for this attack class (Google Jules invisible
  prompt injection, the embracethered.com ASCII smuggler demo); active in-the-wild exploitation
  lags PoCs but is plausibly a matter of time as agentic systems gain real tool access. The kill
  is one line: drop the whole U+E0000–U+E007F block from input text.
- **Strip other zero-width / invisible characters**: zero-width space (U+200B), zero-width
  non-joiner (U+200C), zero-width joiner (U+200D), zero-width no-break space (U+FEFF, except
  as legitimate BOM at file start), word joiner (U+2060), Mongolian vowel separator (U+180E).
  All are injection-vector-shaped and almost never appear in legitimate scientific prose.
- **Apply Unicode NFC normalization** — fold equivalent character sequences into canonical form.
  Standard hygiene; defangs lookalike-encoding tricks.
- **Strip control characters** other than CR, LF, TAB. Same rationale.

**Make this a shared utility, not webfetch-private**: place at `raven.common.text.normalize`.
This brief **creates the `raven.common.text` package** as part of phase 1 (`raven.common`
already has `audio`, `image`, `video`, `gui`, etc.; `text` is the natural addition).
Subsequent consumers — content-parts §4 (websearch result normalization, MCP text content),
future PDF-ingestion, any other handler of untrusted retrieved text — reuse the utility. ~30
lines, one function. Apply it to webfetch output in v0; later briefs reference it as an
established surface.

Out of scope for v0 (overactive, may break legitimate multilingual content): full confusables /
homoglyph detection, aggressive removal of bidi-override characters in code contexts. The simple
zero-width + tag-character + NFC pass kills the high-priority attack class without
false-positive risk to scientific prose.

---

## 3. Tool spec

OpenAI function shape, matching the existing `websearch` style and the verified spec (compat
brief + MCP brief §3):

```python
{
  "type": "function",
  "function": {
    "name": "webfetch",
    "description": "Retrieve a web page's main content as clean text.",
    "parameters": {
      "type": "object",
      "additionalProperties": False,
      "required": ["url"],
      "properties": {
        "url": {"type": "string", "description": "The URL to fetch."},
        # optional: "format": {"type": "string", "enum": ["text", "markdown"], "default": "markdown"}
      }
    }
  }
}
```

`additionalProperties: false` per the hardening note in the MCP brief. Markdown output is the
recommended default — it preserves structure (headings, lists, links) that pure text loses, and
modern models handle it natively.

---

## 4. Pairing with `websearch`

The two tools form an obvious workflow the model picks up without prompting:

1. User asks about something the model doesn't know → model calls `websearch(query)` → gets
   ranked link list.
2. Model picks the most relevant URL → calls `webfetch(url)` → gets clean content.
3. Model answers from fetched content, with the URL available for citation.

The LW-blog-post case is a direct application: user pastes a URL or names a topic, model fetches
the page, discussion proceeds against actual content rather than the model's prior.

---

## 5. Result format — points at the content-parts refactor

`webfetch` v0 returns text/markdown — a single text part under the content-parts model the MCP
brief §5 introduces. Compatible with both:

- The current text-only `perform_tool_calls` (today): return the extracted string.
- The future content-parts representation (when the refactor lands): wrap the extracted text as
  a single text part. Forward-compatible with no caller changes.

If/when the content-parts refactor goes in first, `webfetch` could also return structured parts
(e.g. text + extracted images, or text + a metadata dict). Not v0 scope; flagged for after the
refactor lands.

---

## 6. Out of scope for v0 (parked for later, with the design problems now half-solved)

- **Sandboxed expression-calculator tool** — `simpleeval` is exactly the right tool here, and it
  collapses the design problem. It AST-walks expressions and restricts what's allowed, so
  "sandboxing" reduces to "pick the allowed function set" (math, abs, min/max, round, …) and
  configure size limits. Note the scope: `simpleeval` handles **expressions**, not statements —
  perfect for a calculator (`2+2`, `sqrt(...)`, basic arithmetic, comparisons), wrong for
  "execute a Python script with variables and imports." For the calculator use case, this is now
  ~a-page-of-code, not its own brief.
- **Weather via OpenMeteo** — no key, no cloud account, mirrors the `webfetch` shape. Defer until
  the retrieval workstream wraps; small.
- **PDF fetch + local PDF ingestion** — significantly bigger than originally framed, and
  significantly more valuable. Current Librarian DB supports plain-text-ish formats only;
  automatic PDF text extraction would be a major value-add — *especially* for local documents,
  not just web PDFs (most scientific papers arrive as PDF). Suggested shape:

  - **Sidecar convention: double-extension** (`paper.pdf` + `paper.pdf.txt` or `paper.pdf.md`)
    rather than bare same-stem (`paper.pdf` + `paper.txt`), so the sidecar relationship is
    explicit in the filename. This avoids ambiguity in the edge case where a user genuinely has
    both `notes.pdf` and `notes.txt` as independent content (e.g. slides plus lecture notes) —
    under the bare-same-stem rule the `.txt` would be wrongly treated as derived. Under the
    double-extension rule: a `.pdf.txt` is unambiguously a sidecar; a `.txt` (or `.md`) without
    the `.pdf.` middle segment is content. Discoverable (a user inspecting the DB folder sees the
    relationship), no hidden state, no magic.
  - Original PDF lives in the DB folder (so the user opens it directly with their preferred
    viewer); the sidecar holds the extracted content the chat sees, written once at ingest time
    and refreshed if the source changes.
  - Extractor: PyMuPDF (fitz) is the workhorse — fast, high-quality text + layout, **and
    naturally page-aware** (extraction is page-by-page). For scientific PDFs with equations /
    figures, modern VLM-based extractors (nougat, marker, MinerU) give materially better results
    but are heavier; PyMuPDF first, the heavier extractors as an optional tier later.
  - **Page-aware citations** (the UX win that's almost always missing): preserve per-chunk page
    numbers as metadata at ingest time, surface the page number in any citation referencing a
    PDF chunk, and — where the OS PDF viewer supports it — open the file at that specific page
    via the `#page=N` URL fragment (`xdg-open "file:///path/paper.pdf#page=42"` works in Evince,
    Okular, Firefox, Chrome, Edge, and most modern viewers; macOS Preview is the awkward case but
    Skim supports it). The failure mode this fixes is the depressingly common "here's a 250-page
    PDF, the citation is in there somewhere, and possibly split across lines so Ctrl+F can't find
    it either, good luck" pattern. Page-aware citation turns that into one click.
  - GUI affordance: "open original" button on any citation chunk that came from a PDF, going
    straight to the right page when available.
  - This is its own brief when the time comes — it's a Librarian DB / ingestion expansion, not
    a web tool.

---

## 7. v1 deliverable — math-aware extraction (and adjacent structured-content preservation)

**The problem in v0**: trafilatura is HTML-to-plain-text and not math-aware. On a Wikipedia
mathematics article or any technical page with rendered equations, MathML gets either
stripped entirely or rendered as nested `<mi><mo><mn>` symbol soup with no structure — the
actual content of the page basically disappears for the model. Same failure mode for code
blocks (language hints lost, indentation sometimes mangled), tables (collapsed to
whitespace-separated rows or worse), and figure captions (often dropped when trafilatura
treats figures as boilerplate).

**The lucky-break that makes this tractable**: most math-bearing pages in the wild already
carry the original LaTeX source as an attribute or annotation, because the renderers that
produce the visible math were given LaTeX input. Concretely:

- **Wikipedia / MediaWiki**: `<annotation encoding="application/x-tex">` inside the
  `<math>` element carries the source LaTeX. The PNG fallback's `alt` attribute also
  contains TeX.
- **MathJax-rendered pages**: similar `<annotation>` structure, plus `data-mathml` /
  `data-original-text` attributes depending on configuration.
- **KaTeX-rendered pages**: `<annotation encoding="application/x-tex">` (KaTeX uses the
  MathML annotation convention) or `aria-label` containing the source.
- **arXiv's HTML output**: clean math source via similar mechanisms.

Extracting LaTeX from these annotation elements gives the source directly — no MathML→LaTeX
conversion needed for the majority of pages. And modern LLMs read inline (`$...$`) / display
(`$$...$$`) LaTeX natively (heavy arXiv exposure in training), so the right pipeline is:
extract LaTeX, wrap in math delimiters, hand to trafilatura as opaque text, model sees
clean math source. No information loss, no transcoding artifacts.

**Pragmatic shape for v1:**

1. **Pre-extraction DOM walker** that runs before trafilatura. Walks for math elements
   (`<math>`, `<span class="mwe-math-element">`, `<span class="math">`, MathJax / KaTeX
   containers), pulls LaTeX from annotation tags / data attributes / alt text, substitutes
   `$...$` (inline) or `$$...$$` (display, based on the `display` attribute on `<math>` or
   the wrapping element's class) into the text stream before trafilatura sees it. Trafilatura
   then treats the substituted text as opaque content and preserves it through extraction.
2. **MathML→LaTeX fallback for pages without annotation source**: use `pylatexenc` (most
   maintained MathML→LaTeX converter in the Python ecosystem). Worth a small upfront
   experiment to validate quality on real Wikipedia math before committing to it — some
   hand-written MathML in the wild is rough and auto-conversion can be hit-or-miss.
   If `pylatexenc` quality holds up, this catches the long tail.
3. **Wikipedia-specific extractor** alongside the existing arxiv→HTML form, reddit→old.reddit,
   youtube-transcript handlers (§1 v0): use the Wikipedia REST API directly
   (`en.wikipedia.org/api/rest_v1/page/html/...`). The API output has cleaner math annotations
   than the rendered article HTML, and avoids the cookie-banner / sidebar noise. Cleaner
   extraction, simpler code than scraping rendered pages.

**Adjacent generalizations worth bundling into the same work** (the "structured-content
preservation" pass — math is the most painful case but not the only one):

- **Code blocks**: `<pre><code class="language-X">` → fenced code blocks with language hint
  in markdown output. trafilatura drops the language; the fenced form preserves it.
- **Tables**: HTML `<table>` → GitHub Flavored Markdown tables. trafilatura collapses tables
  into whitespace-separated rows, losing column alignment.
- **Figure captions**: `<figcaption>` → preserved as part of the extracted text, prefixed
  with "Figure:" or similar. trafilatura often drops these as boilerplate.

These four together (math, code, tables, captions) lift webfetch from "passable for prose"
to "actually useful for technical content," which is exactly the regime Librarian's users
spend time in. The work clusters naturally as one extension because they all share the
"DOM-walk-before-trafilatura" pre-extraction pass.

**Integration verification with chat rendering**: this work produces text containing
`$...$` and `$$...$$` delimiters. The vendored markdown renderer needs to leave those alone
(most markdown renderers do, but some eat dollar signs). Quick check during implementation;
if it eats them, either configure or extend the renderer to preserve math-delimiter spans.
The brief flags this as a verify-during-implementation step rather than predicting which
case applies.

**What's deliberately not v1**: full PDF-style layout reconstruction, OCR for image-only
math, semantic conversion of math to MathML on the way back (we only need the LaTeX), or
VLM-based extraction of complex pages. The annotation-tag extraction handles the dominant
case at one or two orders of magnitude less complexity than those alternatives.

---

## Acceptance

- `webfetch(url)` retrieves a static HTML page, returns clean markdown of its main content; a
  multi-turn exchange where the model fetches an LW post and discusses it works end-to-end.
- JS-heavy pages (where Tier 1 extracts below threshold) trigger Tier 2 (Selenium) automatically
  and produce useful content.
- Allowlist, when configured, blocks fetches to non-listed domains with a clear error result.
- A URL pasted by the user in the current turn is fetched even if its host isn't on the
  configured allowlist; the same URL pasted earlier in history (a previous turn, or inside
  retrieved content) does **not** receive auto-allow.
- The chat renderer offers a "send to AI" affordance on every rendered URL; clicking it
  populates the user's pending input with that URL, and the subsequent user message receives
  auto-allow normally. URLs from search results can be forwarded to the AI in one click.
- `webfetch_trust_search_results = True` (when explicitly opted into) extends auto-allow to URLs
  inside websearch tool-result content; default `False` keeps that path locked.
- The default allowlist (when explicitly enabled by setting it to the default constant) lets
  fetches of e.g. an arXiv abstract or a Wikipedia page succeed without further configuration.
- `websearch` + `webfetch` workflow (find link → fetch link) flows without explicit prompt
  engineering on common model families.
- Returned content has Unicode tag characters, zero-width chars, and other invisible-injection
  glyphs stripped; NFC normalization applied. A test page seeded with such content yields clean
  output.
- Requests to private-network addresses (192.168.x.x, 127.0.0.1, 169.254.x.x, IPv6 loopback /
  ULA) are refused with a clear error, regardless of allowlist status; `webfetch_allow_private_
  networks=True` opt-out lets them through for power users (needed for local-Hindsight fetches
  on 127.0.0.1).
- Non-HTTP(S) schemes (`file://`, `ftp://`, etc.) are refused.
- A JS-heavy page whose Tier 2 extraction also falls below threshold returns
  `spaSuspected: true` in the result so the model can give up gracefully.
- arXiv abstract URLs are rewritten to the HTML form when available; Reddit URLs to
  `old.reddit.com`; YouTube URLs return transcript text via `youtube-transcript-api`.
- All refusal / limit results include the canonical user-facing string in their response
  payload (SSRF block, scheme block, DNS failure, allowlist refusal, SPA suspected); a behavior
  test with a small open-weight model shows it copies the canonical phrase verbatim in most
  cases rather than fabricating a variation.
- Returns text/markdown today; trivially upgradeable to a text content-part once the content-
  parts refactor lands.
