# ML test tier — markers + importorskip for ML-dependent tests

**Date:** 2026-04-17
**Status:** Proposed
**Trigger:** `raven/common/nlptools.py` (~30 public functions) has no tests, and is the actual logic behind the server's `sanitize`, `natlang`, `classify`, `translate`, `embeddings` modules.  The server-module tests should exercise the logic in `common`, not in `server`, since that's where it's loaded from.

## Context

- `raven.papers` coverage push recently landed all converter modules at 93%+ (latest: `14f9b69` on `main`).
- Server-side test story is two layers:
  1. Thin HTTP wrappers in `raven/server/modules/*.py` — covered by future `test_server.py`-style tests, probably with a mocked Flask test client.
  2. **The actual logic in `raven/common/nlptools.py`** — this brief.
- LLM-backed logic (via oobabooga) is out of scope for local tests; the LLM server is external.
- GPU-enabled CI is explicitly *not* wanted — free-tier GH CI runners can't host multi-GB ML models, and the ML dep stack itself is heavy.

## Goal

- Introduce a `@pytest.mark.ml` tier.  On CI, these tests are skipped (both because the marker selects them out, and because the deps aren't installed — belt and suspenders).  On the dev machine, `pytest` runs everything by default.
- Add `raven/common/tests/test_nlptools.py` exercising as much of `nlptools.py` as practical, with real small models loaded in-process.
- Local full-coverage runs (`pytest --cov --cov-branch`) should include the ML tier automatically — no env-var dance.

## Policy split (the key idea)

Two orthogonal mechanisms with distinct roles:

| Mechanism | Scope | Answers the question |
|---|---|---|
| `pytest.importorskip("spacy")` at the top of the ML test file | Environmental | "Can I even run?"  (If spaCy isn't installed, skip.) |
| `@pytest.mark.ml` (or file-level `pytestmark = pytest.mark.ml`) | Policy | "Should I run in this test invocation?"  (CI defaults to no.) |

Do **not** use an env-var like `RUN_ML=1` — it duplicates what markers give you and doesn't compose.

## Implementation steps

### 1. Register the marker

In `pyproject.toml`, under `[tool.pytest.ini_options]`:

```toml
markers = [
    "ml: tests that require the full ML stack (spaCy, Flair, torch models). Skipped on CI; run locally by default.",
]
```

Verify existing `[tool.pytest.ini_options]` contents — if the section exists, just add `markers =` to it; if not, create the section.

### 2. Update CI

In `.github/workflows/ci.yml`, change the test step's `pytest raven/ -v --tb=short` to `pytest raven/ -v --tb=short -m "not ml"`.  Do the same in `.github/workflows/coverage.yml` (the coverage job also needs the filter, since that runner doesn't have ML deps installed either).

The `importorskip` at the top of `test_nlptools.py` is still there as a safety net — if someone later forgets the `-m` flag in a new workflow, the tests still skip cleanly because the deps aren't installed.

### 3. Write the tests

Create `raven/common/tests/test_nlptools.py`:

```python
"""Tests for raven.common.nlptools — runs the full ML stack in-process."""

import pytest

# Skip the whole file when any of the heavy deps are missing (CI).
pytest.importorskip("spacy")
pytest.importorskip("flair")
pytest.importorskip("dehyphen")
pytest.importorskip("sentence_transformers")
pytest.importorskip("transformers")

pytestmark = pytest.mark.ml   # file-level marker: all tests here are ML-tier

import torch  # safe: importorskip above guaranteed the stack is present

from raven.common import nlptools
# ... tests ...
```

The file-level `pytestmark` avoids having to decorate every test individually.

### 4. Test design by function type

`nlptools.py` has three flavors of function; shape the tests accordingly.

**a. Pure logic operating on loaded models** — `dehyphenate`, `classify`, `spacy_analyze`, `embed_sentences`, `translate`, `count_frequencies`, `detect_named_entities`, `suggest_keywords`, the private `_join_paragraphs`, `_translate_chunked`.  These are the testing sweet spot: load the model once per session (via a session-scoped fixture), run many cheap tests against it.

Fixture sketch:

```python
@pytest.fixture(scope="session")
def spacy_pipe():
    return nlptools.load_spacy_pipeline("en_core_web_sm", device_string="cpu")

@pytest.fixture(scope="session")
def dehyphenator():
    return nlptools.load_dehyphenator(
        model_name="character-level-lm-news-english",  # or whichever small Flair model raven uses
        device_string="cpu",
    )
```

Check `raven/librarian/config.py` or `raven/server/config.py` for the canonical small model names — use the same ones raven itself uses at default settings, so the tests exercise realistic behavior.

**b. Loaders** — `load_spacy_pipeline`, `load_classifier`, `load_dehyphenator`, `load_embedder`, `load_translator`.  These are integration-tested implicitly via the fixtures in (a).  Add a small number of explicit tests for loader-specific behavior: CPU fallback, model caching (loading the same model twice returns the same object), device-string parsing.

**c. Round-trip serializers** — `serialize_spacy_pipeline` / `deserialize_spacy_pipeline`, `serialize_spacy_docs` / `deserialize_spacy_docs`.  Straightforward: serialize, deserialize, assert equality on a fixed input.  Fast once the pipeline is loaded.

Test runtime budget: with CPU inference on small models, the full suite should land under ~60 s on the dev box.  If it blows past 2 min, split the heaviest fixtures behind `@pytest.mark.ml_slow` or similar.

### 5. Run environment

The venv and `env.sh` (CUDA paths for pip-installed `nvidia/*` libs) are sourced by the user **before** starting Claude Code — CC inherits `VIRTUAL_ENV`, `LD_LIBRARY_PATH`, and `PATH`, so no `source` calls are needed from CC's side.  (`source` would trigger a permission prompt on every invocation, which is why we keep it out of the CC session.)

Once CC is running, just call `pytest` directly:

```bash
pytest                                      # full suite including ML tier
pytest -m "not ml"                          # iterate on pure-logic bugs quickly
pytest -m ml                                # just after touching nlptools
pytest --cov --cov-branch --cov-report=term # full coverage, ML tier included
```

Sanity-check at the start of the session: `python -c "import torch; print(torch.cuda.is_available())"` should print `True` if the user set up the environment correctly.  If it prints `False` and tests that need GPU start failing, flag it — it likely means CC was started without `env.sh` sourced, and the user needs to restart the session.

## Out of scope (for this pass)

- **LLM-backed logic** — `raven/librarian/llmclient.py`, `raven/librarian/scaffold.py`'s LLM paths, anything that would require a running oobabooga instance.  These stay at their current coverage (11%, 90%) until someone picks up the LLM-mock story.
- **Server HTTP wrappers** (`raven/server/modules/*.py`) — separate test layer, probably a future brief.  Once `nlptools` has coverage, the server-wrapper tests can use a mocked or real `nlptools` per test as appropriate.
- **`raven/server/app.py`** — bootstrapping/routing, will need a Flask test client approach.
- **Other untested `common/` modules** (`hfutil`, `deviceinfo`, `netutil`, `bgtask`, `stringmaps`, `readcsv`, `running_average`, `audio/`, `image/`) — good candidates for a follow-up "common coverage" brief; not ML-tier, should run on CI.

## Success criteria

- `pytest -m "not ml"` is fast and passes cleanly on dev box, matches what CI will do.
- `pytest` locally runs everything including `test_nlptools.py`, passes, and the run terminates in reasonable time (target under ~60 s for the ML tier alone, subject to which models get pulled in).
- `raven/common/nlptools.py` coverage lands in the 80–95% band (the `load_*` functions will have branches — CPU fallback, caching — that are awkward to hit; aim for the pure-logic paths).
- CI still green: the `-m "not ml"` filter means existing minimal-deps jobs continue to skip these tests as before (via `importorskip`) *and* skip them via marker selection.
- CHANGELOG entry under "Internal" noting the new `ml` marker + the test module.

## Pitfalls to watch for

- **Don't** put `import torch` at module top outside the `importorskip` guard.  Some raven modules do this; `test_nlptools.py` must not, or the CI collection will fail before the skip kicks in.
- Model downloads during first test run — spaCy's `en_core_web_sm` and Flair's character-level-LM both download on first use.  That's fine on a dev box, but if someone else clones raven fresh, first `pytest` might be slow.  Consider noting this in `CLAUDE.md` under a "running tests locally" section.
- `pytest.importorskip` at module scope means test collection on CI will still *import* `raven/common/tests/test_nlptools.py` — make sure the file body above the importorskip calls doesn't trigger a failing import itself.  Standard pattern: imports of `pytest` and stdlib only before the `importorskip` calls.
- If `nlptools._join_paragraphs` ends up wanting a `TODO: should probably move into dehyphen` (per its existing comment), resist the urge to refactor during the test-writing pass — add a `TODO_DEFERRED` entry and keep the test surface stable.
