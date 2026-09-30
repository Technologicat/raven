# Two things a triage pass should know

Written down 2026-08-12, after a pass over the whole of `TODO_DEFERRED.md`, because both change how the
remaining items read. Consult it before triaging that file or `TODO.md`.

**An old priority label that encoded *cost* is stale; one that encoded *value* still stands.** A single label
collapses the two, and once collapsed they are indistinguishable — so a `[Low]` may mean *low value* or *not
worth the effort*, with nothing in the notation saying which. The worked instance: the datastore lockfile was
`[Low]` because the trigger is rare and a hand-rolled fix would have cost more than it was worth, while its
actual severity was silent loss of an in-progress chat. Neither of those reasons is a statement about value,
and the cost side has since moved by a large factor. For each remaining item, ask which axis its rating came
from. **This bites hardest on the audits and sweeps**, deferred when a fleet-wide mechanical pass meant a day
of tedium and now an afternoon.

**Stylistic inconsistency propagates now, so hygiene work ranks higher than it used to.** Under solo human
development a drifted convention sits where it is; under agentic development it is **distribution matching**
— the existing code is taken as evidence of the intended style, so a drifted estimate becomes the next
generation's evidence, and uncontrolled it drifts arbitrarily far. The same phenomenon at a different level
of description: agentic coding is like a *scheme* for a hyperbolic PDE, where each timestep injects local
truncation error and, without dissipation, those accumulate. (The analogy is to the scheme, not to the
equation — an earlier draft compared a continuous transport equation against a discrete stochastic process,
which is a category error.) That names the restoring force properly: **lint rules and hygiene sweeps are
artificial viscosity** — not *true*, but a damping term that keeps the trajectory bounded, which is exactly
the trade a style rule makes.

Two consequences:

- **Local review cannot detect it, by construction.** Each step is within tolerance of the state it was drawn
  from — that is what makes it drift rather than a defect. Diff review compares against the immediately
  preceding state, so it is structurally blind. Detection requires comparison against a *fixed reference*.
- **Fixed references are the restoring force, and that reranks them.** `CLAUDE.md` conventions and lint rules
  act continuously; a hygiene sweep acts episodically. Losing pycodestyle's E12x removed the only
  *mechanical* restoring force on continuation-line style — which is why the ruff item and the
  `hygiene-sweep` cluster are one concern rather than two.

**Measured 2026-08-10, and it is the prediction confirmed**: 2.25× the E12x violation density in `tests/`
against the rest of the tree (2.77 vs 1.23 per kloc) — the code the model writes most and the human reviews
in Emacs least, with an unaffected baseline in the same repo to compare against. That inverts the usual
ranking of hygiene work, which is normally deferred precisely because it is local and cosmetic. It is
neither, now.
