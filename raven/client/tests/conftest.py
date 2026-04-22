"""Shared test fixtures for `raven.client` integration tests.

The fixtures here are reused by `test_api.py` and `test_mayberemote.py`; both
files exercise the live raven-server via `raven.client.api` and share the same
sample text data. Fixtures are small and cheap; scope is session-level for
plain data, module-level for the API-initialization side effect.

The `importorskip` guard at module level means that if the full client
dependency stack is missing (e.g. `qoi`), every test in this directory skips
cleanly rather than each file needing its own guard.
"""

import os
import pathlib
import textwrap

import pytest

pytest.importorskip("qoi", reason="qoi not installed (needs full dependency stack)")

from raven.client import api
from raven.client import config as client_config


# ---------------------------------------------------------------------------
# API / server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def initialized_api():
    """Initialize the client API and gate on server availability.

    If raven-server is unreachable, the module that requested this fixture is
    skipped entirely — start `raven-server` before running the client tests.
    """
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file)
    if not api.test_connection():
        pytest.skip("raven-server is not running")


@pytest.fixture(scope="session")
def assets_base():
    """Path to the avatar assets directory (used by image-related tests)."""
    return pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets")).expanduser().resolve()


# ---------------------------------------------------------------------------
# Sample text data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_text():
    """A single-sentence prompt suitable for classify/embeddings/etc."""
    return "What is the airspeed velocity of an unladen swallow?"


@pytest.fixture(scope="session")
def scientific_abstract_1():
    """A paragraph-scale sample containing a hyphen-broken word (`signifi-\\ncant`).

    Used by dehyphenator tests — the character-level perplexity model needs
    paragraph-scale context to score the joined form over the broken one.

    Source: Neumann & Gros 2023, https://arxiv.org/abs/2210.00849
    """
    return textwrap.dedent("""
        The recent observation of neural power-law scaling relations has made a signifi-
        cant impact in the field of deep learning. A substantial amount of attention has
        been dedicated as a consequence to the description of scaling laws, although
        mostly for supervised learning and only to a reduced extent for reinforcement
        learning frameworks. In this paper we present an extensive study of performance
        scaling for a cornerstone reinforcement learning algorithm, AlphaZero. On the ba-
        sis of a relationship between Elo rating, playing strength and power-law scaling,
        we train AlphaZero agents on the games Connect Four and Pentago and analyze
        their performance. We find that player strength scales as a power law in neural
        network parameter count when not bottlenecked by available compute, and as a
        power of compute when training optimally sized agents. We observe nearly iden-
        tical scaling exponents for both games. Combining the two observed scaling laws
        we obtain a power law relating optimal size to compute similar to the ones ob-
        served for language models. We find that the predicted scaling of optimal neural
        network size fits our data for both games. We also show that large AlphaZero
        models are more sample efficient, performing better than smaller models with the
        same amount of training data.
    """).strip()


@pytest.fixture(scope="session")
def scientific_abstract_2():
    """A second paragraph-scale sample with multiple hyphen breaks.

    Source: Brown et al. 2020, p. 40, https://arxiv.org/abs/2005.14165
    """
    return textwrap.dedent("""
        Giving multi-task models instructions in natural language was first formalized in a supervised setting with [MKXS18]
        and utilized for some tasks (such as summarizing) in a language model with [RWC+ 19]. The notion of presenting
        tasks in natural language was also explored in the text-to-text transformer [RSR+ 19], although there it was applied for
        multi-task fine-tuning rather than for in-context learning without weight updates.

        Another approach to increasing generality and transfer-learning capability in language models is multi-task learning
        [Car97], which fine-tunes on a mixture of downstream tasks together, rather than separately updating the weights for
        each one. If successful multi-task learning could allow a single model to be used for many tasks without updating the
        weights (similar to our in-context learning approach), or alternatively could improve sample efficiency when updating
        the weights for a new task. Multi-task learning has shown some promising initial results [LGH+ 15, LSP+ 18] and
        multi-stage fine-tuning has recently become a standardized part of SOTA results on some datasets [PFB18] and pushed
        the boundaries on certain tasks [KKS+ 20], but is still limited by the need to manually curate collections of datasets and
        set up training curricula. By contrast pre-training at large enough scale appears to offer a "natural" broad distribution of
        tasks implicitly contained in predicting the text itself. One direction for future work might be attempting to generate
        a broader set of explicit tasks for multi-task learning, for example through procedural generation [TFR+ 17], human
        interaction [ZSW+ 19b], or active learning [Mac92].

        Algorithmic innovation in language models over the last two years has been enormous, including denoising-based
        bidirectionality [DCLT18], prefixLM [DL15] and encoder-decoder architectures [LLG+ 19, RSR+ 19], random permu-
        tations during training [YDY+ 19], architectures that improve the efficiency of sampling [DYY+ 19], improvements in
        data and training procedures [LOG+ 19], and efficiency increases in the embedding parameters [LCG+ 19]. Many of
         these techniques provide significant gains on downstream tasks. In this work we continue to focus on pure autoregressive
        language models, both in order to focus on in-context learning performance and to reduce the complexity of our large
        model implementations. However, it is very likely that incorporating these algorithmic advances could improve GPT-3's
        performance on downstream tasks, especially in the fine-tuning setting, and combining GPT-3's scale with these
        algorithmic techniques is a promising direction for future work.
    """).strip()
