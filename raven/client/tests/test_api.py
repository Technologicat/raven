"""Exercise the API endpoints.

!!! Start `raven.server.app` first before running these tests! !!!
"""

# TODO: Now this is a demo. Convert this to a proper test module, checking outputs and everything. Could use `unpythonic.test.fixtures` as the framework.
# TODO: test *all* endpoints.

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import os
import pathlib
import textwrap

import PIL.Image

from colorama import Fore, Style, init as colorama_init

from unpythonic import timer

from .. import api
from .. import config as client_config

def test():
    """DEBUG/TEST - exercise each of the API endpoints."""

    # --------------------------------------------------------------------------------
    # Initialize

    colorama_init()

    logger.info("test: initialize API")
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file,
                   tts_server_type=client_config.tts_server_type,
                   tts_url=client_config.tts_url,
                   tts_api_key_file=client_config.tts_api_key_file,
                   tts_playback_audio_device=client_config.tts_playback_audio_device)  # let it create a default executor

    logger.info(f"test: check server availability at {client_config.raven_server_url}")
    if api.raven_server_available():
        print(f"{Fore.GREEN}{Style.BRIGHT}Connected to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}Proceeding with self-test.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR: Cannot connect to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL} Is Raven-server running?")
        print(f"{Fore.RED}{Style.BRIGHT}Canceling self-test.{Style.RESET_ALL}")
        return

    # --------------------------------------------------------------------------------
    # classify

    logger.info("test: classify")
    print(f"Emotions supported by the classifier model currently loaded on the server: {api.classify_labels()}")  # get available emotion names from server

    text = "What is the airspeed velocity of an unladen swallow?"
    classification = api.classify(text)
    print(f"Sentiment analysis for '{text}': {classification}")

    # --------------------------------------------------------------------------------
    # imagefx

    logger.info("test: imagefx")

    print(api.avatar_get_available_filters())  # this API function is common between imagefx and avatar modules

    processed_png_bytes = api.imagefx_process_file(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "backdrops", "study.png")).expanduser().resolve(),
                                                   output_format="png",
                                                   filters=[["analog_lowres", {"sigma": 3.0}],  # maximum sigma is 3.0 due to convolution kernel size
                                                            ["analog_lowres", {"sigma": 3.0}],  # how to blur more: unrolled loop
                                                            ["analog_lowres", {"sigma": 3.0}],
                                                            ["analog_lowres", {"sigma": 3.0}],
                                                            ["analog_lowres", {"sigma": 3.0}]])
    image = PIL.Image.open(io.BytesIO(processed_png_bytes))
    print(image.size, image.mode)
    # image.save("study_blurred.png")  # DEBUG so we can see it (but not useful to run every time the self-test runs)

    processed_png_bytes = api.imagefx_upscale_file(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "backdrops", "study.png")).expanduser().resolve(),
                                                   output_format="png",
                                                   upscaled_width=3840,
                                                   upscaled_height=2160,
                                                   preset="C",
                                                   quality="high")
    image = PIL.Image.open(io.BytesIO(processed_png_bytes))
    print(image.size, image.mode)
    # image.save("study_upscaled_4k.png")  # DEBUG so we can see it (but not useful to run every time the self-test runs)

    # --------------------------------------------------------------------------------
    # natlang

    logger.info("test: natlang")
    docs = api.natlang_analyze("This is a test document for NLP analysis.")
    assert len(docs) == 1
    doc = docs[0]
    print("Tokens in analyzed document:")
    for token in doc:
        print(f"  {token.text} [{token.lemma_}] {token.pos_}")
    print("Named entities in analyzed document:")
    for ent in doc.ents:
        print(f"  {ent}")
    print("Sentences in analyzed document:")
    for sent in list(doc.sents):
        print(f"  {sent}")

    docs = api.natlang_analyze(["The quick brown fox jumps over the lazy dog.",
                                "This is another document."])
    assert len(docs) == 2
    for doc in docs:
        print("Tokens in analyzed document:")
        for token in doc:
            print(f"  {token.text} [{token.lemma_}] {token.pos_}")

    # optional "pipes" argument: only enable specified pipes (to run a partial analysis faster)
    docs = api.natlang_analyze("This is a multi-sentence document. It has two sentences, see.",
                               pipes=["tok2vec", "parser", "senter"])
    assert len(docs) == 1
    doc = docs[0]
    print("Sentences in analyzed document:")
    for sent in list(doc.sents):
        print(f"  {sent}")

    # --------------------------------------------------------------------------------
    # dehyphenate, summarize

    logger.info("test: summarize")
    print(api.summarize_summarize(" The quick brown fox jumped over the lazy dog.  This is the second sentence! What?! This incomplete sentence"))

    # Neumann & Gros 2023, https://arxiv.org/abs/2210.00849
    scientific_abstract = textwrap.dedent("""
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
    with timer() as tim:
        scientific_abstract = api.sanitize_dehyphenate(scientific_abstract)
    print(f"dehyphenate scientific abstract 1: {tim.dt:0.6g}s")
    print("=" * 80)
    print(scientific_abstract)
    print("-" * 80)
    with timer() as tim:
        scientific_abstract = api.summarize_summarize(scientific_abstract)
        print(scientific_abstract)
    print(f"summarize scientific abstract 1: {tim.dt:0.6g}s")

    # Brown et al. 2020, p. 40, https://arxiv.org/abs/2005.14165
    input_text = textwrap.dedent("""
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
        set up training curricula. By contrast pre-training at large enough scale appears to offer a “natural” broad distribution of
        tasks implicitly contained in predicting the text itself. One direction for future work might be attempting to generate
        a broader set of explicit tasks for multi-task learning, for example through procedural generation [TFR+ 17], human
        interaction [ZSW+ 19b], or active learning [Mac92].

        Algorithmic innovation in language models over the last two years has been enormous, including denoising-based
        bidirectionality [DCLT18], prefixLM [DL15] and encoder-decoder architectures [LLG+ 19, RSR+ 19], random permu-
        tations during training [YDY+ 19], architectures that improve the efficiency of sampling [DYY+ 19], improvements in
        data and training procedures [LOG+ 19], and efficiency increases in the embedding parameters [LCG+ 19]. Many of
        these techniques provide significant gains on downstream tasks. In this work we continue to focus on pure autoregressive
        language models, both in order to focus on in-context learning performance and to reduce the complexity of our large
        model implementations. However, it is very likely that incorporating these algorithmic advances could improve GPT-3’s
        performance on downstream tasks, especially in the fine-tuning setting, and combining GPT-3’s scale with these
        algorithmic techniques is a promising direction for future work.
    """).strip()
    with timer() as tim:
        input_text = api.sanitize_dehyphenate(input_text)
    print(f"dehyphenate scientific abstract 2: {tim.dt:0.6g}s")
    print("=" * 80)
    print(input_text)
    print("-" * 80)
    with timer() as tim:
        input_text = api.summarize_summarize(input_text)
        print(input_text)
    print(f"summarize scientific abstract 2: {tim.dt:0.6g}s")

    # --------------------------------------------------------------------------------
    # translate

    with timer() as tim:
        print(api.translate_translate(scientific_abstract, source_lang="en", target_lang="fi"))
    print(f"translate summary of scientific abstract 1: {tim.dt:0.6g}s")

    with timer() as tim:
        print(api.translate_translate(input_text, source_lang="en", target_lang="fi"))
    print(f"translate summary of scientific abstract 2: {tim.dt:0.6g}s")

    with timer() as tim:
        # https://en.wikipedia.org/wiki/History_of_special_relativity
        print(api.translate_translate('The failure of any experiment to detect motion through the aether led Hendrik Lorentz, starting in 1892, to develop a theory of electrodynamics based on an immobile luminiferous aether, physical length contraction, and a "local time" in which Maxwell\'s equations retain their form in all inertial frames of reference.', source_lang="en", target_lang="fi"))
    print(f"translate history of theory of special relativity: {tim.dt:0.6g}s")

    with timer() as tim:
        # https://otakuusamagazine.com/macross-meet-the-idols/
        print(api.translate_translate("Sharon Apple. Before Hatsune Miku, before VTubers, there was Sharon Apple. The digital diva of Macross Plus hailed from the in-universe mind of Myung Fang Lone, and sings tunes by legendary composer Yoko Kanno. Sharon wasn't entirely artificially intelligent, though: the unfinished program required Myung to patch in emotions during her concerts.", source_lang="en", target_lang="fi"))
    print(f"translate fandom text on Sharon Apple: {tim.dt:0.6g}s")

    # --------------------------------------------------------------------------------
    # tts

    logger.info("test: tts: list voices")
    print(api.tts_list_voices())

    # --------------------------------------------------------------------------------
    # websearch

    # logger.info("test: websearch")
    # print(f"{text}\n")
    # out = api.websearch_search(text, max_links=3)
    # for item in out["data"]:
    #     if "title" in item and "link" in item:
    #         print(f"{item['title']}\n{item['link']}\n")
    #     elif "title" in item:
    #         print(f"{item['title']}\n")
    #     elif "link" in item:
    #         print(f"{item['link']}\n")
    #     print(f"{item['text']}\n")
    # # There's also out["results"] with preformatted text only.

    # --------------------------------------------------------------------------------
    # embeddings

    logger.info("test: embeddings")
    print(api.embeddings_compute(text).shape)
    print(api.embeddings_compute([text, "Testing, 1, 2, 3."]).shape)

    # --------------------------------------------------------------------------------
    # avatar

    logger.info("test: initialize avatar")
    # send an avatar - mandatory
    avatar_instance_id = api.avatar_load(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "characters", "other", "example.png")).expanduser().resolve())
    try:
        # send animator config - optional, server defaults used if not sent
        api.avatar_load_animator_settings_from_file(avatar_instance_id,
                                                    pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "settings", "animator.json")).expanduser().resolve())
        # send the morph parameters for emotions - optional, server defaults used if not sent
        api.avatar_load_emotion_templates_from_file(avatar_instance_id,
                                                    pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "emotions", "_defaults.json")).expanduser().resolve())
        api.avatar_start(avatar_instance_id)  # start the animator
        gen = api.avatar_result_feed(avatar_instance_id)  # start receiving animation frames (call this *after* you have started the animator)
        api.avatar_start_talking(avatar_instance_id)  # start "talking right now" animation (generic, non-lipsync, random mouth)

        logger.info("test: more avatar tests")
        api.avatar_set_emotion(avatar_instance_id, "surprise")  # manually update emotion
        for _ in range(5):  # get a few frames
            image_format, image_data = next(gen)  # next-gen lol
            print(image_format, len(image_data))
            image_file = io.BytesIO(image_data)
            image = PIL.Image.open(image_file)  # noqa: F841, we're only interested in testing whether the transport works.
        api.avatar_stop_talking(avatar_instance_id)  # stop "talking right now" animation
        api.avatar_stop(avatar_instance_id)  # pause animating the avatar
        api.avatar_start(avatar_instance_id)  # resume animating the avatar
    finally:
        api.avatar_unload(avatar_instance_id)  # this closes the connection too

    # --------------------------------------------------------------------------------

    logger.info("test: all done")

if __name__ == "__main__":
    test()
