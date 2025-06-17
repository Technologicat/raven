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
import PIL.Image

from colorama import Fore, Style, init as colorama_init

from .. import api
from .. import config as client_config

def test():
    """DEBUG/TEST - exercise each of the API endpoints."""
    colorama_init()

    logger.info("test: initialize API")
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file,
                   tts_server_type=client_config.tts_server_type,
                   tts_url=client_config.tts_url,
                   tts_api_key_file=client_config.tts_api_key_file)  # let it create a default executor

    logger.info(f"test: check server availability at {client_config.raven_server_url}")
    if api.raven_server_available():
        print(f"{Fore.GREEN}{Style.BRIGHT}Connected to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}Proceeding with self-test.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR: Cannot connect to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL} Is Raven-server running?")
        print(f"{Fore.RED}{Style.BRIGHT}Canceling self-test.{Style.RESET_ALL}")
        return

    logger.info("test: classify_labels")
    print(api.classify_labels())  # get available emotion names from server

    logger.info("selftext: imagefx")
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

    logger.info("test: tts: list voices")
    print(api.tts_list_voices())

    logger.info("test: classify")
    text = "What is the airspeed velocity of an unladen swallow?"
    print(api.classify(text))  # classify some text, auto-update avatar's emotion from result

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

    logger.info("test: embeddings")
    print(api.embeddings_compute(text).shape)
    print(api.embeddings_compute([text, "Testing, 1, 2, 3."]).shape)

    logger.info("test: get metadata of available postprocessor filters")
    print(api.avatar_get_available_filters())

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

    logger.info("test: all done")

if __name__ == "__main__":
    test()
