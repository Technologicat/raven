# Talkinghead TODO

## Status

As of May 2025.

Talkinghead has become `raven.avatar`, in preparation for Raven's upcoming LLM frontend, which will support an AI-animated avatar for the AI.


## High priority

- To solve the TTS different splittings issue, use a local Kokoro installation directly, it returns the per-token phonemes and timestamps.
  - Kokoro/Misaki should live on the server side, where we use GPU for compute anyway.
  - Then, provide an OpenAI compatible TTS endpoint (mainly for SillyTavern).

- "Images" -> "Characters", update folder name, texts on all buttons, etc.
- More backdrops, suitable for the different characters.

- Voice mixing (supported by Kokoro). Allows for greater variation for voices.
  - Two voices, second voice is optional, can be None.
  - GUI:
    - Add a second combobox, for the second voice. Add the None option, make it the default (so that the default is to use only one voice).
    - Slider for mix balance (step: 10%?).
    - These can fit onto one line in the `raven.avatar.client` GUI (voice names are short).

- Refactor everything, again:
  - Move the remaining GPU-dependent components of Raven to the server side.
    - Embeddings. We already have an endpoint, and it does pretty much the same thing as the current local implementation.
    - NLP. Think about the transport format. Can we JSON spaCy token streams?
  - Add an instance ID to all Talkinghead web API endpoints, to support multiple clients simultaneously.
    - `/api/talkinghead/load` should generate a new instance ID and spawn a new instance if none was given. Then, always return the instance ID that was affected by the command.
      - Instantiate an animator and an encoder.
      - Network transport is automatically instantiated when a client connects to `/api/talkinghead/result_feed`
    - Add `/api/talkinghead/unload` to delete an instance.
      - Delete the corresponding animator and encoder. Make the network transport automatically shut down on the server side (exit the generator if its encoder instance goes missing).
  - Add blur filter for use with backdrops (send an image and a postprocessor chain, receive postprocessed image?).
  - Think of naming of the app constellation's various parts.
    - `raven.server.app` - AGPL-licensed server app, because the server code is based on the old ST-Extras.
    - `raven.avatar.pose_editor` - AGPL-licensed pose editor app, because adapted from ST-Extras.
    - `raven.avatar.client` -> `raven.avatar.settings_editor`? - BSD-licensed avatar postproc editor and character tester.
    - What to do with the current `raven.avatar.common`? BSD-licensed code, needed both by the avatar client as well as by the talkinghead module of the server.
  - `app` -> `raven.visualizer.app`
  - `preprocess` -> `raven.visualizer.importer` (rename the console_script to `raven-visualizer-importer-cli` or something)
    - Change terminology everywhere, this is an importer (BibTeX input, to Raven-visualizer dataset output)
  - `llmclient` -> `raven.librarian.cli`
  - `hybridir` -> common? Could be used for advanced search in visualizer.
  - `chattree` -> `librarian.chattree`

- Fdialog use site boilerplate reduction? We have lots of these dialogs in Raven.


### Documentation

- Update README, document all web API endpoints.

- Polish up the documentation:
  - Add pictures to README.
    - Screenshot of the pose editor (`raven.avatar.pose_editor.app`). Anything else we should say about it?
    - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks. Link the original THA tech reports.
    - Examples of postprocessor filter results.
    - How each postprocessor example config looks when rendering the example character.
  - Document the postproc chain editor (`raven.avatar.client.app`).
  - Update the character-making instructions for SD Forge with Wai v14.0, RemBG extension with isnet-general-use/isnet-anime.
  - Add license note:
      All parts where I (@Technologicat) am the only author have been relicensed under 2-clause BSD. This includes the video postprocessor.
      Only the `avatar/server` and `avatar/pose_editor` folders, which each contain a separate app, are licensed under AGPL.
      The upscaler is licensed under MIT, matching the license of the Anime4K engine it uses.
      The character "avatar/assets/images/example.png" is the example character from the poser engine THA3, copyright Pramook Khungurn, and is licensed for non-commercial use.
      All other image assets are original to this software, and are licensed under CC-BY-SA 4.0.

### Examples

- Add some example characters created in Stable Diffusion.
  - Original characters.
  - Science legends: Curie, Newton, Lovelace, Archimedes.


## Low priority

Not scheduled for now.

### Backend

- Is it possible to discard the server's output stream (flushing away the remaining frames waiting for send) when the animator is paused, to prevent a hurried-looking hiccup when the animator is later resumed?

- Option to apply postprocessor before/after upscale? Currently always applied after upscale.

- Split into a separate repo and think about branding.
  - This is essentially a drop-in replacement for *SillyTavern-extras*, with modules `talkinghead`, `classify` (which Talkinghead needs), `websearch` (which Raven needs), and `embeddings` (fast endpoint for SillyTavern).
  - But as development continues, we will likely take things into a new direction, so this is effectively no longer ST-extras.
  - Also, ST has already removed Talkinghead support, so we'd need to provide a new client extension anyway.

- Add inspection capabilities for all Talkinghead settings similar to those the postprocessor already has. Would make it easier to build GUIs, always getting the right defaults and ranges for parameters.

- Web API pose control
  - Low-level direct control: body rotation, head rotation, iris position
    - Can already use `/api/talkinghead/set_overrides` for this
  - High-level hints: look at camera, look away (on which side), stand straight, randomize new sway pose


## Far future

Definitely not scheduled. Ideas for future enhancements.

- Several talkingheads running simultaneously.
  - Animator is already a class, and so is Encoder, but there is currently just one global instance of each.
  - Needs some kind of ID system.
  - Need to delete the corresponding instances when the result_feed is closed by client?

- Low compute mode: static poses + postprocessor.
  - Poses would be generated from `talkinghead.png` using THA3, as usual, but only once per session. Each pose would be cached.
  - To prevent postproc hiccups (in dynamic effects such as CRT TV simulation) during static pose generation in CPU mode, there are at least two possible approaches.
    - Generate all poses when the plugin starts. At 2 FPS and 28 poses, this would lead to a 14-second delay. Not good.
    - Run the postprocessor in a yet different thread, and postproc the most recent poser output available.
      - This would introduce one more frame of buffering, and split the render thread into two: the poser (which is 99% of the current `Animator`),
        and the postprocessor (which is invoked by `Animator`, but implemented in a separate class).
  - This *might* make it feasible to use CPU mode for static poses with postprocessing.
    - But I'll need to benchmark the postproc code first, whether it's fast enough to run on CPU in realtime.
  - Alpha-blending between the static poses would need to be implemented in the `talkinghead` module, similarly to how the frontend switches between static expression sprites.
    - Maybe a clean way would be to provide different posing strategies (alternative poser classes): realtime posing, or static posing with alpha-blending.

- Small performance optimization: see if we could use more in-place updates in the postprocessor, to reduce allocation of temporary tensors.
  - The effect on speed will be small; the compute-heaviest part is the inference of the THA3 deep-learning model.

- Add more postprocessing filters. Possible ideas, no guarantee I'll ever get around to them:
  - Pixelize, posterize (8-bit look)
  - More kinds of digital data connection glitches
    - Apply to random rectangles; may need to persist for a few frames to animate and/or make them more noticeable
    - Types:
      - Constant-color rectangle
      - Missing data (zero out the alpha?)
      - Blur (leads to replacing by average color, with controllable sigma)
      - Zigzag deformation (perhaps not needed now that we have `digital_glitches`, which is similar, but with a rectangular shape, and applied to full lines of video)
