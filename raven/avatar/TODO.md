# Talkinghead TODO

## Status

As of May 2025.

Talkinghead has become `raven.avatar`, in preparation for Raven's upcoming LLM frontend, which will support an AI-animated avatar for the AI.


## High priority

### General

- Final touches for first version:
  - Move assets to an assets subfolder, to clean up the main level of avatar
  - Add web API to get postproc filters metadata from the running server as JSON. Needed for web clients that want to provide a postprocessor editor GUI.
  - Add license note:
      All parts where I'm the only author have been relicensed under 2-clause BSD. Only the server and pose editor folders, which each contain a separate app, are licensed under AGPL. The upscaler is licensed under MIT, matching the license of the Anime4K engine it uses.

- Feature orthogonality
  - Classify should just classify; call `/api/talkinghead/set_emotion` to set the talkinghead emotion
  - `/api/talkinghead/load` should just load a character; should have separate `/api/talkinghead/start` (now called "reload") and `/api/talkinghead/stop` (now called "unload")

- Is it possible to discard the server's output stream (flushing away the remaining frames waiting for send) when the animator is paused?

- Later: split into a separate repo and think about branding.
  - This is essentially a drop-in replacement for *SillyTavern-extras*, with modules `talkinghead`, `classify` (which Talkinghead needs), `websearch` (which Raven needs), and `embeddings` (fast endpoint for SillyTavern).
  - But as development continues, we will likely take things into a new direction, so this is effectively no longer ST-extras.

### Documentation

- Update README, document the web API endpoints.

- Polish up the documentation:
  - Add pictures to README.
    - Screenshot of the pose editor (`raven.avatar.pose_editor.app`). Anything else we should say about it?
    - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI. Link the original THA tech reports.
    - Examples of postprocessor filter results.
    - How each postprocessor example config looks when rendering the example character.
  - Document the postproc chain editor (`raven.avatar.client.app`).
  - Update the character-making instructions for SD Forge with Wai v14.0, RemBG extension with isnet-general-use/isnet-anime.

### Examples

- Add some example characters created in Stable Diffusion.
  - Original characters only.


## Low priority

Not scheduled for now.

### Backend

- Similar inspection capabilities for all Talkinghead settings as the postprocessor already has. Would make it easier to build GUIs, always getting the right defaults and ranges for parameters.

- Option to apply postprocessor before/after upscale? Currently always applied after upscale.

- Web API pose control
  - Low-level: body rotation, head rotation, iris position
  - High-level: look at camera, look away (on which side), stand straight, randomize new sway pose

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
  - Digital data connection glitches
    - Apply to random rectangles; may need to persist for a few frames to animate and/or make them more noticeable
    - Types:
      - Constant-color rectangle
      - Missing data (zero out the alpha?)
      - Blur (leads to replacing by average color, with controllable sigma)
      - Zigzag deformation (perhaps not needed now that we have `digital_glitches`, which is similar, but with a rectangular shape, and applied to full lines of video)

### Frontend

- Add a way to upload new JSON configs (`_animator.json`, `_emotions.json`), because the frontend could be running on a remote machine somewhere.
  - Send new uploaded config to backend.

### Both frontend and backend

- Lip-sync talking animation to TTS output.
  - THA3 has morphs for A, I, U, E, O, and the "mouth delta" shape Δ.

## Far future

Definitely not scheduled. Ideas for future enhancements.

- Fast, high-quality output scaling mechanism.
  - On a 4k display, the character becomes rather small.
  - The algorithm should be cartoon-aware, some modern-day equivalent of waifu2x. A GAN such as 4x-AnimeSharp or Remacri would be nice, but too slow.
  - Maybe the scaler should run at the client side to avoid the need to stream 1024x1024 PNGs.
    - Which algorithms are simple enough for a small custom implementation?
- Several talkingheads running simultaneously.
