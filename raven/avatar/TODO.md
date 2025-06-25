# Raven-avatar TODO

## Status

As of June 2025.

Talkinghead has become *Raven-avatar*, in preparation for Raven's upcoming LLM frontend *Raven-librarian*, which will support an AI-animated avatar for the AI.


## High priority

### Documentation

- Update README, document all web API endpoints.

- Polish up the documentation:
  - Add pictures to README.
    - Screenshot of the pose editor (`raven.avatar.pose_editor.app`). Anything else we should say about it?
    - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks. Link the original THA tech reports.
    - Examples of postprocessor filter results.
    - How each postprocessor example config looks when rendering the example character.
  - Document how to make the add-on cels for the characters.
  - Document the postproc chain editor (`raven.avatar.settings_editor.app`).
  - Update the character-making instructions for SD Forge with Wai v14.0, RemBG extension with isnet-general-use/isnet-anime.

### Examples

- Add some example characters created in Stable Diffusion.
  - Original characters.
  - Science legends: Curie, Newton, Lovelace, Archimedes.


## Medium priority

Maybe later.

- More backdrops, suitable for the different characters.

- Server-side backdrop image renderer as a postprocessor effect.
  - This would allow embedding the character into a scene so that the scene gets the same postprocessing applied to it as the character does.
  - In anime terms, the client-side backdrop (in `raven.avatar.settings_editor.app`) is essentially a separate backdrop cel, placed behind the CG animated character cel.

- Voice mixing. Would allow for greater variation for voices.
  - Supported by Kokoro-FastAPI; need to add this functionality to our own server too.
  - Two voices, second voice is optional, can be None.
  - GUI:
    - Add a second combobox, for the second voice. Add the None option, make it the default (so that the default is to use only one voice).
    - Slider for mix balance (step: 10%?).
    - These can fit onto one line in the `raven.avatar.settings_editor.app` GUI (voice names are short).


## Low priority

Not scheduled for now.

### Backend

- The animefx effects depend on the *current* emotion remaining the emotion that triggered the effect, so multiple emotions can't trigger quickly in succession (it will cut the previous animation).
  Fixing this needs a proper animation-queuing system.

- Is it possible to discard the server's output stream (flushing away the remaining frames waiting for send) when the animator is paused, to prevent a hurried-looking hiccup when the animator is later resumed?

- Option to apply postprocessor before/after upscale? Currently always applied after upscale.

- ST has removed Talkinghead support, so we need to provide a new client extension if we want that.

- Add inspection capabilities for all Avatar settings similar to those the postprocessor already has. Would make it easier to build GUIs, always getting the right defaults and ranges for parameters.

- Web API pose control
  - Low-level direct control: body rotation, head rotation, iris position
    - Can already use `/api/avatar/set_overrides` for this
    - Useful also for disabling some animated parts, e.g. eye animations for a nerdy character with opaque glasses
  - High-level hints: look at camera, look away (on which side), stand straight, randomize new sway pose

- Support things like cheeks blush by supporting multiple source images per character, and interpolation between them.
  - We could keep all image variants loaded onto the GPU, so switching between them would be realtime.
  - "/api/avatar/load" needs to be able to receive multiple files.
  - Need new "morphs" that actually interpolate between source images.

- Allow different simultaneous avatar instances to run on different GPUs.
  - Make the device specification a parameter of `avatar_load`. Instantiate one poser per unique GPU.
    OTOH, this requires the client to know about the server's GPU setup, which is not necessarily a good thing.


## Far future

Definitely not scheduled. Ideas for future enhancements.

- Low compute mode: static poses + postprocessor.
  - Poses would be generated from a character image using THA3, as usual, but only once per session. Each pose would be cached.
  - To prevent postproc hiccups (in dynamic effects such as CRT TV simulation) during static pose generation in CPU mode, there are at least two possible approaches.
    - Generate all poses when the plugin starts. At 2 FPS and 28 poses, this would lead to a 14-second delay. Not good.
    - Run the postprocessor in a yet different thread, and postproc the most recent poser output available.
      - This would introduce one more frame of buffering, and split the render thread into two: the poser (which is 99% of the current `Animator`),
        and the postprocessor (which is invoked by `Animator`, but implemented in a separate class).
  - This *might* make it feasible to use CPU mode for static poses with postprocessing.
    - But I'll need to benchmark the postproc code first, whether it's fast enough to run on CPU in realtime.
  - Alpha-blending between the static poses would need to be implemented in the `avatar` module, similarly to how the frontend switches between static expression sprites.
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
