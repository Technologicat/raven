## Raven-avatar (fork of Talkinghead)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Raven-avatar (fork of Talkinghead)](#raven-avatar-fork-of-talkinghead)
    - [Introduction](#introduction)
    - [Live mode with `raven-llmclient`](#live-mode-with-raven-llmclient)
    - [Live mode with SillyTavern](#live-mode-with-sillytavern)
        - [Testing your installation](#testing-your-installation)
        - [Configuration](#configuration)
        - [Emotion templates](#emotion-templates)
        - [Animator configuration](#animator-configuration)
        - [Postprocessor configuration](#postprocessor-configuration)
        - [Postprocessor example: HDR, scifi hologram](#postprocessor-example-hdr-scifi-hologram)
        - [Postprocessor example: cheap video camera, amber monochrome computer monitor](#postprocessor-example-cheap-video-camera-amber-monochrome-computer-monitor)
        - [Postprocessor example: HDR, cheap video camera, 1980s VHS tape](#postprocessor-example-hdr-cheap-video-camera-1980s-vhs-tape)
        - [Complete example: animator and postprocessor settings](#complete-example-animator-and-postprocessor-settings)
    - [THA3 Pose Editor](#tha3-pose-editor)
    - [Troubleshooting](#troubleshooting)
        - [Low framerate](#low-framerate)
        - [Low VRAM - what to do?](#low-vram---what-to-do)
        - [Missing THA3 model at startup](#missing-tha3-model-at-startup)
        - [Known missing features](#known-missing-features)
        - [Known bugs](#known-bugs)
    - [Creating a character](#creating-a-character)
        - [Tips for Stable Diffusion](#tips-for-stable-diffusion)
    - [Acknowledgements](#acknowledgements)

<!-- markdown-toc end -->

### About this documentation

*This is the old documentation for the Talkinghead module, minimally updated to Raven-avatar (e.g. paths have been updated). For the latest on Raven-avatar, see the [Raven-avatar README](../avatar/README.md).*

*Documentation for the rest of Raven-server is not yet available. Likely it will be in August 2025. Please check back later.*

In the meantime:

- See `raven.server.config` for enabling/disabling server modules, and for specifying HuggingFace model repos to download models from.
- If you have a CUDA-capable GPU, enable GPU support in `raven.server.config`, by setting up the desired server modules to run on a CUDA device.
  - Be sure to install the CUDA optional dependencies of Raven (see [main README](../../README.md)).
- Look at the Python bindings of the web API in `raven.client.api` to get an idea of what the server can do.

*SillyTavern* compatibility:

- The `classify`, `embeddings`, and `websearch` modules work as drop-in replacements for those modules in the discontinued *SillyTavern-extras*.
  - The `websearch` module provides also a new endpoint (`/api/websearch2`) that returns structured search results. Using this requires new client code.
- The `tts` module provides an OpenAI compatible TTS endpoint you can use in *SillyTavern*.
- The `avatar` and `imagefx` modules are **not** compatible with *SillyTavern*, and need new client code. They are currenly meant for use in the Raven constellation. See `raven.avatar.settings_editor.app` for an example.


### Introduction

The *Raven-avatar* component renders a **live, AI-based custom anime avatar for your AI character**.

This produces animation from **one static anime-style 2D image** via the THA3 AI posing engine, facilitating easy creation of characters e.g. with *Stable Diffusion* or another AI image generator.

We also provide TTS (text to speech) via the Kokoro speech synthesizer. The AI avatar can be automatically lipsynced to the TTS.

The animator is built on top of a deep learning model, so optimal performance requires a fast GPU. The model can vary the character's expression, and pose some joints by up to 15 degrees. This allows producing parametric animation on the fly, just like from a traditional 2D or 3D model - but from a small generative AI. Modern GPUs have enough compute to do this in realtime.

We optionally support also cel blending to modify the texture that goes into the poser model (e.g. for sweatdrops or blush), and additional anime-style cel effects, such as floating question marks or anger veins. The additional cels are currently supplied separately for each character. The additional cel effects can also be turned off in the configuration file.

As with any AI technology, there are limitations:

- The AI-generated posed video frames may not look perfect, and in particular the THA3 poser model does not support large hats or props. For details (and many example outputs), refer to the [tech report](https://web.archive.org/web/20220606125507/https://pkhungurn.github.io/talking-head-anime-3/full.html) by the poser model's original author.

- TTS lipsync may have timing inaccuracies due to limitations of the TTS engine, and the sometimes unpredictable latency of the audio system.
  - Our code does its best, but for cases when that is not enough, we provide a global delay setting for shifting the timing (both in the client API as well as in the `raven-avatar-settings-editor` GUI app).


### SillyTavern

**As of 2025, SillyTavern no longer supports Talkinghead. This section is out of date.**

#### Testing your installation

To check that `raven-avatar` works, you can use the example character. Just copy `raven/vendor/tha3/images/example.png` to `SillyTavern/public/characters/yourcharacternamehere/talkinghead.png`.

To check that changing the character's expression works, use `/emote xxx` in SillyTavern, where `xxx` is name of one of the 28 emotions. See e.g. the filenames of the emotion templates in `raven/avatar/emotions/`.

The *Character Expressions* control panel also has a full list of emotions. In fact, instead of using the `/emote xxx` command, clicking one of the sprite slots in that control panel should apply that expression to the character.

If manually changing the character's expression works, then changing it automatically with `classify` will also work, provided that `classify` itself works.

#### Configuration

The live mode is configured per-character, via files **at the client end**:

- `SillyTavern/public/characters/yourcharacternamehere/talkinghead.png`: required. The **input image** for the animator.
  - The `talkinghead` extension does not use or even see the other `.png` files. They are used by *Character Expressions* when *talkinghead mode* is disabled.
- `SillyTavern/public/characters/yourcharacternamehere/_animator.json`: optional. **Animator and postprocessor settings**.
  - If a character does not have this file, server-side default settings are used.
- `SillyTavern/public/characters/yourcharacternamehere/_emotions.json`: optional. **Custom emotion templates**.
  - If a character does not have this file, server-side default settings are used. Most of the time, there is no need to customize the emotion templates per-character.
  - *At the client end*, only this one file is needed (or even supported) to customize the emotion templates.

By default, the **sprite position** on the screen is static. However, by enabling the **MovingUI** checkbox in SillyTavern's *User Settings ⊳ Advanced*, you can manually position the sprite in the GUI, by dragging its move handle. Note that there is some empty space in the sprite canvas around the sides of the character, so the character will not be able to fit flush against the edge of the window (since that empty space hits the edge of the window first). To cut away that empty space, see the crop options in *Animator configuration*.

Due to the base pose used by the posing engine, the character's legs are always cut off at the bottom of the image; the sprite is designed to be placed at the bottom of the window. You may need to create a custom background image that works with such a placement. Of the default SillyTavern backgrounds, at least the cyberpunk bedroom looks fine.

**IMPORTANT**: Changing your web browser's zoom level will change the size of the character, too, because doing so rescales all images, including the live feed.

We rate-limit the output to 25 FPS (maximum, default) to avoid DoSing the SillyTavern GUI, and we attempt to reach a constant 25 FPS. If the renderer runs faster, the average GPU usage will be lower, because the animation engine only generates as many frames as are actually consumed. If the renderer runs slower, the latest available frame will be re-sent as many times as needed, to isolate the client side from any render hiccups. The maximum FPS defaults to 25, but is configurable; see *Animator configuration*.

#### Emotion templates

The *THA3 Pose Editor* app included with *raven-avatar* is a GUI editor for these templates.

The batch export of the pose editor produces a set of static expression images (and corresponding emotion templates), but also an `_emotions.json`, in your chosen output folder. You can use this file at the client end as `SillyTavern/public/characters/yourcharacternamehere/_emotions.json`. This is convenient if you have customized your emotion templates, and wish to share one of your characters with other users, making it automatically use your version of the templates.

The file `_emotions.json` uses the same format as the factory settings in `raven/avatar/assets/emotions/_defaults.json`.

Emotion template lookup order is:

- The set of per-character custom templates sent by the ST client, read from `SillyTavern/public/characters/yourcharacternamehere/_emotions.json` if it exists.
- Server defaults, from the individual files `raven/avatar/assets/emotions/emotionnamehere.json`.
  - These are customizable. You can e.g. overwrite `curiosity.json` to change the default template for the *"curiosity"* emotion.
  - **IMPORTANT**: *However, updating SillyTavern-extras from git may overwrite your changes to the server-side default emotion templates. Keep a backup if you customize these.*
- Factory settings, from `raven/avatar/assets/emotions/_defaults.json`.
  - **IMPORTANT**: Never overwrite or remove this file.

Any emotion that is missing from a particular level in the lookup order falls through to be looked up at the next level.

If you want to edit the emotion templates manually (without using the GUI) for some reason, the following may be useful sources of information:

- `posedict_keys` in [`raven/server/modules/avatarutil.py`](../server/modules/avatarutil.py) lists the morphs available in THA3.
- [`raven/vendor/tha3/poser/modes/pose_parameters.py`](../vendor/tha3/poser/modes/pose_parameters.py) contains some more detail.
  - *"Arity 2"* means `posedict_keys` has separate left/right morphs.
- The GUI panel implementations in [`raven/avatar/pose_editor/app.py`](../avatar/pose_editor/app.py).

Any morph that is not mentioned for a particular emotion defaults to zero. Thus only those morphs that have nonzero values need to be mentioned.


#### Animator configuration

*The available settings keys and examples are kept up-to-date on a best-effort basis, but there is a risk of this documentation being out of date. When in doubt, refer to the actual source code, which comes with extensive docstrings and comments. The final authoritative source is the implementation itself.*

Animator and postprocessor settings lookup order is:

- The custom per-character settings sent by the ST client, read from `SillyTavern/public/characters/yourcharacternamehere/_animator.json` if it exists.
- Server defaults, from `raven/avatar/assets/settings/animator.json`, if it exists.
  - This file is customizable.
  - **IMPORTANT**: *However, updating SillyTavern-extras from git may overwrite your changes to the server-side animator and postprocessor configuration. Keep a backup if you customize this.*
- Built-in defaults, hardcoded as `animator_defaults` in [`raven/server/config.py`](../server/config.py).
  - **IMPORTANT**: Never change these!
  - The built-in defaults are used for validation of available settings, so they are guaranteed to be complete.
  - This file also documents (in comments) what each setting does.

Any setting that is missing from a particular level in the lookup order falls through to be looked up at the next level.

The idea of per-character animator and postprocessor settings is that this allows giving some personality to different characters. For example, they may sway by different amounts, the breathing cycle duration may be different, and importantly, the postprocessor settings may be different - which allows e.g. making a specific character into a scifi hologram, while others render normally.

Here is a complete example of `animator.json`, showing the default values (TODO: this example is out of date):

```json
{"target_fps": 25,
 "crop_left": 0.0,
 "crop_right": 0.0,
 "crop_top": 0.0,
 "crop_bottom": 0.0,
 "pose_interpolator_step": 0.1,
 "blink_interval_min": 2.0,
 "blink_interval_max": 5.0,
 "blink_probability": 0.03,
 "blink_confusion_duration": 10.0,
 "talking_fps": 12,
 "talking_morph": "mouth_aaa_index",
 "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],
 "sway_interval_min": 5.0,
 "sway_interval_max": 10.0,
 "sway_macro_strength": 0.6,
 "sway_micro_strength": 0.02,
 "breathing_cycle_duration": 4.0,
 "postprocessor_chain": []}
```

Note that some settings make more sense as server defaults, while others make more sense as per-character settings.

Particularly, `target_fps` makes the most sense to set globally at the server side, in `raven/avatar/assets/settings/animator.json`, while almost everything else makes more sense per-character, in `SillyTavern/public/characters/yourcharacternamehere/_animator.json`. Nevertheless, providing server-side defaults is a good idea, since the per-character animation configuration is optional.

**What each settings does**:

- `target_fps`: Desired output frames per second. Note this only affects smoothness of the output, provided that the hardware is fast enough. The speed at which the animation evolves is based on wall time. Snapshots are rendered at the target FPS, or if the hardware is slower, then as often as hardware allows. Regardless of render FPS, network send always occurs at `target_fps`, provided that the connection is fast enough. *Recommendation*: For smooth animation, make `target_fps` lower than what your hardware could produce, so that some compute remains untapped, available to smooth over the occasional hiccup from other running programs.
- `crop_left`, `crop_right`, `crop_top`, `crop_bottom`: in units where the width and height of the image are both 2.0. Cut away empty space on the canvas around the character. Note the poser always internally runs on the full 512x512 image due to its design, but the rest (particularly the postprocessor) can take advantage of the smaller size of the cropped image.
- `pose_interpolator_step`: A value such that `0 < step <= 1`. Sets how fast pose and expression changes are. The step is applied at each frame at a reference of 25 FPS (to standardize the meaning of the setting), with automatic internal FPS-correction to the actual output FPS. Note that the animation is nonlinear: the change starts suddenly, and slows down. The step controls how much of the *remaining distance* to the current target pose is covered in 1/25 seconds. Once the remaining distance approaches zero, the pose then snaps to the target pose, once the distance becomes small enough for this final discontinuous jump to become unnoticeable.
- `blink_interval_min`: seconds. After blinking, lower limit for random minimum time until next blink is allowed.
- `blink_interval_max`: seconds. After blinking, upper limit for random minimum time until next blink is allowed.
- `blink_probability`: Applied at each frame at a reference of 25 FPS, with automatic internal FPS-correction to the actual output FPS. This is the probability of initiating a blink in each 1/25 second interval.
- `blink_confusion_duration`: seconds. Upon entering the `"confusion"` emotion, the character may blink quickly in succession, temporarily disregarding the blink interval settings. This sets how long that state lasts.
- `talking_fps`: How often to re-randomize the mouth during the talking animation. The default value is based on the fact that early 2000s anime used ~12 FPS as the fastest actual framerate of new cels, not counting camera panning effects and such.
- `talking_morph`: Which mouth-open morph to use for talking. For available values, see `posedict_keys` in [`raven/server/modules/avatarutil.py`](../server/modules/avatarutil.py).
- `sway_morphs`: Which morphs participate in the sway (fidgeting) animation. This setting is mainly useful for disabling some or all of them, e.g. for a robot character. For available values, see `posedict_keys` in [`raven/server/modules/avatarutil.py`](../server/modules/avatarutil.py).
- `sway_interval_min`: seconds. Lower limit for random time interval until randomizing a new target pose for the sway animation.
- `sway_interval_max`: seconds. Upper limit for random time interval until randomizing a new target pose for the sway animation.
- `sway_macro_strength`: A value such that `0 < strength <= 1`. In the sway target pose, this sets the maximum absolute deviation from the target pose specified by the current emotion, but also the maximum deviation from the center position. The setting is applied to each sway morph separately. The emotion pose itself may use higher values for the morphs; in such cases, sway will only occur toward the center. For details, see `compute_sway_target_pose` in [`raven/server/modules/avatar.py`](../server/modules/avatar.py).
- `sway_micro_strength`: A value such that `0 < strength <= 1`. This is the maximum absolute value of random noise added to the sway target pose at each 1/25 second interval. To this, no limiting is applied, other than a clamp of the final randomized value of each sway morph to the valid range [-1, 1]. A small amount of random jitter makes the character look less robotic.
- `breathing_cycle_duration`: seconds. The duration of a full cycle of the breathing animation.
- `postprocessor_chain`: Pixel-space glitch artistry settings. The default is empty (no postprocessing); see below for examples of what can be done with this. For details, see [`raven/common/video/postprocessor.py`](../common/video/postprocessor.py).

#### Postprocessor configuration

**As of the Raven-avatar move, we now have a GUI postprocessor settings editor. See `raven.avatar.settings_editor.app`.**

*The available settings keys and examples are kept up-to-date on a best-effort basis, but there is a risk of this documentation being out of date. When in doubt, refer to the actual source code, which comes with extensive docstrings and comments. The final authoritative source is the implementation itself.*

The postprocessor configuration is stored as part of the animator configuration, stored under the key `"postprocessor_chain"`.

Postprocessing requires some additional compute, depending on the filters used and their settings. When `avatar` runs on the GPU, also the postprocessing filters run on the GPU. In gaming technology terms, they are essentially fragment shaders, implemented in PyTorch.

The filters in the postprocessor chain are applied to the image in the order in which they appear in the list. That is, the filters themselves support rendering in any order. However, for best results, it is useful to keep in mind the process a real physical signal would travel through:

*Light* ⊳ *Camera* ⊳ *Transport* ⊳ *Display*

and set the order for the filters based on that. However, this does not mean that there is just one correct ordering. Some filters are *general-use*, and may make sense at several points in the chain, depending on what you wish to simulate. Feel free to improvise, but make sure to understand why your filter chain makes sense.

The chain is allowed have several instances of the same filter. This is useful e.g. for multiple copies of an effect with different parameter values, or for applying the same general-use effect at more than one point in the chain. Note that some dynamic filters require tracking some state. These filters have a `name` parameter. The dynamic state storage is accessed by name, so the different instances should be configured with different names, so that they will not step on each others' toes in tracking their state.

The following postprocessing filters are available. Options for each filter are documented in the docstrings in [`raven/common/video/postprocessor.py`](../common/video/postprocessor.py).

**Light**:

- `bloom`: Bloom effect (fake HDR). Popular in early 2000s anime. Makes bright parts of the image bleed light into their surroundings, enhancing perceived contrast. Only makes sense when the avatar is rendered on a relatively dark background (such as the cyberpunk bedroom in the ST default backgrounds).

**Camera**:

- `chromatic_aberration`: Simulates the two types of [chromatic aberration](https://en.wikipedia.org/wiki/Chromatic_aberration) in a camera lens, axial (index of refraction varying w.r.t. wavelength) and transverse (focal distance varying w.r.t. wavelength).
- `vignetting`: Simulates [vignetting](https://en.wikipedia.org/wiki/Vignetting), i.e. less light hitting the corners of a film frame or CCD sensor, causing the corners to be slightly darker than the center.

**Transport**:

- `analog_lowres`: Simulates a low-resolution analog video signal by blurring the image.
- `analog_rippling_hsync`: Simulates bad horizontal synchronization (hsync) of an analog video signal, causing a wavy effect that causes the outline of the character to ripple.
- `analog_runaway_hsync`: Simulates a rippling, runaway hsync near the top or bottom edge of an image. This can happen with some equipment if the video cable is too long.
- `analog_vhsglitches`: Simulates a damaged 1980s VHS tape. In each 25 FPS frame, causes random lines to glitch with VHS noise.
- `analog_vhstracking`: Simulates a 1980s VHS tape with bad tracking. The image floats up and down, and a band of VHS noise appears at the bottom.
- `digital_glitches`: A glitchy digital video transport as sometimes depicted in sci-fi, with random blocks of lines suddenly shifted horizontally temporarily.

**Display**:

- `translucency`: Makes the character translucent, as if a scifi hologram.
- `banding`: Simulates the look of a CRT display as it looks when filmed on video without syncing. Brighter and darker bands travel through the image.
- `scanlines`: Simulates CRT TV like scanlines. Optionally dynamic (flipping the dimmed field at each frame).
  - From my experiments with the Phosphor deinterlacer in VLC, which implements the same effect, dynamic mode for `scanlines` would look *absolutely magical* when synchronized with display refresh, closely reproducing the look of an actual CRT TV. However, that is not possible here. Thus, it looks best at low but reasonable FPS, and a very high display refresh rate, so that small timing variations will not make much of a difference in how long a given field is actually displayed on the physical monitor.
  - If the timing is too uneven, the illusion breaks. In that case, consider using the static mode (`"dynamic": false`).

**General use**:

- `noise`: Adds noise to the brightness (luminance) or to the alpha channel (translucency).
- `desaturate`: A desaturation filter with bells and whistles. Beside converting the image to grayscale, can optionally pass through colors that match the hue of a given RGB color (e.g. keep red things, while desaturating the rest), and tint the final result (e.g. for an amber monochrome computer monitor look).

The noise filters could represent the display of a lo-fi scifi hologram, as well as noise in an analog video tape (which in this scheme belongs to "transport").

The `desaturate` filter could represent either a black and white video camera, or a monochrome display.

#### Postprocessor example: HDR, scifi hologram

The bloom works best on a dark background. We use `noise` to add an imperfection to the simulated display device, causing individual pixels to dynamically vary in their brightness (luminance). The `banding` and `scanlines` filters complete the look of how holograms are often depicted in scifi video games and movies. The `"dynamic": true` makes the dimmed field (top or bottom) flip each frame, like on a CRT television, and `"channel": "A"` applies the effect to the alpha channel, making the "hologram" translucent. (The default is `"channel": "Y"`, affecting the brightness, but not translucency.)

```
"postprocessor_chain": [["bloom", {}],
                        ["noise", {"strength": 0.1, "sigma": 0.0, "channel": "Y"}],
                        ["banding", {}],
                        ["scanlines", {"dynamic": true, "channel": "A"}]
                       ]
```

Note that we could also use the `translucency` filter to make the character translucent, e.g.: `["translucency", {"alpha": 0.7}]`.

Also, for some glitching video transport that shifts random blocks of lines horizontally, we could add these:

```
["digital_glitches", {"strength": 0.05, "name": "shift_right"}],
["digital_glitches", {"strength": -0.05, "name": "shift_left"}],
```

Having a unique name for each instance is important, because the name acts as a texture cache key.

#### Postprocessor example: cheap video camera, amber monochrome computer monitor

We first simulate a cheap video camera with low-quality optics via the `chromatic_aberration` and `vignetting` filters.

We then use `desaturate` with the tint option to produce the amber monochrome look.

The `banding` and `scanlines` filters suit this look, so we apply them here, too. They could be left out to simulate a higher-quality display device. Setting `"dynamic": false` makes the scanlines stay stationary.

```
"postprocessor_chain": [["chromatic_aberration", {}],
                        ["vignetting", {}],
                        ["desaturate", {"tint_rgb": [1.0, 0.5, 0.2]}],
                        ["banding", {}],
                        ["scanlines", {"dynamic": false, "channel": "A"}]
                       ]
```

#### Postprocessor example: HDR, cheap video camera, 1980s VHS tape

After capturing the light with a cheap video camera (just like in the previous example), we simulate the effects of transporting the signal on a 1980s VHS tape. First, we blur the image with `analog_lowres`. Then we apply `noise` with a nonzero `sigma` to make the noise blobs larger than a single pixel, and a rather high `strength`. This simulates the brightness noise on a VHS tape. Then we make the image ripple horizontally with `analog_rippling_hsync`, and add a damaged video tape effect with `analog_vhsglitches`. Finally, we add a bad VHS tracking effect to complete the "bad analog video tape" look.

Then we again render the output on a simulated CRT TV, as appropriate for the 1980s time period.

```
"postprocessor_chain": [["bloom", {}],
                        ["analog_lowres", {}],
                        ["noise", {"strength": 0.3, "sigma": 2.0, "channel": "Y"}],
                        ["analog_rippling_hsync", {}],
                        ["analog_vhsglitches", {"unboost": 1.0}],
                        ["analog_vhstracking", {}],
                        ["banding", {}],
                        ["scanlines", {"dynamic": true, "channel": "A"}]
                       ]
```

#### Complete example: animator and postprocessor settings

This example combines the default values for the animator with the "scifi hologram" postprocessor example above.

This part goes **at the server end** as `raven/avatar/assets/settings/animator.json`, to make it apply to all avatars that do not provide their own values for these settings:

```json
{"target_fps": 25,
 "pose_interpolator_step": 0.1,
 "blink_interval_min": 2.0,
 "blink_interval_max": 5.0,
 "blink_probability": 0.03,
 "blink_confusion_duration": 10.0,
 "talking_fps": 12,
 "talking_morph": "mouth_aaa_index",
 "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],
 "sway_interval_min": 5.0,
 "sway_interval_max": 10.0,
 "sway_macro_strength": 0.6,
 "sway_micro_strength": 0.02,
 "breathing_cycle_duration": 4.0
}
```

This part goes **at the client end** as `SillyTavern/public/characters/yourcharacternamehere/_animator.json`, to make it apply only to a specific character (i.e. the one that we want to make into a scifi hologram):

```json
{"postprocessor_chain": [["bloom", {}],
                         ["translucency", {"alpha": 0.9}],
                         ["noise", {"strength": 0.1, "sigma": 0.0, "channel": "A"}],
                         ["banding", {}],
                         ["scanlines", {"dynamic": true}]
                        ]
}
```

To refresh a running avatar after updating any of its settings files, make `avatar` reload your character. (Pausing and resuming the animation isn't enough.) Upon loading a character, the settings are re-read from disk both at client at server ends.


### THA3 Pose Editor

This is a standalone graphical app that you can run locally on the machine where you installed `raven-avatar`. It is based on the original manual poser app in the THA3 tech demo, but this version has some important new convenience features and usability improvements. The GUI toolkit has also changed from wxPython to [DearPyGui](https://github.com/hoffstadt/DearPyGui/), so that this integrates better with my other stuff.

The pose editor uses the same THA3 poser models as the live mode. If the directory `raven/vendor/tha3/models/` does not exist, the model files are automatically downloaded from HuggingFace and installed there.

With this app, you can:

- **Graphically edit the emotion templates** used by the live mode.
  - They are JSON files, found in `raven/avatar/assets/emotions/`.
    - The GUI also has a dropdown to quickload any preset.
  - **NEVER** delete or modify `_defaults.json`. That file stores the factory settings, and the app will not run without it.
  - For blunder recovery: to reset an emotion back to its factory setting, see the `--factory-reset=EMOTION` command-line option, which will use the factory settings to overwrite the corresponding emotion preset JSON. To reset **all** emotion presets to factory settings, see `--factory-reset-all`. Careful, these operations **cannot** be undone!
- **Batch-generate the 28 static expression sprites** for a character.
  - Input is the same single static image format as used by the live mode.
  - You can then use the generated images as the static expression sprites for your AI character. No need to run the live mode.
  - You may also want to do this even if you mostly use the live mode, in the rare case you want to save compute and VRAM.

To run the pose editor, open a terminal in your `raven` directory, and:

```bash
$(pdm venv activate)
python -m raven.avatar.editor
```

Run the editor with the `--help` option for a description of its command-line options. The command-line options of the pose editor are **completely independent** from the options of `raven.server` itself.

Currently, you can choose the device to run on (GPU or CPU), and which THA3 model to use. By default, the pose editor uses GPU and the `separable_float` model.

GPU mode gives the best response, but CPU mode (~2 FPS) is useful at least for batch-exporting static sprites when your VRAM is already full of AI.


### Creating a character

To create an AI avatar that `avatar` understands:

- The image must be of size 512x512, in PNG format.
- **The image must have an alpha channel**.
  - Any pixel with nonzero alpha is part of the character.
  - If the edges of the silhouette look like a cheap photoshop job (especially when ST renders the character on a different background), check them manually for background bleed.
- Using any method you prefer, create a front view of your character within [these specifications](readme/Character_Card_Guide.png).
  - In practice, you can create an image of the character in the correct pose first, and align it as a separate step.
  - If you use Stable Diffusion, see separate section below.
  - **IMPORTANT**: *The character's eyes and mouth must be open*, so that the model sees what they look like when open.
    - See [the THA3 example character](../vendor/tha3/images/example.png).
    - If that's easier to produce, an open-mouth smile also works.
- To add an alpha channel to an image that has the character otherwise fine, but on a background:
  - In Stable Diffusion, you can try the [rembg](https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg) extension for Automatic1111 to get a rough first approximation.
  - Also, you can try the *Fuzzy Select* (magic wand) tool in traditional image editors such as GIMP or Photoshop.
  - Manual pixel-per-pixel editing of edges is recommended for best results. Takes about 20 minutes per character.
    - If you rendered the character on a light background, use a dark background layer when editing the edges, and vice versa.
    - This makes it much easier to see which pixels have background bleed and need to be erased.
- Finally, align the character on the canvas to conform to the placement the THA3 posing engine expects.
  - We recommend using [the THA3 example character](../vendor/tha3/images/example.png) as an alignment template.
  - **IMPORTANT**: Export the final edited image, *without any background layer*, as a PNG with an alpha channel.
- Load up the result into *SillyTavern* as a `talkinghead.png`, and see how well it performs.

#### Tips for Stable Diffusion

**These tips are old, for SD 1.5.** As of May 2025, I'd recommend a checkpoint based on *Illustrious-XL*.

**Time needed**: about 1.5h. Most of that time will be spent rendering lots of gens to get a suitable one, but you should set aside 20-30 minutes to cut your final character cleanly from the background, using image editing software such as GIMP or Photoshop.

It is possible to create an `avatar` character render with Stable Diffusion. We assume that you already have a local installation of the [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg) webui.

- Don't initially worry about the alpha channel. You can add the alpha channel after you have generated the image.
- Try the various **VTuber checkpoints** floating around the Internet.
  - These are trained on talking anime heads in particular, so it's much easier getting a pose that works as input for THA3.
  - Many human-focused SD checkpoints render best quality at 512x768 (portrait). You can always crop the image later.
- I've had good results with `meina-pro-mistoon-hll3`.
  - It can produce good quality anime art (that looks like it came from an actual anime), and it knows how to pose a talking head.
  - It's capable of NSFW so be careful. Use the negative prompt appropriately.
  - As the VAE, the standard `vae-ft-mse-840000-ema-pruned.ckpt` is fine.
  - Settings: *512x768, 20 steps, DPM++ 2M Karras, CFG scale 7*.
  - Optionally, you can use the [Dynamic Thresholding (CFG Scale Fix)](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) extension for Automatic1111 to render the image at CFG 15 (to increase the chances of SD following the prompt correctly), but make the result look like as if it was rendered at CFG 7.
    - Recommended settings: *Half Cosine Up, minimum CFG scale 3, mimic CFG scale 7*, all else at default values.
- Expect to render **upwards of a hundred** *txt2img* gens to get **one** result good enough for further refinement. At least you can produce and triage them quickly.
- **Make it easy for yourself to find and fix the edges.**
  - If your character's outline consists mainly of dark colors, prompt for a light background, and vice versa.
- As always with SD, some unexpected words may generate undesirable elements that are impossible to get rid of.
  - For example, I wanted an AI character wearing a *"futuristic track suit"*, but SD interpreted the *"futuristic"* to mean that the character should be posed on a background containing unrelated scifi tech greebles, or worse, that the result should look something like the female lead of [*Saikano* (2002)](https://en.wikipedia.org/wiki/Saikano). Removing that word solved it, but did change the outfit style, too.

**Prompt** for `meina-pro-mistoon-hll3`:

```
(front view, symmetry:1.2), ...character description here..., standing, arms at sides, open mouth, smiling,
simple white background, single-color white background, (illustration, 2d, cg, masterpiece:1.2)
```

The `front view` and `symmetry`, appropriately weighted and placed at the beginning, greatly increase the chances of actually getting a direct front view.

**Negative prompt**:

```
(three quarters view, detailed background:1.2), full body shot, (blurry, sketch, 3d, photo:1.2),
...character-specific negatives here..., negative_hand-neg, verybadimagenegative_v1.3
```

As usual, the negative embeddings can be found on [Civitai](https://civitai.com/) ([negative_hand-neg](https://civitai.com/models/56519), [verybadimagenegative_v1.3](https://civitai.com/models/11772))

Then just test it, and equip the negative prompt with NSFW terms if needed.

The camera angle terms in the prompt may need some experimentation. Above, we put `full body shot` in the negative prompt, because in SD 1.5, at least with many anime models, full body shots often get a garbled face. However, a full body shot can actually be useful here, because it has the legs available so you can crop them at whatever point they need to be cropped to align the character's face with the template.

One possible solution is to ask for a `full body shot`, and *txt2img* for a good pose and composition only, no matter the face. Then *img2img* the result, using the [ADetailer](https://github.com/Bing-su/adetailer) extension for Automatic1111 (0.75 denoise, with [ControlNet inpaint](https://stable-diffusion-art.com/controlnet/#ControlNet_Inpainting) enabled) to fix the face. You can also use *ADetailer* in *txt2img* mode, but that wastes compute (and wall time) on fixing the face in the large majority of gens that do not have the perfect composition and/or outfit.

Finally, you may want to upscale, to have enough pixels available to align and crop a good-looking result. Beside latent upscaling with `ControlNet Tile` [[1]](https://github.com/Mikubill/sd-webui-controlnet/issues/1033) [[2]](https://civitai.com/models/59811/4k-resolution-upscale-8x-controlnet-tile-resample-in-depth-with-resources) [[3]](https://stable-diffusion-art.com/controlnet/#Tile_resample), you could try especially the `Remacri` or `AnimeSharp` GANs (in the *Extras* tab of Automatic1111). Many AI upscalers can be downloaded at [OpenModelDB](https://openmodeldb.info/).

**ADetailer notes**

- Some versions of ADetailer may fail to render anything into the final output image if the main denoise is set to 0, no matter the ADetailer denoise setting.
  - To work around this, use a small value for the main denoise (0.05) to force it to render, without changing the rest of the image too much.
- When inpainting, **the inpaint mask must cover the whole area that contains the features to be detected**. Otherwise ADetailer will start to process correctly, but since the inpaint mask doesn't cover the area to be edited, it can't write there in the final output image.
  - This makes sense in hindsight: when inpainting, the area to be edited must be masked. It doesn't matter how the inpainted image data is produced.


### Acknowledgements

This software incorporates the [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo) AI-based anime posing engine developed by Pramook Khungurn. The THA3 code is used under the MIT license, and the THA3 AI models are used under the Creative Commons Attribution 4.0 International license. The THA3 example character is used under the Creative Commons Attribution-NonCommercial 4.0 International license. The trained models are currently mirrored [on HuggingFace](https://huggingface.co/OktayAlpk/talking-head-anime-3).

The pose editor app has been rewritten twice: first updated into a working app and expanded for *SillyTavern-extras*, then ported to DearPyGui for *Raven-avatar*. The live mode (the animation driver) is original to *SillyTavern-extras*, although initially inspired by THA3's *IFacialMocap* VTuber tech demo. The avatar settings editor (GUI for postprocessor settings) is original to *Raven-avatar*.

The components of *Raven-avatar* that derive from *SillyTavern-extras* (`raven.server`, `raven.avatar.pose_editor`) are licensed under the same license as *SillyTavern-Extras*, namely *GNU Affero General Public License v3*.

New components, or any components where I (@Technologicat) am the only author (particularly `raven.avatar.settings_editor` and `raven.common.video.postprocessor`) are licensed under the 2-clause BSD license, like the rest of *Raven*.

As an exception, the `raven.common.video.upscaler` module is licensed under the MIT license, to match the license of the Anime4K engine it uses (so that they can be easily taken together anywhere).
