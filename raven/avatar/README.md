# Raven-avatar

The *Raven-avatar* component renders a live avatar for the AI, using AI animation technology.

Essentially, this uses **GPU** computing (CUDA) to convert a *single static image* of an **anime-style** character into an **animated talking head**. The server can apply realtime video postprocessing effects as well as upscale the video in realtime. The avatar can also lipsync to the speech output of the `tts` module provided by *Raven-server*, providing a more convincing talking head.

The resulting video is streamed over HTTP, so it can be easily rendered anywhere. We provide Python bindings to integrate the avatar easily into Python apps, but a JavaScript client for web apps would also be possible (although not currently provided).

The server-side implementation uses PyTorch, including the fragment shaders in the video postprocessor. The AI animation technology is Talking Head Anime 3 (THA3) by @pkhungurn (Pramook Khungurn).


## Why anime?

This visual style was chosen for three main reasons:

- Efficient use of development resources. Being abstracted farther from reality, anime style is not as susceptible to the uncanny valley effect as photorealism is. Hence acceptable quality can be reached with relatively small development resources, making the technology ideal for small teams.
- Historical reasons. The THA3 animator was previously used in the now-discontinued *SillyTavern-extras*, and I happened to have worked on it, so I could quickly port the code here.
- Aesthetics. It looks nice.


## Codebase structure

Where to find the relevant files:

- The server side implementation: `raven.server.modules.avatar`
- The server side web API endpoints: `raven.server.app` (**start here** when developing your own **JavaScript** apps)
- Python bindings for the web API: `raven.client.api` (**start here** when developing your own **Python** apps)
- The client apps: `raven.avatar.*`
- Assets: `raven.avatar.assets`

The client apps include the pose editor, and the video postprocessor settings editor.

Assets include character images (512x512 PNG RGBA), backdrop images (any resolution and format), emotion templates (JSON), and animator settings (JSON).

Backdrops are applied at the client side in `raven.avatar.settings_editor.app`. If you want to do that in your own client, currently you'll have to implement something similar (render and optionally postprocess a background texture, then blit the video on top of it).


## Quick tips for character creation

AI animation quality depends on the input image. Sometimes the engine can be finicky about its input.

- Especially glasses, moustaches, and older characters may confuse the face parts detector.
- For best results, there should be nothing between the upper edge of the eyes and the lower edge of the eyebrows. You may need to stylize your character accordingly.
- If the mouth refuses to animate, try copy/pasting a mouth from an existing character.
- If one eye animates properly but the other one does not, try copy/pasting and mirroring the working one.
- Use `example.png` as an alignment template.
- When tweaking is necessary, separate the face parts into layers in your favorite image editing app, and move/edit them pixel by pixel until the result works.
- For live-testing, export your work-in-progress image, and load it in `raven.avatar.settings_editor.app`.
  - You can quickly cycle through the character's emotions by pressing Ctrl+E (Mac: ⌘+E) and then pressing the up/down arrow keys. Home/End work too, to jump to the first or last item, respectively.
- To get an intuitive feel of how the AI animator interprets your image, load the image in `raven.avatar.pose_editor.app`, and adjust the individual morphs in the GUI.


## Limitations

- AI animation quality is not perfect. The engine can make mistakes that cannot be corrected (short of retraining the neural networks).

- Input resolution of the THA3 engine is 512x512. This size is what the neural networks were trained on, so it cannot be changed.
  - Hence, very small details in the character will be lost. Plan accordingly.
  - The newer [THA4](https://arxiv.org/abs/2311.17409) engine has no publicly available implementation or weights (at least to my knowledge), and is also not fast enough for realtime.
  - You can use the upscaler to increase the output video resolution (in an anime style aware way), but obviously this will not regenerate missing details.

- AI animation is GPU compute hungry. At the default settings, the avatar barely runs at a bit under 25 FPS on an RTX 3070 Ti mobile laptop GPU.


## License

Those parts of *Raven-avatar* where I (@Technologicat / Juha Jeronen) am the only author have been relicensed under 2-clause BSD. This includes `raven.common.video.postprocessor` and `raven.avatar.settings_editor`.

Only `raven.server` and `raven.avatar.pose_editor`, which are separate apps, are licensed under AGPL, to comply with the original license of *SillyTavern-extras*.

The module `raven.common.video.upscaler` is licensed under MIT, matching the license of the Anime4K engine it uses.

The image `raven/avatar/assets/characters/example.png` is the example character from the AI animation engine THA3, copyright Pramook Khungurn, and is licensed for non-commercial use.

All other image assets are original to *Raven-avatar*, were made with GenAI, and are licensed under CC-BY-SA 4.0.


## Acknowledgements

*Raven-avatar* began as a fork of the Talkinghead module of the now-discontinued SillyTavern-extras.

It has been since extended; e.g. the TTS lipsync, the realtime video upscaler, and the postprocessor settings GUI editor app are all new.
