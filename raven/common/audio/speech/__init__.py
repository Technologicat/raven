"""Speech engines — TTS (Kokoro) and STT (Whisper) as in-process libraries.

No Flask, no HTTP. The server's `raven.server.modules.{tts,stt}` are thin
transport wrappers around this layer; `raven.client.mayberemote.{TTS,STT}`
lets apps run the engines locally when the server isn't reachable.
"""
