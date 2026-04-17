"""Speech engines — TTS (Kokoro) and STT (Whisper) as in-process libraries.

The server's `raven.server.modules.{tts,stt}` are thin transport wrappers
around this layer. Client-side remote/local dispatch lives in
`raven.client.mayberemote.{TTS,STT}`.
"""
