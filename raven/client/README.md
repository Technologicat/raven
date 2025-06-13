# Raven-client

The *Raven-client* component is not a standalone app, but rather, a toolkit for building your own Python apps that use *Raven-server*.

You likely want `raven.client.api`, which wraps the web API endpoints into Python functions, and adds lipsync support to the TTS/avatar combination.

Before calling the actual API functions, call `raven.client.api.initialize`. See `raven.client.config` for suggestions for the server URLs and API keys needed for initialization.
