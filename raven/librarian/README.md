<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<p align="center">
<img src="../../img/screenshot-librarian.png" alt="Screenshot of Raven-librarian" width="800"/> <br/>
<i>Raven-librarian is a multiversal LLM frontend with a document database and a talking AI avatar.</i>
</p>

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
    - [Nonlinear history](#nonlinear-history)
        - [Why a nonlinear history?](#why-a-nonlinear-history)
    - [Document database](#document-database)
    - [Tools](#tools)
- [Voice mode](#voice-mode)
- [GUI walkthrough](#gui-walkthrough)
- [Configuration](#configuration)
    - [Server connections](#server-connections)
    - [Voice mode](#voice-mode-1)
    - [System prompt, AI character personality, communication style](#system-prompt-ai-character-personality-communication-style)
    - [AI avatar](#ai-avatar)
- [Future vision](#future-vision)
- [Appendix: Brief notes on how to set up a local LLM](#appendix-brief-notes-on-how-to-set-up-a-local-llm)

<!-- markdown-toc end -->

# Introduction

:exclamation: *This document is a very early WIP, with many placeholders.* :exclamation:

**Raven-librarian** is a multiversal LLM (large language model) frontend, mainly meant for working with local AI.

- **Graphical user interface** (GUI). Easy to use.
- **Multiversal**. The chat history is natively nonlinear, respecting the natural shape of conversation between a user and an LLM.
- **Animated AI avatar** with emotional reactions, lipsynced speech (English), and optional machine-translated subtitles (in a language of your choice).
- **Voice mode**. Talk with the AI using your mic. (English only for now.)
- **Fully local**, if you have a local LLM.
- **Document database** for fact grounding. Talk with the AI about the content of your documents. Powered by a local hybrid semantic and keyword search engine for optimal results.
- **Tool-calling**. The AI has access to tools provided by the *Librarian* software. (Websearch only for now; we plan to extend this later.)
- **Open source**. 2-clause BSD license.

**:exclamation: *Raven-librarian* is currently under development. :exclamation:**

Much (but not all) of the basic functionality is complete, the codebase should be in a semi-maintainable state, and most bugs have been squashed. If you find a bug that is not listed in [TODO.md](TODO.md), please [open an issue](https://github.com/Technologicat/raven/issues).

That said, some important features are still missing, and others will be expanded upon, schedule and funding permitting.

# Features

- nonlinear history, RAG, tools

## Nonlinear history

*Raven-librarian* stores chats in a tree format. The system prompt forms the root node. The AI's initial greeting, which forms the start point of a new chat, is immediately below the system prompt. The conversations then branch out from there.

Importantly, *there is no linear history*. The concept of "a chat" (as in "recent chats" or "chat files") is not even defined. A *linearized* history is built dynamically whenever needed, by following the parent link chain up the tree, starting from the tip of the current branch.

See the figure for a schematic illustration.

<p align="center">
<img src="../../img/chattree-diagram.png" alt="Raven-librarian stores chats in a tree format." width="800"/> <br/>
<i>Raven-librarian stores chats in a tree format. Here <b>SYS</b> is the system prompt node, <b>NEW</b> is the start node for a new chat, and <b>HEAD</b> is the tip of the current branch. Linearized history (highlighted in green for the <b>HEAD</b> shown) is built dynamically, by following the parent link chain up the tree. Any AI response can be rerolled, creating a new sibling node.</i>
</p>

With this storage scheme, a chat branch is just its **HEAD** pointer; roughly, like in `git`. This makes some actions cheap. For example, starting a new chat only resets the **HEAD** pointer to the AI's greeting.

The nodes are versioned, for an upcoming editing feature for fixing typos and making similar small edits that don't change the flow of the chat (i.e. meant for use in cases where any messages downstream of the edit still make sense as-is). By design, each version is immutable.

Normally, when you write and send a new message to the AI, it is added below the **HEAD**, and it then becomes the new **HEAD**. The AI then replies. The AI's reply is added below your message, and becomes the new **HEAD**.

It is possible to **reroll** the AI's reply. This replaces the AI's message with a freshly generated one. The old reply is kept in the tree, but not shown to the AI when it writes the new one. It is stored as a sibling node, and it retains any links to nodes that were downstream of it (i.e. the whole subtree is preserved). The new reply becomes the **HEAD**. Rerolling is convenient for quickly generating alternative replies to the same question.

More generally, for changing the direction of the conversation, the chat supports **branching**. You can branch at any node. Branching the chat only sets the **HEAD** pointer to the node where you branch at; it is then up to you as to how to continue the conversation from there.

It is possible to permanently forget a subtree by **deleting** it. This will also delete all messages downstream of the deleted node, and cannot be undone.

The chat tree is stored (by default) in `~/.config/raven/llmclient/data.json`.

### Why a nonlinear history?

In short, this is a better conceptual fit for working with LLMs.

An LLM is essentially a stochastic model conditioned on the prefix: the text so far. Used autoregressively, it is a discrete time evolution operator, roughly in the same sense as the (continuous) time evolution operator in quantum mechanics. The LLM sampler collapses the probability distribution for the next token, reifying one possible textual future one token at a time. Thus the sampler plays the role of the observer in the Copenhagen interpretation of QM. (Paraphrased from [Janus, 2021](https://generative.ink/posts/language-models-are-multiverse-generators/).)

From a practical perspective, keep in mind that an LLM only "remembers" in two ways:

- Static knowledge stored in weights, from training, and
- The text in the context.

Insofar as the LLM's thread of thought can be said to be "at" a place in the latent space, that place - and thus also the likely destinations to which the conversation leads - is fully determined by the context.

That context can be rewound, extended, **engineered**.

For example, it often happens that the LLM's first response to a question is not perfect. There is always variance in an LLM's output. Rerolling sometimes helps.

But also the old responses may contain useful parts. To improve upon further, one can perform *synthesis* - that is, combine several replies by manual editing before continuing the conversation. This is a form of context engineering.

Rerolling also comes in useful for **epistemic analysis**. When the AI's reply is rerolled:

- If the LLM states the same thing over and over, it believes in what it said (regardless of whether what it said is actually true or not).
- If the LLM gives a wildly different response each time, then it didn't actually know, didn't notice that it didn't know, and confabulated (a.k.a. "*hallucinated*") something random.

Finally, sometimes it is interesting or useful to **visit alternative branches**. (Or, in sci-fi terminology, alternative timelines in the multiverse of discourse.) This allows a discussion to become a garden of forking paths, facilitating a more complex and complete exploration of a topic than one linear chat.

A tree structure, and a GUI to match that, facilitates this curation, analysis, and exploration.

**Notes**

*As someone on the internet pointed out, the term "hallucination" is misleading, since in humans, it refers to **inputs** that are not grounded in external reality - or as WordNet puts it, "a sensory perception of something that does not exist". However, the common usage of "LLM hallucination" refers to ungrounded **outputs**. A better semantic match here is "a fabricated memory believed to be true", which is termed a "confabulation".*

As of 12/2025, many AI chat apps still operate in the paradigm of a traditional linear history. Many have some form of branching, but none seem to have taken the idea to its logical conclusion yet. *SillyTavern* offers swipes and arbitrary branching, but uses a linear storage format ([*Timelines*](https://github.com/SillyTavern/SillyTavern-Timelines) is a hack on top of that). Loom ([original](https://github.com/socketteer/loom); [obsidian](https://github.com/cosmicoptima/loom)) is probably still the only natively nonlinear LLM GUI (that predates *Raven-librarian*); and it is focused on text completion via base models, not chat.

## Document database

- for now, plain text only
  - Any plain text format that your LLM can read is fine (e.g. `.txt`, `.md`, `.bib`)
  - To extract text from PDF files, use `pdftotext`; or failing that, `ocrmypdf`
  - We plan to integrate PDF and HTML importers later
- where to put documents, what Librarian does with them
  - can be configured in `raven.librarian.config`
  - default is `~/.config/raven/llmclient/documents`
- search index auto-update mechanism (offline & online update)

The search index automatically reflects the current state of the document database. Removing the documents from the document database also automatically removes them from the index.

It is possible to manually force rebuilding the index, by deleting the whole index while *Raven-librarian* is not running, and then starting *Raven-librarian*.

## Tools

- Tool use = function-calling
- JSON specs in system prompt
- LLM calls those functions when it feels like it, and interprets the results
- This allows the LLM to automatically retrieve useful material into its context
- Tools provided
  - Websearch
  - We plan to expand this later

# Voice mode

Librarian features a lipsynced talking AI avatar, as well as speech recognition for text input. These features combine into a **voice mode**.

For configuring the AI's voice, see [configuration](#configuration) below.

The AI avatar can optionally display machine-translated subtitles for its speech.

To speak to the AI, click the **mic button**. Once you are done talking, click again, or wait until the automatic silence detector ends the recording.

- voice mode (STT, TTS)

# GUI walkthrough

- where features are
  - why some buttons are at the bottom
  - why some buttons are below every chat message
    - why some buttons only appear for certain types of chat message
  - normally, the user sends a message, to which the AI then replies
    - Agent loop: the AI may call tools. If it does, once the tool call completes, control returns to the AI, and it can resume writing another message. Once there are no more tool calls, and the AI has written its final response, control returns to the user.
    - But you can send an empty message. This omits the user's turn, asking the AI to take that turn instead.
    - You can interrupt the AI generation, and resume (continue) it
      - Continuing can be useful also if the output token limit ran out before the AI was done replying
- what the GUI toggles below the avatar do
  - the toggles are persistent, default place to store the app state is `~/.config/raven/llmclient/state.json`

# Configuration

As explained in the main README, configuration is currently fed in as several Python modules that exist specifically as configuration files.

## Server connections

- LLM backend URL and API key: [`raven.librarian.config`](config.py)
  - Whether you need an API key depends on your LLM. By default, a local installation of [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) does **not** use an API key.

- Raven-server URL and API key: [`raven.client.config`](../client/config.py)
  - By default, *Raven-server* does **not** use an API key.
  - If you want to set up an API key for your *Raven-server*, see the `--secure` command line option of `raven-server`.
    - Note that this is a very light form of authentication that only requires providing a shared secret (the API key). The API key is transmitted in plain text.
    - Importantly, the `--secure` mode does **not** encrypt the connection.

## Voice mode

The AI's voice is configured in the AI avatar configuration.

- TTS is part of avatar config in [`raven.librarian.config`](config.py)
- STT model is configured in [`raven.server.config`](../server/config.py)
- for subtitles, machine translation model is selected in [`raven.server.config`](../server/config.py)
- audio devices are selected in [`raven.client.config`](../client/config.py); see also `raven-check-audio-devices` command-line tool to list devices present on your system

## System prompt, AI character personality, communication style

- [`raven.librarian.config`](config.py)
- technically, just a system prompt - this goes to the beginning of every chat
- but in practice, useful to think of it as *system prompt + AI character card* (the default out-of-the-box configuration does this)

## AI avatar

- character choice in [`raven.librarian.config`](config.py)
  - the AI avatar and the AI character name/personality are set up separately
  - to avoid surprises, make sure these match
- AI voice (TTS) is also configured in [`raven.librarian.config`](config.py)
- Use the GUI app `raven-avatar-settings-editor` to create or edit the `animator.json` configuration file (avatar video postprocessor settings)

# Future vision

Overall target: *100% local, personal co-researcher*

Areas to improve:

- **Long-term memory** for LLM
  - context engineering based
  - essentially a second document database instance, where a document = a chat message (with its node ID so we can walk links)
    - the DB's local search engine is very useful here
  - semiautomatic, but also with explicit tool access for the AI
  - undecided if we also need a third document database instance as an explicit memory storage place (read/write) for the LLM
- Improve **GUI to access old chats**
  - add recent chats list
    - as mentioned, "recent chat" is not really defined with the natively nonlinear format; maybe "user's first message" is a good enough splitting point
  - add tree view
    - N levels around chosen point, to keep rendering relatively fast
    - currently missing a suitable GUI widget in DPG; seems we'll have to roll our own (let's port [xdottir](https://github.com/Technologicat/xdottir) to DPG)
  - add search
    - incremental fragment search (like in *Raven-visualizer*)
    - use the local search engine here, too (since we need to search-index chats for the memory feature, anyway)
- Improve **document database**
  - scopes (AI research, engineering sciences, My Little Pony fanfics, ...)
  - PDF import
- Add **more tools** for the AI to call
  - explicit access to document database (not just RAG autosearch as it is now)
  - explicit access to memory (once the memory feature is added)
  - web download access to read web pages?
    - convenient for inputting recent talking points in fast-moving fields such as AI
    - potentially [dangerous](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
      - trusted link sources? User's messages only? Websearch results... a careful maybe?
      - untrusted link: always ask user, do not provide an option to disable the confirmation. (Provide option to auto-disallow all untrusted links, though.)
  - not sure if Librarian needs these:
    - calculator with [simpleeval](https://github.com/danthedeckie/simpleeval) sandbox?
    - weather with [open-meteo](https://open-meteo.com/en/docs)?
  - Keep It Simple Stupid: too many tools → confused LLM
- Add **source attribution** for Librarian's replies
  - for determinism and 100% reliability, handle this in the scaffold, not in the LLM
  - which documents or links the LLM saw when writing the response (even if the final reply didn't use all of the sources; this is the best this approach can do?)
    - maybe needs heuristic filtering; there can be e.g. 200 document database autosearch results if configured so
    - we could scan for document IDs (already stored in RAG autosearches) in the LLM's reply text, and then auto-cite any matching ones
- **Chat tagging**
  - tag a subtree as work, hobby, etc.
- **Message editing**
  - E.g. to fix typos before exporting a linearized chatlog
- Integration with *Raven-visualizer*
  - Integrate the document database with the semantic map visualization
  - Select data points in Visualizer, talk about those studies in Librarian
  - Ask Librarian a free-form question, let it highlight useful studies in Visualizer (based on document database search results)

# Appendix: Brief notes on how to set up a local LLM

- recommended for privacy
- setting up a local LLM with text-generation-webui (where to find install instructions; links to recommended models on HF; important command-line options)
