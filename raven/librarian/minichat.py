"""Minimal chat client for testing/debugging that Raven can connect to your LLM.

Although a "minimal" client, this does have some fancy features, such as GNU readline input history,
auto-persisted branching chat history, RAG (retrieval-augmented generation; query your plain-text documents),
and tool-calling support.

This module demonstrates how to build an LLM client using `raven.librarian.llmclient`.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-minichat version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import atexit
    import json
    import os
    import pathlib
    import platform
    import requests
    import sys
    from typing import Dict, List, Optional

    from mcpyrate import colorizer

    from unpythonic import sym, Values
    from unpythonic.env import env

    from .. import __version__

    from ..client import api
    from ..client import config as client_config

    from . import chattree
    from . import config as librarian_config
    from . import hybridir
    from . import llmclient
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

def minimal_chat_client(backend_url):
    """Minimal LLM chat client, for testing/debugging."""

    history_file = librarian_config.llmclient_userdata_dir / "history"      # user input history (readline)
    datastore_file = librarian_config.llmclient_userdata_dir / "data.json"  # chat node datastore
    state_file = librarian_config.llmclient_userdata_dir / "state.json"     # important node IDs for the chat client state

    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    datastore = None  # initialized later, during app startup

    def load_app_state(settings: env, datastore: chattree.PersistentForest) -> Dict:
        if datastore is None:
            assert False  # The `datastore` container must exist before this internal function is called

        def new_datastore():
            state["new_chat_HEAD"] = llmclient.factory_reset_chat_datastore(datastore, settings)  # do this first - this creates the first two nodes (system prompt with character card, and the AI's initial greeting)
            state["HEAD"] = state["new_chat_HEAD"]  # current last node in chat; like HEAD pointer in git

        try:
            with open(state_file, "r", encoding="utf-8") as json_file:
                state = json.load(json_file)
        except FileNotFoundError:
            new_datastore()
            state["docs_enabled"] = True

        if not datastore.nodes:  # No stored chat history -> reset datastore
            logger.warning("load_app_state: no chat nodes in datastore, creating new datastore")
            new_datastore()

        if "new_chat_HEAD" not in state:  # New-chat start node ID missing -> reset datastore
            logger.warning(f"load_app_state: missing key 'new_chat_HEAD' in '{state_file}', creating new datastore")
            new_datastore()

        if "HEAD" not in state:  # Current chat node ID missing -> start at new chat
            logger.warning(f"load_app_state: missing key 'HEAD' in '{state_file}', resetting it to 'new_chat_HEAD'")
            state["HEAD"] = state["new_chat_HEAD"]

        if "docs_enabled" not in state:
            logger.warning(f"load_app_state: missing key 'docs_enabled' in '{state_file}', using default")
            state["docs_enabled"] = True

        if "speculate_enabled" not in state:
            logger.warning(f"load_app_state: missing key 'speculate_enabled' in '{state_file}', using default")
            state["speculate_enabled"] = False

        # Refresh the system prompt in the datastore (to the one in this client's source code)
        new_chat_node_id = state["new_chat_HEAD"]
        system_prompt_node_id = datastore.nodes[new_chat_node_id]["parent"]
        old_system_prompt_revision_id = datastore.get_revision(node_id=system_prompt_node_id)
        datastore.add_revision(node_id=system_prompt_node_id,
                               payload={"message": llmclient.create_initial_system_message(settings)})
        datastore.delete_revision(node_id=system_prompt_node_id,
                                  revision_id=old_system_prompt_revision_id)

        llmclient.upgrade(datastore, system_prompt_node_id)  # v0.2.3+: data format change

        print(colorizer.colorize(f"Loaded app state from '{state_file}'.", colorizer.Style.BRIGHT))
        return state

    def save_app_state(state: Dict) -> None:
        # validate
        required_keys = ("new_chat_HEAD",
                         "HEAD",
                         "docs_enabled",
                         "speculate_enabled")
        if any(key not in state for key in required_keys):
            raise KeyError  # at least one required setting missing from `state`

        with open(state_file, "w", encoding="utf-8") as json_file:
            json.dump(state, json_file, indent=2)

    # Ugh for the presentation order, but this is needed in two places, starting immediately below.
    def chat_show_model_info():
        print(f"    {colorizer.colorize('Model', colorizer.Style.BRIGHT)}: {settings.model}")
        print(f"    {colorizer.colorize('Character', colorizer.Style.BRIGHT)}: {settings.char} [defined in this client]")
        print()

    # Main program
    try:
        # API key already loaded during module bootup; here, we just inform the user.
        if "Authorization" in llmclient.headers:
            print()
            print(f"Loaded LLM API key from '{str(librarian_config.llm_api_key_file)}'.")
            print()
        else:
            print()
            print(f"No LLM API key configured. If your LLM needs an API key to connect, put it into '{str(librarian_config.llm_api_key_file)}'.")
            print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
            print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
            print()

        try:
            llmclient.list_models(backend_url)  # just do something, to try to connect
        except requests.exceptions.ConnectionError as exc:
            print(colorizer.colorize(f"Cannot connect to LLM backend at {backend_url}.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
            msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
            logger.error(msg)
            sys.exit(255)
        else:
            print(colorizer.colorize(f"Connected to LLM backend at {backend_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
            settings = llmclient.setup(backend_url=backend_url)
            chat_show_model_info()

        # Persistent, branching chat history.
        datastore = chattree.PersistentForest(datastore_file)  # This autoloads and auto-persists.
        if datastore.nodes:
            print(colorizer.colorize(f"Loaded chat datastore from '{datastore_file}'.", colorizer.Style.BRIGHT))
        state = load_app_state(settings, datastore)
        print()

        # Load RAG database (it will auto-persist at app exit).
        retriever, _unused_scanner = hybridir.setup(docs_dir=docs_dir,
                                                    recursive=librarian_config.llm_docs_dir_recursive,
                                                    db_dir=db_dir,
                                                    embedding_model_name=librarian_config.qa_embedding_model)
        docs_enabled_str = "ON" if state["docs_enabled"] else "OFF"
        colorful_rag_status = colorizer.colorize(f"RAG (retrieval-augmented generation) autosearch is currently {docs_enabled_str}.",
                                                 colorizer.Style.BRIGHT)
        print(f"{colorful_rag_status} Toggle with the `!docs` command.")
        print(f"    Document store is at '{str(librarian_config.llm_docs_dir)}'.")
        # The retriever's `documents` attribute must be locked before accessing.
        with retriever.datastore_lock:
            plural_s = "s" if len(retriever.documents) != 1 else ""
            print(f"        {len(retriever.documents)} document{plural_s} loaded.")
        print(f"    Search indices are saved in '{str(librarian_config.llm_database_dir)}'.")
        print()

        import readline  # noqa: F401, side effect: enable GNU readline in builtin input()
        # import rlcompleter  # noqa: F401, side effects: readline tab completion for Python code
        print(colorizer.colorize(f"GNU readline available. Saving user inputs to '{str(history_file)}'.", colorizer.Style.BRIGHT))
        print(colorizer.colorize("Use up/down arrows to browse previous inputs. Enter to send. ", colorizer.Style.BRIGHT))
        print()
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        # Set up autosave at exit.
        def persist():
            librarian_config.llmclient_userdata_dir.mkdir(parents=True, exist_ok=True)

            # Save readline history
            readline.set_history_length(1000)
            readline.write_history_file(history_file)

            try:
                save_app_state(state)
            except KeyError:
                logger.warning(f"During app shutdown: app `state` missing at least one required key, cannot persist it. Existing keys: {list(state.keys())}")

            # Before saving (which happens automatically at exit),
            # remove any nodes not reachable from the initial message, and also remove dead links.
            # There shouldn't be any, but this way we exercise these features, too.
            try:
                new_chat_node_id = state["new_chat_HEAD"]
                system_prompt_node_id = datastore.nodes[new_chat_node_id]["parent"]
            except KeyError as exc:
                logger.warning(f"During app shutdown: while pruning chat forest: {type(exc)}: {exc}")
            else:
                datastore.prune_unreachable_nodes(system_prompt_node_id)
                datastore.prune_dead_links(system_prompt_node_id)
        # We register later than `chattree.PersistentForest` (which we already instantiated above), so ours runs first.
        # Hence we'll have the chance to prune before the forest is persisted to disk.
        #     https://docs.python.org/3/library/atexit.html
        atexit.register(persist)

        print(colorizer.colorize("Starting chat.", colorizer.Style.BRIGHT))
        print()
        def chat_show_help():
            print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
            print("    raven.librarian.minichat - Minimal LLM client for testing/debugging.")
            print()
            print("    Special commands (tab-completion available):")
            print("        !clear                  - Start new chat")
            print(f"        !docs [True|False]      - RAG autosearch on/off/toggle (currently {state['docs_enabled']}; document store at '{str(librarian_config.llm_docs_dir)}')")
            print(f"        !speculate [True|False] - LLM speculate on/off/toggle (currently {state['speculate_enabled']}); used only if docs is True.")
            print("                                  If speculate is False, try to use only RAG results to answer.")
            print("                                  If speculate is True, let the LLM respond however it wants.")
            print("        !dump                   - See raw contents of chat node datastore")
            print("        !head some-node-id      - Switch to another chat branch (get the node ID from `!dump`)")
            print("        !history                - Print a cleaned-up transcript of the current chat branch")
            print("        !model                  - Show which model is in use")
            print("        !models                 - List all models available at connected backend")
            print("        !help                   - Show this message again")
            print()
            print("    Press Ctrl+D to exit chat.")
            print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
            print()
        chat_show_help()

        # We prefill the space for commands that take an argument.
        commands = ["!clear",
                    "!docs ",
                    "!dump",
                    "!head ",
                    "!help",
                    "!history",
                    "!model",
                    "!models",
                    "!speculate"]
        def get_completions(candidates, text):
            """Return matching completions for `text`.

            `candidates`: Every possible completion the system knows of, in the current context.
            `text`: Prefix text to complete. Can be the empty string, which matches all candidates.
            """
            if not text:  # If no text, all candidates match.
                return candidates
            assert text  # we have text

            # Score the completions for the given prefix `text`.
            # https://stackoverflow.com/questions/6718196/determine-the-common-prefix-of-multiple-strings
            scores = [len(os.path.commonprefix([text, candidate])) for candidate in candidates]
            max_score = max(scores)
            if max_score == 0:  # no match
                return None
            assert max_score > 0  # we have at least one match, of at least one character

            # Possible completions are those that scored best (i.e. match the longest matched prefix).
            completions = [candidate for candidate, score in zip(candidates, scores) if score == max_score]
            return completions
        # https://docs.python.org/3/library/readline.html#readline-completion
        def completer(text, state):  # completer for special commands
            buffer_content = readline.get_line_buffer()  # context: text before the part being completed (up to last delim)

            # TODO: fix one more failure mode, e.g. "!help !<tab>"
            if buffer_content.startswith("!") and text.startswith("!"):  # completing a command?
                candidates = commands
            elif buffer_content.startswith("!docs"):  # in `!docs` command, expecting an argument?
                candidates = ["True", "False"]
            elif buffer_content.startswith("!head"):  # in `!head` command, expecting an argument?
                candidates = list(sorted(datastore.nodes.keys()))
            elif buffer_content.startswith("!speculate"):  # in `!speculate` command, expecting an argument?
                candidates = ["True", "False"]
            else:  # anything else -> no completions
                return None

            completions = get_completions(candidates, text)
            if completions is None:  # no match
                return None
            if state >= len(completions):  # no more completions
                return None
            return completions[state]
        readline.set_completer(completer)
        readline.set_completer_delims(" ")

        # Support tab completion also on MacOSX. Not sure which way is better here.
        # Neither seems The Right Thing:
        #   - Detecting the platform (as we do now) assumes that MacOSX will always use `libedit`
        #     to provide `readline`.
        #   - Detecting the `readline` module's `__doc__` assumes that any future versions of `libedit`
        #     keep the mention of `libedit` in the docstring.
        # https://stackoverflow.com/questions/7116038/python-repl-tab-completion-on-macos
        # https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on
        #
        # if 'libedit' in readline.__doc__:  # MacOSX uses libedit, not GNU readline
        #     readline.parse_and_bind("bind ^I rl_complete")
        # else:  # Linux, Windows
        #     readline.parse_and_bind("tab: complete")
        if platform.system() == "Darwin":  # MacOSX
            readline.parse_and_bind("bind ^I rl_complete")
        else:  # "Linux", "Windows"
            readline.parse_and_bind("tab: complete")

        def chat_show_list_of_models():
            available_models = llmclient.list_models(backend_url)
            print(colorizer.colorize("    Available models:", colorizer.Style.BRIGHT))
            for model_name in available_models:
                print(f"        {model_name}")
            print()

        # TODO: we don't need the `color=False` option now that we fixed `colorize` to work with `readline`/`input`.
        def format_message_number(message_number: Optional[int], color: bool) -> None:
            if message_number is not None:
                out = f"[#{message_number}]"
                if color:
                    out = colorizer.colorize(out, colorizer.Style.DIM)
                return out
            return ""

        def format_persona(role: str, color: bool) -> None:
            persona = settings.role_names.get(role, None)
            if persona is None:
                out = f"<<{role}>>"  # currently, this include "<<system>>" and "<<tool>>"
                if color:
                    out = colorizer.colorize(out, colorizer.Style.DIM)
                return out
            else:
                out = persona
                if color:
                    out = colorizer.colorize(out, colorizer.Style.BRIGHT)
                return out

        def format_message_heading(message_number: Optional[int], role: str, color: bool):
            colorful_number = format_message_number(message_number, color)
            colorful_persona = format_persona(role, color)
            if message_number is not None:
                return f"{colorful_number} {colorful_persona}: "
            else:
                return f"{colorful_persona}: "

        def chat_print_message(message_number: Optional[int], role: str, text: str) -> None:
            print(format_message_heading(message_number, role, color=True), end="")
            print(llmclient.remove_role_name_from_start_of_line(settings=settings, role=role, text=text))

        def chat_print_history(history: List[Dict], show_numbers: bool = True) -> None:
            if show_numbers:
                for k, message in enumerate(history):
                    chat_print_message(message_number=k, role=message["role"], text=message["content"])
                    print()
            else:
                for message in history:
                    chat_print_message(message_number=None, role=message["role"], text=message["content"])
                    print()

        action_proceed = sym("proceed")  # proceed current round as normal
        action_next_round = sym("next_round")  # skip to start of next round, e.g. after a special command

        def user_turn() -> Values:
            history = llmclient.linearize_chat(datastore, state["HEAD"])
            user_message_number = len(history)

            # Print a user input prompt and get the user's input.
            #
            # The `readline` module takes its user input prompt from what we supply to `input`, so we must print the prompt via `input`, colors and all.
            # The colorizer automatically wraps the ANSI color escape codes (for the terminal app) in ASCII escape codes (for `readline` itself)
            # that tell `readline` not to include them in its visual length calculation.
            #
            # This avoids the input prompt getting overwritten when browsing history entries, and prevents backspacing over the input prompt.
            # https://stackoverflow.com/questions/75987688/how-can-readline-be-told-not-to-erase-externally-supplied-prompt
            input_prompt = format_message_heading(user_message_number, role="user", color=True)
            user_message_text = input(input_prompt)
            print()

            # Interpret special commands for this LLM client
            if user_message_text == "!clear":
                print(colorizer.colorize("Starting new chat session.", colorizer.Style.BRIGHT))
                state["HEAD"] = state["new_chat_HEAD"]
                print(f"HEAD is now at '{state['HEAD']}'.")
                print()
                chat_print_history(llmclient.linearize_chat(datastore, state["HEAD"]))
                return Values(action=action_next_round)
            elif user_message_text.startswith("!docs"):  # TODO: refactor
                split_command_text = user_message_text.split()
                nargs = len(split_command_text) - 1
                if nargs == 0:
                    state["docs_enabled"] = not state["docs_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        state["docs_enabled"] = True
                    elif arg == "False":
                        state["docs_enabled"] = False
                    else:
                        print(f"!docs: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action_next_round)
                else:
                    print("!docs: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                docs_enabled_str = "ON" if state["docs_enabled"] else "OFF"
                print(f"RAG autosearch is now {docs_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text == "!dump":
                print(colorizer.colorize("Raw datastore content:", colorizer.Style.BRIGHT) + f" (current HEAD is at {state['HEAD']})")
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print(f"{datastore}", end="")  # -> str; also, already has the final blank line
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!head"):  # switch to another chat branch
                try:
                    _, new_head_id = user_message_text.split()
                except ValueError:
                    print("!head: wrong number of arguments; expected exactly one, the node ID to switch to; see `!dump` for available chat nodes.")
                    print()
                    return Values(action=action_next_round)
                if new_head_id not in datastore.nodes:
                    print(f"!head: no such chat node '{new_head_id}'; see `!dump` for available chat nodes.")
                    print()
                    return Values(action=action_next_round)
                state["HEAD"] = new_head_id
                print(f"HEAD is now at '{state['HEAD']}'.")
                print()
                chat_print_history(llmclient.linearize_chat(datastore, state["HEAD"]))
                return Values(action=action_next_round)
            elif user_message_text == "!help":
                chat_show_help()
                return Values(action=action_next_round)
            elif user_message_text == "!history":
                print(colorizer.colorize("Chat history (cleaned up):", colorizer.Style.BRIGHT))
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                chat_print_history(history)
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print()
                return Values(action=action_next_round)
            elif user_message_text == "!model":
                chat_show_model_info()
                return Values(action=action_next_round)
            elif user_message_text == "!models":
                chat_show_list_of_models()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!speculate"):  # TODO: refactor
                split_command_text = user_message_text.split()
                nargs = len(split_command_text) - 1
                if nargs == 0:
                    state["speculate_enabled"] = not state["speculate_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        state["speculate_enabled"] = True
                    elif arg == "False":
                        state["speculate_enabled"] = False
                    else:
                        print(f"!speculate: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action=action_next_round)
                else:
                    print("!speculate: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                speculate_enabled_str = "ON" if state["speculate_enabled"] else "OFF"
                print(f"LLM speculation is now {speculate_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!") and len(user_message_text.split("\n")) == 1:
                print(f"Unrecognized command '{user_message_text}'; use `!help` for available commands.")
                return Values(action=action_next_round)
            # Not a special command.

            # Add the user's message to the chat.
            user_message_node_id = datastore.create_node(payload={"message": llmclient.create_chat_message(settings=settings,
                                                                                                           role="user",
                                                                                                           text=user_message_text)},
                                                         parent_id=state["HEAD"])
            state["HEAD"] = user_message_node_id
            return Values(action=action_proceed, text=user_message_text)

        def rag_search_with_bypass(query: str) -> Values:
            if not state["docs_enabled"]:
                return Values(action=action_proceed, matches=[])

            docs_results = retriever.query(query,
                                           k=10,
                                           return_extra_info=False)

            # First line of defense (against hallucinations): docs on, no matches for given query, speculate off -> bypass LLM
            if not docs_results and not state["speculate_enabled"]:
                nomatch_text = "No matches in knowledge base. Please try another query."
                nomatch_message_node_id = datastore.create_node(payload={"message": llmclient.create_chat_message(settings=settings,
                                                                                                                  role="assistant",
                                                                                                                  text=nomatch_text)},
                                                                parent_id=state["HEAD"])
                nomatch_message_node_payload = datastore.get_payload(nomatch_message_node_id)
                nomatch_message_node_payload["retrieval"] = {"query": query,
                                                             "results": []}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
                state["HEAD"] = nomatch_message_node_id

                history = llmclient.linearize_chat(datastore, state["HEAD"])
                nomatch_message_number = len(history)
                chat_print_message(message_number=nomatch_message_number,
                                   role="assistant",
                                   text=nomatch_text)
                print()

                return Values(action=action_next_round)

            return Values(action=action_proceed, matches=docs_results)

        # Perform the temporary injects. These are not meant to be persistent, so we don't even add them
        # as nodes to the chat tree, but only into the temporary linearized history.
        injectors = [llmclient.format_chat_datetime_now,  # let the LLM know the current local time and date
                     llmclient.format_reminder_to_focus_on_latest_input]  # remind the LLM to focus on user's last message (some models such as the distills of DeepSeek-R1 need this to support multi-turn conversation)
        def perform_injects(history: List[Dict], docs_matches: List[Dict]) -> None:
            # # This causes Qwen3 to miss the user's last message. Maybe better to put the RAG results at another position.
            # #
            # # Format RAG results like a tool-call reply to the user's message.
            # # First, find the user's latest message in the linearized history.
            # for depth, message in enumerate(reversed(history)):
            #     if message["role"] == "user":
            #         break
            # else:  # no user message found (should not happen)
            #     depth = None
            #     message = None
            #
            # if message is not None:
            #     position = len(history) - depth
            #     for docs_result in reversed(docs_matches):  # reverse to keep original order, because we insert each item at the same position.
            #         # TODO: Should the RAG match notification show the query string, too?
            #         search_result_text = f"Knowledge-base match from '{docs_result['document_id']}':\n\n{docs_result['text'].strip()}\n-----"
            #         message_to_inject = llmclient.create_chat_message(settings=settings,
            #                                                           role="tool",
            #                                                           text=search_result_text)
            #         history.insert(position, message_to_inject)

            # Insert RAG results at the start of the history, as system messages.
            for docs_result in reversed(docs_matches):  # reverse to keep original order, because we insert each item at the same position.
                # TODO: Should the RAG match notification show the query string, too?
                search_result_text = f"[System information: Knowledge-base match from '{docs_result['document_id']}'.]\n\n{docs_result['text'].strip()}\n-----"
                message_to_inject = llmclient.create_chat_message(settings=settings,
                                                                  role="system",
                                                                  text=search_result_text)
                history.insert(1, message_to_inject)  # after system prompt / character card combo

            # Always-on injects, e.g. current local datetime
            for thunk in injectors:
                message_to_inject = llmclient.create_chat_message(settings=settings,
                                                                  role="system",
                                                                  text=thunk())
                history.append(message_to_inject)

            # If docs on, speculate off (-> `perform_injects` gets called if there is at least one RAG match), remind the LLM to use information from context only.
            #                           This increases the changes of the user's query working correctly when the search returns irrelevant results.
            # If docs off, the whole point is to use the LLM's static knowledge, so in that case don't bother.
            if state["docs_enabled"] and not state["speculate_enabled"]:
                message_to_inject = llmclient.create_chat_message(settings=settings,
                                                                  role="system",
                                                                  text=llmclient.format_reminder_to_use_information_from_context_only())
                history.append(message_to_inject)

            # # DEBUG - show history with injects.
            # # Message numbers counted from the modified history (with injects) would be wrong, so don't show them.
            # chat_print_history(history, show_numbers=False)

        def ai_turn(user_message_text: str) -> Values:
            # Perform the RAG autosearch (if enabled; will check automatically).
            # If docs is on, no match, and speculate is off -> bypass the LLM.
            #
            # NOTE: This is very rudimentary.
            #   - We simply use the user's new message as-is as the query.
            #   - Hence, this does NOT match on any earlier message, and may result in spurious matches.
            #     E.g. "Can cats jump?" and "Does your knowledge base say if cats can jump?" return
            #     different results, because the term "knowledge base" in the latter may match e.g.
            #     AI/CS articles that the user happens to have included in the KB.
            #     - In this example, with the example data, the shorter query correctly returns no matches.
            #     - The longer query returns two AI agent abstracts, leaving it to the LLM to put the
            #       pieces together and notice that the user's query and provided KB context don't actually match.
            #   - This could be improved by querying the LLM itself - "given the chat history so far and
            #     the user's most recent message, please formulate query terms for a knowledge base search."
            #     and then run the search with the final output of that.
            #   - We could also build a slightly more complex scaffold to support tool-calling,
            #     and instruct the LLM to send a query when it itself thinks it needs to.
            rag_query = user_message_text
            rag_result = rag_search_with_bypass(query=rag_query)
            if rag_result["action"] is action_next_round:  # bypass triggered
                return Values(action=action_next_round)

            # AI's turn: LLM generation interleaved with tool responses, until there are no tool calls in the LLM's latest reply.
            while True:
                history = llmclient.linearize_chat(datastore, state["HEAD"])  # latest history
                ai_message_number = len(history)

                # Prepare the final LLM prompt, by including the temporary injects.
                perform_injects(history, docs_matches=rag_result["matches"])

                # Invoke the LLM.
                print(format_message_number(ai_message_number, color=True))
                chars = 0
                def progress_callback(n_chunks, chunk_text):  # any UI live-update code goes here, in the callback
                    # TODO: think of a better way to split to lines
                    nonlocal chars
                    chars += len(chunk_text)
                    if "\n" in chunk_text:  # one token at a time; should have either one linefeed or no linefeed
                        chars = 0  # good enough?
                    elif chars >= librarian_config.llm_line_wrap_width:
                        print()
                        chars = 0
                    print(chunk_text, end="")
                    sys.stdout.flush()
                # `invoke` uses a linearized history, as expected by the LLM API.
                out = llmclient.invoke(settings, history, progress_callback)  # `out.data` is now the complete message object (in the format returned by `create_chat_message`)
                print()  # print the final newline

                # Clean up the LLM's reply (heuristically). This version goes into the chat history.
                out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=True)

                # Show LLM performance statistics
                print(colorizer.colorize(f"[{out.n_tokens}t, {out.dt:0.2f}s, {out.n_tokens/out.dt:0.2f}t/s]", colorizer.Style.DIM))
                print()

                # Add the LLM's message to the chat.
                #
                # Note the token count of the message actually saved into the chat log may be different from `out.n_tokens`, e.g. if the AI is interrupted or when thoughts blocks are discarded.
                # However, to correctly compute the generation speed, we need to use the original count before any editing, since `out.dt` was measured for that.
                ai_message_node_id = datastore.create_node(payload={"message": out.data,
                                                                    "generation_metadata": {"model": out.model,
                                                                                            "n_tokens": out.n_tokens,  # could count final tokens with `llmclient.token_count(settings, out.data["content"])`
                                                                                            "dt": out.dt}},
                                                           parent_id=state["HEAD"])
                ai_message_node_payload = datastore.get_payload(ai_message_node_id)
                if state["docs_enabled"]:
                    ai_message_node_payload["retrieval"] = {"query": rag_query,
                                                            "results": rag_result["matches"]}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
                state["HEAD"] = ai_message_node_id

                # Handle tool calls requested by the LLM, if any.
                #
                # Call the tool(s) specified by the LLM, with arguments specified by the LLM, and add the result to the chat.
                #
                # Each response goes into its own message, with `role="tool"`.
                #
                tool_message_number = ai_message_number + 1
                tool_response_records = llmclient.perform_tool_calls(settings, message=out.data)

                # When there are no more tool calls, the LLM is done replying.
                # Each tool call produces exactly one response, so we may as well check this from the number of responses.
                if not tool_response_records:
                    break

                # Add the tool response messages to the chat.
                for tool_response_record in tool_response_records:
                    payload = {"message": tool_response_record.data,
                               "generation_metadata": {"status": tool_response_record.status}}  # status is "success" or "error"
                    if "toolcall_id" in tool_response_record:
                        payload["generation_metadata"]["toolcall_id"] = tool_response_record.toolcall_id
                    if "dt" in tool_response_record:
                        payload["generation_metadata"]["dt"] = tool_response_record.dt
                    tool_response_message_node_id = datastore.create_node(payload=payload,
                                                                          parent_id=state["HEAD"])
                    state["HEAD"] = tool_response_message_node_id

                    chat_print_message(message_number=tool_message_number,
                                       role="tool",
                                       text=tool_response_record.data["content"])
                    print()

                    tool_message_number += 1

                # # DEBUG - show history after the tool calls, before the LLM starts writing again.
                # history = llmclient.linearize_chat(datastore, state["HEAD"])
                # chat_print_history(history, show_numbers=False)

            return Values(action=action_proceed)

        # Show initial history (loaded from datastore, or blank upon first start)
        chat_print_history(llmclient.linearize_chat(datastore, state["HEAD"]))

        # Main loop
        while True:
            user_result = user_turn()
            if user_result["action"] is action_next_round:
                continue

            # The AI needs the text of the user's latest message for the RAG autosearch query.
            ai_result = ai_turn(user_message_text=user_result["text"])
            if ai_result["action"] is action_next_round:
                continue  # Silly, since this is the last thing in the loop, but for symmetry.

    except (EOFError, KeyboardInterrupt):
        print()
        print(colorizer.colorize("Exiting chat.", colorizer.Style.BRIGHT))
        print()

def main():
    parser = argparse.ArgumentParser(description="""Minimal LLM chat client, for testing/debugging. You can use this for testing that Raven can connect to your LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="backend_url", nargs="?", default=librarian_config.llm_backend_url, type=str, metavar="url", help=f"where to access the LLM API (default, currently '{librarian_config.llm_backend_url}', is set in `raven/librarian/config.py`)")
    opts = parser.parse_args()

    print()
    if api.raven_server_available():
        print(colorizer.colorize(f"Connected to Raven-server at {client_config.raven_server_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
        print(colorizer.colorize("The LLM will have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
    else:
        print(colorizer.colorize(f"WARNING: Cannot connect to Raven-server at {client_config.raven_server_url}", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))
        print(colorizer.colorize("The LLM will NOT have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))

    # print(llmclient.websearch_wrapper("what is the airspeed velocity of an unladen swallow"))  # DEBUG
    minimal_chat_client(opts.backend_url)

if __name__ == "__main__":
    main()
