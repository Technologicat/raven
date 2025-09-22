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
    import os
    import pathlib
    import platform
    import requests
    import sys
    from typing import Dict, List, Optional

    from mcpyrate import colorizer

    from unpythonic import sym, Values

    from .. import __version__

    from ..client import api
    from ..client import config as client_config

    from . import appstate
    from . import chatutil
    from . import config as librarian_config
    from . import hybridir
    from . import llmclient
    from . import scaffold
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")
print()

def minimal_chat_client(backend_url) -> None:
    """Minimal LLM chat client, for testing/debugging."""

    history_file = librarian_config.llmclient_userdata_dir / "history"      # user input history (readline)
    datastore_file = librarian_config.llmclient_userdata_dir / "data.json"  # chat node datastore
    state_file = librarian_config.llmclient_userdata_dir / "state.json"     # important node IDs for the chat client state

    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    datastore = None  # initialized later, during app startup

    # Ugh for the presentation order, but this is needed in two places, starting immediately below.
    def chat_show_model_info() -> None:
        print(f"    {colorizer.colorize('Model', colorizer.Style.BRIGHT)}: {llm_settings.model}")
        print(f"    {colorizer.colorize('Character', colorizer.Style.BRIGHT)}: {llm_settings.char} [defined in this client]")
        print()

    # Main program
    try:
        if api.raven_server_available():
            # Websearch is set up as a tool in `raven.librarian.llmclient`, and the same module handles the communication
            # with the Raven server when the LLM performs a websearch tool-call. Here we just inform the user.
            print(colorizer.colorize(f"Connected to Raven-server at {client_config.raven_server_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
            print(colorizer.colorize("The LLM will have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
            print()
        else:
            print(colorizer.colorize(f"WARNING: Cannot connect to Raven-server at {client_config.raven_server_url}", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))
            print(colorizer.colorize("The LLM will NOT have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))
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
            llm_settings = llmclient.setup(backend_url=backend_url)
            chat_show_model_info()

        # API key already loaded during module bootup; here, we just inform the user.
        if "Authorization" in llmclient.headers:
            print(f"{colorizer.Fore.GREEN}{colorizer.Style.BRIGHT}Loaded LLM API key from '{str(librarian_config.llm_api_key_file)}'.{colorizer.Style.RESET_ALL}")
            print()
        else:
            print(f"{colorizer.Fore.YELLOW}{colorizer.Style.BRIGHT}No LLM API key configured.{colorizer.Style.RESET_ALL} If your LLM needs an API key to connect, put it into '{str(librarian_config.llm_api_key_file)}'.")
            print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
            print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
            print()

        # Persistent, branching chat history, and app settings (these will auto-persist at app exit).
        datastore, app_state = appstate.load(llm_settings, datastore_file, state_file)
        print()

        # Load RAG database (it will auto-persist at app exit).
        retriever, _unused_scanner = hybridir.setup(docs_dir=docs_dir,
                                                    recursive=librarian_config.llm_docs_dir_recursive,
                                                    db_dir=db_dir,
                                                    embedding_model_name=librarian_config.qa_embedding_model)
        docs_enabled_str = "ON" if app_state["docs_enabled"] else "OFF"
        colorful_rag_status = colorizer.colorize(f"Document database (retrieval-augmented generation, RAG) is currently {docs_enabled_str}.",
                                                 colorizer.Style.BRIGHT)
        print(f"{colorful_rag_status} Toggle with the `!docs` command.")
        print(f"    Its document store is at '{str(librarian_config.llm_docs_dir)}' (put your plain-text documents here).")
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
        def persist() -> None:
            librarian_config.llmclient_userdata_dir.mkdir(parents=True, exist_ok=True)

            # Save readline history
            readline.set_history_length(1000)
            readline.write_history_file(history_file)

            # Before saving the chat database (which happens automatically at exit),
            # remove any nodes not reachable from the initial message, and also remove dead links.
            # There shouldn't be any, but this way we exercise these features, too.
            try:
                new_chat_node_id = app_state["new_chat_HEAD"]
                system_prompt_node_id = datastore.get_parent(new_chat_node_id)
            except KeyError as exc:
                logger.warning(f"During app shutdown: while pruning chat forest: {type(exc)}: {exc}")
            else:
                datastore.prune_unreachable_nodes(system_prompt_node_id)
                datastore.prune_dead_links(system_prompt_node_id)
        # We register later than `datastore` does (which `appstate.load` sets up), so ours runs first.
        # Hence we'll have the chance to prune before the forest is persisted to disk.
        #     https://docs.python.org/3/library/atexit.html
        atexit.register(persist)

        print(colorizer.colorize("Starting chat.", colorizer.Style.BRIGHT))
        print()
        def chat_show_help() -> None:
            print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
            print("    raven.librarian.minichat - Minimal LLM client for testing/debugging.")
            print()
            print("    Special commands (tab-completion available):")
            print("        !clear                  - Start new chat")
            print(f"        !docs [True|False]      - Document database on/off/toggle (currently {app_state['docs_enabled']}; document store at '{str(librarian_config.llm_docs_dir)}')")
            print(f"        !speculate [True|False] - LLM speculate on/off/toggle (currently {app_state['speculate_enabled']}); used only if docs is True.")
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
        def get_completions(candidates: List[str], text: str) -> List[str]:
            """Return a list of matching completions for `text`.

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
        def completer(text: str, state: int) -> str:  # completer for special commands
            buffer_content = readline.get_line_buffer()  # context: text before the part being completed (up to last delim)

            # TODO: fix one more failure mode, e.g. "!help !<tab>"
            if buffer_content.startswith("!") and text.startswith("!"):  # completing a command?
                candidates = commands
            elif buffer_content.startswith("!docs"):  # in `!docs` command, expecting an argument?
                candidates = ["True", "False"]
            elif buffer_content.startswith("!head"):  # in `!head` command, expecting an argument?
                with datastore.lock:
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

        def chat_show_list_of_models() -> None:
            available_models = llmclient.list_models(backend_url)
            print(colorizer.colorize("    Available models:", colorizer.Style.BRIGHT))
            for model_name in available_models:
                print(f"        {model_name}")
            print()

        def chat_print_message(message_number: Optional[int], role: str, text: str) -> None:
            print(chatutil.format_message_heading(llm_settings=llm_settings,
                                                  message_number=message_number,
                                                  role=role,
                                                  markup="ansi"),
                  end="")
            print(chatutil.remove_role_name_from_start_of_line(llm_settings=llm_settings, role=role, text=text))

        def chat_print_history(history: List[Dict], show_numbers: bool = True) -> None:
            if show_numbers:
                for k, message in enumerate(history):
                    chat_print_message(message_number=k, role=message["role"], text=message["content"])
                    print()
            else:
                for message in history:
                    chat_print_message(message_number=None, role=message["role"], text=message["content"])
                    print()

        action_proceed = sym("proceed")  # proceed current round as usual
        action_next_round = sym("next_round")  # skip to start of next round (after the user entered a special command)

        def user_turn() -> Values:
            history = chatutil.linearize_chat(datastore, app_state["HEAD"])
            user_message_number = len(history)

            # Print a user input prompt and get the user's input.
            #
            # The `readline` module takes its user input prompt from what we supply to `input`, so we must print the prompt via `input`, colors and all.
            # The colorizer automatically wraps the ANSI color escape codes (for the terminal app) in ASCII escape codes (for `readline` itself)
            # that tell `readline` not to include them in its visual length calculation.
            #
            # This avoids the input prompt getting overwritten when browsing history entries, and prevents backspacing over the input prompt.
            # https://stackoverflow.com/questions/75987688/how-can-readline-be-told-not-to-erase-externally-supplied-prompt
            input_prompt = chatutil.format_message_heading(llm_settings=llm_settings,
                                                           message_number=user_message_number,
                                                           role="user",
                                                           markup="ansi")
            user_message_text = input(input_prompt)
            print()

            # Interpret special commands for this LLM client
            if user_message_text == "!clear":
                print(colorizer.colorize("Starting new chat session.", colorizer.Style.BRIGHT))
                app_state["HEAD"] = app_state["new_chat_HEAD"]
                print(f"HEAD is now at '{app_state['HEAD']}'.")
                print()
                chat_print_history(chatutil.linearize_chat(datastore, app_state["HEAD"]))
                return Values(action=action_next_round)
            elif user_message_text.startswith("!docs"):  # TODO: refactor
                split_command_text = user_message_text.split()
                nargs = len(split_command_text) - 1
                if nargs == 0:
                    app_state["docs_enabled"] = not app_state["docs_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        app_state["docs_enabled"] = True
                    elif arg == "False":
                        app_state["docs_enabled"] = False
                    else:
                        print(f"!docs: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action_next_round)
                else:
                    print("!docs: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                docs_enabled_str = "ON" if app_state["docs_enabled"] else "OFF"
                print(f"Document database is now {docs_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text == "!dump":
                print(colorizer.colorize("Raw datastore content:", colorizer.Style.BRIGHT) + f" (current HEAD is at {app_state['HEAD']})")
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
                app_state["HEAD"] = new_head_id
                print(f"HEAD is now at '{app_state['HEAD']}'.")
                print()
                chat_print_history(chatutil.linearize_chat(datastore, app_state["HEAD"]))
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
                    app_state["speculate_enabled"] = not app_state["speculate_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        app_state["speculate_enabled"] = True
                    elif arg == "False":
                        app_state["speculate_enabled"] = False
                    else:
                        print(f"!speculate: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action=action_next_round)
                else:
                    print("!speculate: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                speculate_enabled_str = "ON" if app_state["speculate_enabled"] else "OFF"
                print(f"LLM speculation is now {speculate_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!") and len(user_message_text.split("\n")) == 1:
                print(f"Unrecognized command '{user_message_text}'; use `!help` for available commands.")
                return Values(action=action_next_round)
            # Not a special command.

            # Add the user's message to the chat.
            new_head_node_id = scaffold.user_turn(llm_settings=llm_settings,
                                                  datastore=datastore,
                                                  head_node_id=app_state["HEAD"],
                                                  user_message_text=user_message_text)
            app_state["HEAD"] = new_head_node_id
            return Values(action=action_proceed, text=user_message_text)

        def ai_turn(user_message_text: str) -> Values:
            # NOTE: Rudimentary approach to RAG search, using the user's message text as the query. (Good enough to demonstrate the functionality. Improve later.)
            docs_query = user_message_text if app_state["docs_enabled"] else None

            history = chatutil.linearize_chat(datastore, app_state["HEAD"])  # latest history (ugh, we only need this here to get its length, for the sequential message number)
            ai_message_number = len(history)

            def on_llm_start() -> None:
                nonlocal ai_message_number  # for documenting intent only
                print(chatutil.format_message_number(ai_message_number, markup="ansi"))

            chars = 0
            def on_llm_progress(n_chunks: int, chunk_text: str) -> None:
                nonlocal chars
                chars += len(chunk_text)
                if "\n" in chunk_text:  # one token at a time; should have either one linefeed or no linefeed
                    chars = 0  # good enough?
                elif chars >= librarian_config.llm_line_wrap_width:  # TODO: think of a better way to split to lines
                    print()
                    chars = 0
                print(chunk_text, end="")
                sys.stdout.flush()
                return llmclient.action_ack  # let the LLM keep generating (we could return `action_stop` to interrupt the LLM, keeping the content received so far)

            def on_llm_done(node_id: str) -> None:
                nonlocal ai_message_number

                app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls

                print()  # Print the final newline

                # Show LLM performance statistics
                ai_message_node_payload = datastore.get_payload(node_id)
                n_tokens = ai_message_node_payload["generation_metadata"]["n_tokens"]
                dt = ai_message_node_payload["generation_metadata"]["dt"]
                speed = n_tokens / dt
                print(colorizer.colorize(f"[{n_tokens}t, {dt:0.2f}s, {speed:0.2f}t/s]", colorizer.Style.DIM))
                print()

                ai_message_number += 1

            def on_nomatch_done(node_id: str) -> None:
                nomatch_message_node_payload = datastore.get_payload(node_id)
                chat_print_message(message_number=ai_message_number,
                                   role="assistant",
                                   text=nomatch_message_node_payload["message"]["content"])
                print()

            def on_tool_done(node_id: str) -> None:
                nonlocal ai_message_number

                app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls

                nomatch_message_node_payload = datastore.get_payload(node_id)
                chat_print_message(message_number=ai_message_number,
                                   role="tool",
                                   text=nomatch_message_node_payload["message"]["content"])
                print()

                ai_message_number += 1

            new_head_node_id = scaffold.ai_turn(llm_settings=llm_settings,
                                                datastore=datastore,
                                                retriever=retriever,
                                                head_node_id=app_state["HEAD"],
                                                docs_query=docs_query,
                                                speculate=app_state["speculate_enabled"],
                                                markup="ansi",
                                                on_docs_start=None,
                                                on_docs_done=None,
                                                on_prompt_ready=None,  # debug/info hook
                                                on_llm_start=on_llm_start,
                                                on_llm_progress=on_llm_progress,
                                                on_llm_done=on_llm_done,
                                                on_nomatch_done=on_nomatch_done,
                                                on_tools_start=None,
                                                on_tool_done=on_tool_done,
                                                on_tools_done=None)
            app_state["HEAD"] = new_head_node_id
            return Values(action=action_proceed)

        # Show initial history (loaded from datastore, or blank upon first start)
        chat_print_history(chatutil.linearize_chat(datastore, app_state["HEAD"]))

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

def main() -> None:
    parser = argparse.ArgumentParser(description="""Minimal LLM chat client, for testing/debugging. You can use this for testing that Raven can connect to your LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="backend_url", nargs="?", default=librarian_config.llm_backend_url, type=str, metavar="url", help=f"where to access the LLM API (default, currently '{librarian_config.llm_backend_url}', is set in `raven/librarian/config.py`)")
    opts = parser.parse_args()

    # print(llmclient.websearch_wrapper("what is the airspeed velocity of an unladen swallow"))  # DEBUG
    minimal_chat_client(opts.backend_url)

if __name__ == "__main__":
    main()
