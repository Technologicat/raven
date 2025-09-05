"""Small utilities for LLM chat clients."""

__all__ = ["format_message_number",
           "format_persona",
           "format_message_heading"]

from typing import Optional

from mcpyrate import colorizer

from unpythonic.env import env

def _yell_if_unsupported_markup(markup):
    if markup not in ("ansi", "markdown", None):
        raise ValueError(f"unknown markup kind '{markup}'; valid values: 'ansi' (*nix terminal), 'markdown', and the special value `None`.")

def format_message_number(message_number: Optional[int],
                          markup: Optional[str]) -> None:
    _yell_if_unsupported_markup(markup)
    if message_number is not None:
        out = f"[#{message_number}]"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"*{out}*"
        return out
    return ""

def format_persona(llm_settings: env,
                   role: str,
                   markup: Optional[str]) -> None:
    _yell_if_unsupported_markup(markup)
    persona = llm_settings.role_names.get(role, None)
    if persona is None:
        out = f"<<{role}>>"  # currently, this include "<<system>>" and "<<tool>>"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"`{out}`"  # use verbatim mode; otherwise looks like an HTML tag
        return out
    else:
        out = persona
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.BRIGHT)
        elif markup == "markdown":
            out = f"**{out}**"
        return out

def format_message_heading(llm_settings: env,
                           message_number: Optional[int],
                           role: str,
                           markup: Optional[str]):
    _yell_if_unsupported_markup(markup)
    markedup_number = format_message_number(message_number, markup)
    markedup_persona = format_persona(llm_settings, role, markup)
    if message_number is not None:
        return f"{markedup_number} {markedup_persona}: "
    else:
        return f"{markedup_persona}: "
