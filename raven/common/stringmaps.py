"""String normalization.

Currently, conversions between unicode subscript and superscript characters and their corresponding regular characters.
"""

regular_to_subscript_numbers = {"0": "₀",
                                "1": "₁",
                                "2": "₂",
                                "3": "₃",
                                "4": "₄",
                                "5": "₅",
                                "6": "₆",
                                "7": "₇",
                                "8": "₈",
                                "9": "₉"}
subscript_to_regular_numbers = {v: k for k, v in regular_to_subscript_numbers.items()}

regular_to_subscript_symbols = {"+": "₊",
                                "-": "₋",
                                "=": "₌",
                                "(": "₍",
                                ")": "₎"}
subscript_to_regular_symbols = {v: k for k, v in regular_to_subscript_symbols.items()}

regular_to_subscript_letters = {"a": "ₐ",
                                "e": "ₑ",
                                "ə": "ₔ",  # latin small letter schwa
                                "h": "ₕ",
                                "i": "ᵢ",
                                "j": "ⱼ",
                                "k": "ₖ",
                                "l": "ₗ",
                                "m": "ₘ",
                                "n": "ₙ",
                                "o": "ₒ",
                                "p": "ₚ",
                                "r": "ᵣ",
                                "s": "ₛ",
                                "t": "ₜ",
                                "u": "ᵤ",
                                "v": "ᵥ",
                                "x": "ₓ",
                                "β": "ᵦ",
                                "γ": "ᵧ",
                                "ρ": "ᵨ",
                                "ϕ": "ᵩ",  # symbol phi (0x3d5)
                                "φ": "ᵩ",  # letter phi (0x3c6)
                                "χ": "ᵪ"}
subscript_to_regular_letters = {v: k for k, v in regular_to_subscript_letters.items()}  # letter phi overrides symbol phi in this inverse

regular_to_subscript = {**regular_to_subscript_numbers,
                        **regular_to_subscript_symbols,
                        **regular_to_subscript_letters}
subscript_to_regular = {v: k for k, v in regular_to_subscript.items()}

regular_to_superscript_numbers = {"0": "⁰",
                                  "1": "¹",
                                  "2": "²",
                                  "3": "³",
                                  "4": "⁴",
                                  "5": "⁵",
                                  "6": "⁶",
                                  "7": "⁷",
                                  "8": "⁸",
                                  "9": "⁹"}
superscript_to_regular_numbers = {v: k for k, v in regular_to_superscript_numbers.items()}

regular_to_superscript_symbols = {"+": "⁺",
                                  "-": "⁻",
                                  "=": "⁼",
                                  "(": "⁽",
                                  ")": "⁾"}
superscript_to_regular_symbols = {v: k for k, v in regular_to_superscript_symbols.items()}

regular_to_superscript_letters = {"i": "ⁱ",
                                  "n": "ⁿ"}
superscript_to_regular_letters = {v: k for k, v in regular_to_superscript_letters.items()}

regular_to_superscript = {**regular_to_superscript_numbers,
                          **regular_to_superscript_symbols,
                          **regular_to_superscript_letters}
superscript_to_regular = {v: k for k, v in regular_to_superscript.items()}

# Here "." is important for `raven-arxiv2id` to correctly handle files downloaded by `raven-arxiv-download`.
filename_safe_nonalphanum = " -_',."
