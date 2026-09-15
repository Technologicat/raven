"""The recent text to detect an emotion from, when the text arrives one piece at a time."""

__all__ = ["EmotionWindow"]

import collections


class EmotionWindow:
    """The recent text to detect an emotion from, when the text arrives one piece at a time.

    Detection from a single line is unstable, so an update is due only every `interval` lines, and each
    update reads the last `size` lines together. With `size = 4 * interval`, consecutive updates share 75%
    of their text.

    A piece may hold several lines, and blank lines are not counted, so how a model spaces its paragraphs
    does not change how much text a window holds.

    `interval`, `size`: counted in lines.

    `separator`: what to join the lines with.
    """
    # Chosen as 5 and 20 on Raven-librarian's streaming replies from Qwen3, back when blank lines counted too;
    # Qwen3 separates paragraphs with a blank line, so that was about 2.5 and 10 paragraphs. Possibly chosen
    # for generation at about 100 tokens per second, too: an interval in lines is a different interval in time
    # at a different generation speed. These are the nearest equivalent that keeps the 75% overlap, and have
    # not been checked in their own right.
    DEFAULT_INTERVAL = 3
    DEFAULT_SIZE = 4 * DEFAULT_INTERVAL

    def __init__(self,
                 interval: int = DEFAULT_INTERVAL,
                 size: int = DEFAULT_SIZE,
                 separator: str = "\n") -> None:
        self.interval = interval
        self.separator = separator
        self._lines = collections.deque(maxlen=size)
        self._count = 0

    def add(self, piece: str) -> str | None:
        """Add `piece`. Return the text to detect the emotion from if an update is due now, else `None`.

        The first line added is always due, so the emotion starts following as soon as there is any text.
        A piece with no text in it is never due.
        """
        due = False
        for line in piece.split("\n"):
            if not line.strip():
                continue
            self._lines.append(line)
            due = due or (self._count % self.interval == 0)
            self._count += 1
        return self.separator.join(self._lines) if due else None
