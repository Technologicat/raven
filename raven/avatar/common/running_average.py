"""A simple running average calculator.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["RunningAverage"]

class RunningAverage:
    """A simple running average, for things like FPS (frames per second) counters."""
    def __init__(self):
        self.count = 100
        self.data = []

    def add_datapoint(self, data: float) -> None:
        self.data.append(data)
        while len(self.data) > self.count:
            del self.data[0]

    def average(self) -> float:
        if len(self.data) == 0:
            return 0.0
        else:
            return sum(self.data) / len(self.data)
