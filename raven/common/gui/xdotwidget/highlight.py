"""Highlight state management with fade animations.

This module manages the visual highlighting of graph elements,
including hover highlights that fade out when the element is
no longer hovered.

The interaction with Raven's GUI animation system, to actually run the
animated parts, is handled by `widget.py`; this is an internal class.
"""

__all__ = ["HighlightState"]

import threading
import time
from typing import Dict, Optional, Set

from ... import numutils

from .graph import Element, Node


class HighlightState:
    """Manages highlight state for graph elements.

    Supports:
    - Instant hover highlighting (when mouse enters an element)
    - Fade-out animations (when mouse leaves an element)
    - Programmatic highlighting (set by API, not affected by hover)
    - Multiple simultaneous fade-out animations

    The highlight intensity for each element is available via get_intensity().
    """

    def __init__(self, fade_duration: float = 2.0):
        """Initialize highlight state.

        `fade_duration`: Duration of fade-out animation in seconds.
        """
        self.fade_duration = fade_duration

        # Currently hovered element (instant on)
        self._hover: Optional[Element] = None

        # Elements fading out: element -> (start_time, initial_intensity)
        self._fading: Dict[Element, tuple] = {}
        self._fading_lock = threading.RLock()

        # Programmatically highlighted elements (not affected by hover)
        # TODO: These are two unrelated highlight sources, only cleared together. Unify the mechanisms; but check first that doing so doesn't break anything in `raven-xdot-viewer` (there might be a reason we sometimes use IDs).
        self._programmatic: Set[Element] = set()  # by element reference
        self._programmatic_node_ids: Set[str] = set()  # by node ID
        self._programmatic_lock = threading.RLock()

    def set_hover(self, element: Optional[Element]) -> None:
        """Set the currently hovered element.

        When a new element is hovered, any previous hover starts fading out.
        """
        if element == self._hover:
            return

        with self._fading_lock:
            # Previous hover starts fading
            if self._hover is not None:
                # Start fade from full intensity
                self._fading[self._hover] = (time.time_ns(), 1.0)

            self._hover = element

            # If the new hover was fading, stop it
            if element is not None and element in self._fading:
                del self._fading[element]

    def get_hover(self) -> Optional[Element]:
        """Return the currently hovered element, or None."""
        return self._hover

    def set_highlighted_nodes(self, node_ids: Set[str]) -> None:
        """Set programmatically highlighted nodes by their IDs.

        These elements will be highlighted regardless of hover state.

        This set is separate from those highlighted by direct element reference, see `set_highlighted`.
        """
        with self._programmatic_lock:
            self._programmatic_node_ids = set(node_ids)

    def get_highlighted_node_ids(self) -> Set[str]:
        """Return the set of programmatically highlighted node IDs."""
        return set(self._programmatic_node_ids)

    def set_highlighted(self, elements: Set[Element]) -> None:
        """Set the programmatically highlighted elements by element reference.

        These elements will be highlighted regardless of hover state.

        This set is separate from those highlighted by IDs, see `set_highlighted_nodes`.
        """
        with self._programmatic_lock:
            self._programmatic = set(elements)

    def get_highlighted(self) -> Set[Element]:
        """Return the set of programmatically highlighted elements."""
        return set(self._programmatic)

    def clear_programmatic(self) -> None:
        """Clear all programmatic highlights, both direct element reference and ID-based."""
        with self._programmatic_lock:
            self._programmatic.clear()
            self._programmatic_node_ids.clear()

    def is_highlighted(self, element: Element, graph=None) -> bool:
        """Check if an element is currently highlighted.

        `element`: The element to check.
        `graph`: Optional Graph, used to check programmatic node IDs.

        Returns True if the element is hovered, fading, or programmatic.
        """
        with self._fading_lock, self._programmatic_lock:
            if element == self._hover:
                return True
            if element in self._fading:
                return True
            if element in self._programmatic:
                return True

            # Check programmatic node IDs
            if isinstance(element, Node) and element.internal_name:
                if element.internal_name in self._programmatic_node_ids:
                    return True

            return False

    def get_intensity(self, element: Element, graph=None) -> float:
        """Get the highlight intensity for an element.

        `element`: The element to check.
        `graph`: Optional Graph, used to check programmatic node IDs.

        Returns a value in [0, 1] where:
        - 1.0 = fully highlighted (hovered or programmatic)
        - 0.0 = not highlighted
        - (0, 1) = fading out
        """
        with self._fading_lock, self._programmatic_lock:
            # Programmatic highlight is always full intensity
            if element in self._programmatic:
                return 1.0

            # Check programmatic node IDs
            if isinstance(element, Node) and element.internal_name:
                if element.internal_name in self._programmatic_node_ids:
                    return 1.0

            # Hover is full intensity
            if element == self._hover:
                return 1.0

            # Check fading
            if element in self._fading:
                start_time, initial_intensity = self._fading[element]
                elapsed = (time.time_ns() - start_time) // 10**9
                progress = elapsed / self.fade_duration

                if progress >= 1.0:
                    return 0.0

                # Exponential decay (smooth transition)
                return initial_intensity * (1.0 - numutils.nonanalytic_smooth_transition(progress))

            return 0.0

    def update(self) -> bool:
        """Advance fade animations.

        Returns True if any animations are still running.
        """
        now = time.time_ns()
        finished = []

        with self._fading_lock:
            for element, (start_time, initial_intensity) in self._fading.items():
                elapsed = (now - start_time) // 10**9
                if elapsed >= self.fade_duration:
                    finished.append(element)

            for element in finished:
                del self._fading[element]

            return len(self._fading) > 0

    def get_all_highlighted(self, graph=None) -> Set[Element]:
        """Return all currently highlighted elements.

        `graph`: Optional Graph, used to resolve programmatic node IDs.

        Returns the union of hover, fading, and programmatic highlights.
        """
        result = set()
        if self._hover is not None:
            result.add(self._hover)

        with self._fading_lock, self._programmatic_lock:
            result.update(self._fading.keys())
            result.update(self._programmatic)

            # Resolve programmatic node IDs
            if graph is not None and self._programmatic_node_ids:
                for node_id in self._programmatic_node_ids:
                    node = graph.get_node_by_name(node_id)
                    if node is not None:
                        result.add(node)

            return result

    def is_animating(self) -> bool:
        """Return True if any fade animations are running."""
        with self._fading_lock:
            return len(self._fading) > 0
