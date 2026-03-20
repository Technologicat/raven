"""Shared test utilities for Raven."""

__all__ = ["approx"]


def approx(a, b, tol=0.01):
    """Check approximate float equality."""
    return abs(a - b) < tol
