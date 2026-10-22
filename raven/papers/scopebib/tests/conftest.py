"""Fixtures for the scope judge's tests.

`llm_settings` is the Librarian's: the real settings object, built without a backend, with the backend it
stands in for serving the same model throughout. Imported rather than copied, so the two cannot drift.
"""

from raven.librarian.tests.conftest import llm_settings  # noqa: F401 -- a fixture, found by pytest by name
