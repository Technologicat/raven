"""Fixtures shared by Raven-server's tests."""

import pytest


@pytest.fixture
def fake_browser():
    """The class of a stand-in for a Selenium driver: it navigates, answers scripts, and can be made to fail.

    `answers`: whether the browser responds at all. One that does not fails every navigation and every
               script, as a driver whose tab has crashed does.

    `navigation_fails`: whether navigations fail while the browser still answers, as for an ordinary bad page.
    """
    exceptions = pytest.importorskip("selenium.common.exceptions",
                                     reason="selenium is not in the CI minimal dep subset")

    class FakeBrowser:
        def __init__(self, *, answers: bool = True, navigation_fails: bool = False, page: str = "<html>a page</html>"):
            self.answers = answers
            self.navigation_fails = navigation_fails
            self.page_source = page
            self.visited = []
            self.quit_called = False

        def get(self, url):
            self.visited.append(url)
            if not self.answers:
                raise exceptions.WebDriverException("Message: tab crashed")
            if self.navigation_fails:
                raise exceptions.WebDriverException("Message: net::ERR_NAME_NOT_RESOLVED")

        def execute_script(self, script):
            if not self.answers:
                raise exceptions.WebDriverException("Message: tab crashed")
            return 1

        def quit(self):
            self.quit_called = True

    return FakeBrowser
