"""What Raven-server's two web modules, `websearch` and `webfetch`, share.

Both drive a headless browser, which navigates one page at a time, and both do their work for a client that
may stop waiting before it is done.

Cheap to import: `selenium` is imported only when a driver is made, which is what lets `webfetch`'s pure
helpers, and their tests, import without it.
"""

__all__ = ["USER_AGENT",
           "WebToolException", "Cancelled",
           "get_driver",
           "lock_unless_cancelled"]

import logging
logger = logging.getLogger(__name__)

import contextlib
import threading
from collections.abc import Callable, Iterator

from colorama import Fore, Style

# The user agent both modules present, the browser and `webfetch`'s plain GET alike. Some sites serve thin
# content to an agent they do not recognize. See `navigator.userAgent` in a web browser's JavaScript console
# (to access it, try pressing F12 or Ctrl+Shift+C).
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


class WebToolException(Exception):
    """Base class for the ways a web module's job can end without its result."""

class Cancelled(WebToolException):
    """The job stopped because its `is_cancelled` said to, before it had a result."""


def get_driver(page_load_timeout: float | None = None):
    """Create a headless browser driver, Chrome if installed, else Firefox. `None` if neither is.

    `page_load_timeout`: seconds a navigation may take before it raises
                         `selenium.common.exceptions.TimeoutException`. `None` leaves Selenium's own default.
    """
    maybe_driver = _make_driver()
    if maybe_driver is not None and page_load_timeout is not None:
        maybe_driver.set_page_load_timeout(page_load_timeout)
    return maybe_driver

def _is_colab():
    """False. We never run inside colab. Provided for compatibility only."""
    return False

def _make_driver():
    from selenium import webdriver  # noqa: PLC0415 -- deferred; see the module docstring
    from selenium.webdriver.chrome.options import Options as ChromeOptions  # noqa: PLC0415
    from selenium.webdriver.chrome.service import Service as ChromeService  # noqa: PLC0415
    from selenium.webdriver.firefox.options import Options as FirefoxOptions  # noqa: PLC0415
    from selenium.webdriver.firefox.service import Service as FirefoxService  # noqa: PLC0415
    try:
        logger.info("get_driver: Initializing Chrome driver...")
        options = ChromeOptions()
        options.add_argument('--disable-infobars')
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--lang=en-GB")
        options.add_argument(f"--user-agent={USER_AGENT}")

        if _is_colab():
            return webdriver.Chrome('chromedriver', options=options)
        else:
            chromeService = ChromeService()
            return webdriver.Chrome(service=chromeService, options=options)
    except Exception:
        try:
            logger.info("get_driver: Chrome not found, using Firefox instead.")
            logger.info("get_driver: Initializing Firefox driver...")
            firefoxService = FirefoxService()
            options = FirefoxOptions()
            options.add_argument("--headless")
            options.set_preference("intl.accept_languages", "en,en_US")
            options.set_preference("general.useragent.override", USER_AGENT)  # https://stackoverflow.com/a/72465725
            return webdriver.Firefox(service=firefoxService, options=options)
        except Exception:
            print(f"{Fore.RED}{Style.BRIGHT}ERROR{Style.RESET_ALL} (details below)")
            logger.error("get_driver: Firefox not found either; no headless browser available.")
            return None


@contextlib.contextmanager
def lock_unless_cancelled(lock: threading.Lock,
                          is_cancelled: Callable[[], bool],
                          poll_interval: float = 0.1) -> Iterator[None]:
    """Hold `lock` for the body of a `with`, waiting for it only while `is_cancelled()` answers `False`.

    Raises `Cancelled`, without entering the body, if the wait is cancelled. Otherwise the lock is released
    when the body exits, however it exits.

    `poll_interval`: how often, in seconds, the wait asks `is_cancelled`.
    """
    # A job queued behind another would otherwise wait out the whole of the one ahead of it and then run in
    # full for a client that has gone.
    while not lock.acquire(timeout=poll_interval):
        if is_cancelled():
            raise Cancelled
    try:
        yield
    finally:
        lock.release()
