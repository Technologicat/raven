"""Web search client for Raven-avatar.

Based on:

    https://github.com/SillyTavern/SillyTavern-WebSearch-Selenium/blob/main/src/index.ts

and its old Python implementation:

    https://github.com/SillyTavern/SillyTavern-Extras/blob/main/modules/websearch/script.py
"""

__all__ = ["init_module", "is_available", "search"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import pathlib
import time
from typing import Dict, List, Optional, Tuple, Union
import urllib.parse

from colorama import Fore, Style

from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.remote.webdriver import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from unpythonic import memoize

from ..common import config

# --------------------------------------------------------------------------------
# Bootup

# See `navigator.userAgent` in a web browser's JavaScript console (to access it, try pressing F12 or Ctrl+Shift+C)
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

dump_dir = pathlib.Path(config.config_base_dir).expanduser().resolve() / "websearch"

def create_directory(path: Union[str, pathlib.Path]) -> None:
    p = pathlib.Path(path).expanduser().resolve()
    pathlib.Path.mkdir(p, parents=True, exist_ok=True)
create_directory(dump_dir)

dump_filename = dump_dir / "debug.html"

def is_colab():
    """False. We never run inside colab. Provided for compatibility only."""
    return False

def get_driver():
    try:
        logger.info("get_driver: Initializing Chrome driver...")
        options = ChromeOptions()
        options.add_argument('--disable-infobars')
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--lang=en-GB")
        options.add_argument(f"--user-agent={user_agent}")

        if is_colab():
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
            options.set_preference("general.useragent.override", user_agent)  # https://stackoverflow.com/a/72465725
            return webdriver.Firefox(service=firefoxService, options=options)
        except Exception:
            print(f"{Fore.RED}{Style.BRIGHT}ERROR{Style.RESET_ALL} (details below)")
            logger.error("get_driver: Firefox not found either. Disabling websearch.")
            return None

driver = None
def init_module():
    """Initialize the websearch module."""
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}websearch{Style.RESET_ALL}...")
    global driver
    driver = get_driver()
    if driver is not None:
        def quit_driver():
            driver.quit()
        atexit.register(quit_driver)

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (driver is not None)

# --------------------------------------------------------------------------------
# Utilities

def encodeURIComponent(text: str) -> str:
    """Rough Python equivalent for JavaScript's `encodeURIComponent`.

    See:
        https://stackoverflow.com/questions/6431061/python-encoding-characters-with-urllib-quote
    """
    return urllib.parse.quote(text, safe="!~*'()")

def wait_for_id(element_id: str, delay: float = 5.0) -> None:
    """Wait until an element with id `element_id` appears in the page being loaded by the web driver.

    Give up after `delay` seconds, and return anyway.
    """
    try:
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, element_id)))
    except Exception:
        logger.info(f"wait_for_id: Element with id '{element_id}' not found, proceeding without.")

def wait_for_selector(selector: str, delay: float = 5.0) -> None:
    """Wait until an element matching the CSS selector `selector` appears in the page being loaded by the web driver.

    Give up after `delay` seconds, and return anyway.
    """
    try:
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
    except Exception:
        logger.info(f"wait_for_id: Element matching selector '{selector}' not found, proceeding without.")

def get_page_height() -> int:
    """Get the current height of the page in the web driver, in pixels."""
    return driver.execute_script("return document.body.scrollHeight")

def wait_for_page_height_increase(old_height: int) -> int:
    """Wait until the page height in the web driver changes to a value different from `old_height`.

    Returns the new height.

    Times out after 5 seconds. Returns whatever the height is at that moment.
    """
    for k in range(5):
        page_height = get_page_height()
        if page_height > old_height:
            return page_height
        time.sleep(1.0)
    return old_height

def click_element(element: WebElement) -> None:
    """Click a web element in the page in the web driver.

    Can be used to dismiss popups and such.
    """
    try:
        WebDriverWait(driver, 1.0).until(EC.element_to_be_clickable(element))
    except exceptions.TimeoutException:
        return
    else:
        element.click()

def find_first_element_by_id(element_id: str) -> Optional[WebElement]:
    """Find the first element with id `element_id` in the page in the web driver.

    Return the element if found, else return `None`.
    """
    it = driver.find_elements(By.ID, element_id)
    try:
        element = next(it)
    except StopIteration:
        return None
    else:
        return element

def get_content_by_selector(selector: str) -> List[str]:
    """Return a list of the `text` of each element matching the CSS selector `selector` in the page in the web driver."""
    collected = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            collected.append(f"{el.text}")
    return collected

def get_attr_by_selector(selector: str, attr: str) -> List[str]:
    """Return a list of the attribute `attr` of each element matching the CSS selector `selector` in the page in the web driver.

    This can be used to extract link destinations (by matching "a" elements, and `attr="href"`).
    """
    collected = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            collected.append(el.get_attribute(attr))
    return collected

def debug_dump():
    """Dump the page from the web driver to an HTML file, for debugging."""
    with open(dump_filename, "w", encoding='utf-8') as html_file:
        html_file.write(driver.page_source)

# --------------------------------------------------------------------------------
# API
#
# We use memoization to cache results for each unique query during the same session.

def format_results(texts: List[str],
                   titles: Optional[List[str]] = None,
                   links: Optional[List[str]] = None) -> str:
    """Format search results.

    Returns the tuple `(preformatted_text, results)`,
    where `preformatted_text` (for one entry) looks like::

        Page title of this result
        https://some.link/thingy/

        Lorem ipsum dolor sit amet...
        Blah blah blah...

    Both `titles` and `links` are optional, but if present,
    must have as many entries as `texts`.

    In the return value, `results` is a list of dicts::

        [{"text": ...,
          "link": ...,
          "title": ...}
        ]

    This is convenient for further manual formatting.
    """
    results = [{"text": text} for text in texts]
    if links:
        for result, link in zip(results, links):
            result["link"] = link
    if titles:
        for result, title in zip(results, titles):
            result["title"] = title

    def format_result(result):
        if "title" in result and "link" in result:
            heading = f"Web result from: {result['link']}\n{result['title']}"
        elif "title" in result:
            heading = result["title"]
        elif "link" in result:
            heading = f"Web result from: {result['link']}"
        else:
            return f"{result['text']}\n"
        return f"{heading}\n\n{result['text']}\n"

    preformatted_text = "-----\n".join(format_result(result) for result in results)
    return preformatted_text, results

@memoize
def search_google(query: str, max_links: int = 10) -> Tuple[str, Dict]:
    logger.info(f"search_google: Searching Google for {query}...")
    driver.get(f"https://google.com/search?hl=en&q={encodeURIComponent(query)}&num={max_links}")
    wait_for_id("res")
    debug_dump()

    # Accept cookies
    if element := find_first_element_by_id("L2AGLb"):
        click_element(element)

    # TODO: add these back?
    # # Answer box
    # text.write("\n".join(get_content_by_selector(selector=".wDYxhc")))
    # # Knowledge panel
    # text.write("\n".join(get_content_by_selector(selector=".hgKElc")))

    # Page snippets
    texts = get_content_by_selector(selector=".r025kc.lVm3ye")
    # texts_old = get_content_by_selector(selector=".yDYNvb.lyLwlc"))  # Old selectors for page snippets (for compatibility)
    links = get_attr_by_selector(selector=".yuRUbf a", attr="href")

    preformatted_text, results = format_results(texts=texts, links=links)
    logger.debug(f"search_google: Found: {preformatted_text}")
    return preformatted_text, results

@memoize
def search_duckduckgo(query: str, max_links: int = 10) -> Tuple[str, Dict]:
    logger.info(f"search_duckduckgo: Searching DuckDuckGo for {encodeURIComponent(query)}...")
    driver.get(f"https://duckduckgo.com/?kl=wt-wt&kp=-2&kav=1&kf=-1&kac=-1&kbh=-1&ko=-1&k1=-1&kv=n&kz=-1&kat=-1&kbg=-1&kbe=0&kpsb=-1&q={query}")
    wait_for_id("web_content_wrapper")
    debug_dump()

    links = get_attr_by_selector(selector='[data-testid="result-title-a"]', attr="href")

    # Scroll down to load more results if needed
    page_height = get_page_height()
    if len(links) < max_links:
        for k in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            page_height = wait_for_page_height_increase(page_height)
            links = get_attr_by_selector(selector='[data-testid="result-title-a"]', attr="href")
            if len(links) >= max_links:
                break
    texts = get_content_by_selector('[data-result="snippet"]')

    preformatted_text, results = format_results(texts=texts, links=links)
    logger.debug(f"search_duckduckgo: Found: {preformatted_text}")
    return preformatted_text, results

# # StartPage. Doesn't work yet. Likely missing some magic parameters from query.
# #
# # As of May 2025:
# #
# # Wikipedia box, if any, comes first:
# #
# # wiki-qi-container css-1oh4tmw: container for Wikipedia box, containing the actual result and the feedback button
# # result css-14ta8x9: container for the Wikipedia result
# # headline css-i42su6: Wikipedia result title; the actual title is hidden in the link css-1yddzfy inside this; even then, need to strip HTML tags (<span class="link-text"></span)ACTUAL TITLE HERE); link-text is used also elsewhere
# # description css-vyv70p: Wikipedia result subtitle
# # short-extract css-13o7eu2: Wikipedia box shortened snippet
# # full-extract css-1hyfx7x: Wikipedia box full snippet
# #
# # Then the web results:
# #
# # w-gl css-oerspo: container for web results
# #
# # result css-o7i03b: container for one web result
# # wgl-display-url css-u4i8t0: web result link
# # wgl-title css-i3irj7: web result title
# # description css-1507v2l: web result snippet
# #
# @memoize
# def search_startpage(query: str, max_links: int = 10) -> Tuple[str, Dict]:
#     logger.info(f"search_startpage: Searching StartPage for {query}...")
#     driver.get(f"https://www.startpage.com/sp/search?q={encodeURIComponent(query)}")
#     wait_for_selector(".w-gl.css-oerspo")
#     debug_dump()
#
#     # Page snippets
#     titles = get_content_by_selector(".wgl-title.css-i3irj7")
#     links = get_attr_by_selector(".wgl-display-url.css-u4i8t0 a")
#     texts = get_content_by_selector(".description.css-1507v2l")
#
#     # TODO: number of links - here we likely need to click on the next page link.
#
#     preformatted_text, results = format_results(texts=texts, titles=titles, links=links)
#     logger.debug(f"search_startpage: Found: {preformatted_text}")
#     return preformatted_text, results

def search(query: str, engine: str = "duckduckgo", max_links: int = 10) -> Tuple[str, Dict]:
    if engine == "duckduckgo":
        return search_duckduckgo(query, max_links)
    elif engine == "google":
        return search_google(query, max_links)
    # elif engine == "startpage":
    #     return search_startpage(query, max_links)
    assert False

# --------------------------------------------------------------------------------
# Example

def main():
    preformatted_text, results = search_duckduckgo("sharon apple")
    print(preformatted_text)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

# Example result:
#
# INFO:__main__:get_driver: Initializing Chrome driver...
# INFO:__main__:search_duckduckgo: Searching DuckDuckGo for sharon%20apple...
# https://macross.fandom.com/wiki/Sharon_Apple
#
# Sharon Apple (シャロン・アップル, Sharon Appuru) is a virtual idol and one of the antagonists of the Macross Plus OVA and its theatrical adaptation, Macross Plus Movie Edition. She also appeared in the short-lived manga adaptation, Macross Plus: TAC Name. She is an artificially created idol singer whose voice and thoughts are provided by Myung Fang Lone. manifests herself ...
# -----
# https://en.wikipedia.org/wiki/List_of_Macross_characters
#
# The new alliance is tested when the crazed Sharon Apple takes control of the Ghost and attacks them. Guld however convinces Isamu to leave and rescue the imprisoned Myung. There he is faced by intense fire from the Macross itself, also controlled by Sharon Apple. Soon after, Isamu succumbs to hypnotizing powers and takes his fighter ...
# -----
# https://en.wikipedia.org/wiki/Macross_Plus
#
# Sharon Apple's songs are performed by a number of different artists - namely Gabriela Robin, Akino Arai, Mai Yamane, Melodie Sexton, Wuyontana and the Raiché Coutev Sisters. The most notable song in the series is Myung's song "Voices", which is performed by Arai and is the only Japanese-language song in the soundtrack. For the English dub of ...
# -----
# http://www.macross2.net/m3/macrossplus/virturoid.htm
#
# Virturoid is a computer-generated virtual reality singer created by the Macross Consortium in 2039. She became a popular idol, but also caused the Sharon Apple Incident in 2040 when she tried to take over the SDF-1 Macross.
# -----
# https://blackgate.com/2020/01/25/her-masters-voice-the-world-of-virtual-idols-part-i/
#
# Sharon Apple (Macross) The first full-on virtual idol to really capture international attention however came ten years later with the arrival of Sharon Apple — from 1995's follow-up entry in the Macross anime series, the four-part video release called Macross Plus. She is very different from Minmay, in numerous ways!
# -----
# https://myanimelist.net/character/7888/Sharon_Apple/
#
# Sharon Apple Sharon Apple is an artificial idol. She exists as a computer which produces a hologram. While her producers say that she has an artificial intelligence that includes emotional programming. (Source: Wikipedia) Voice Actors. Hyoudou, Mako. Japanese. Harte, Melora. English. Federici, Roberta. Italian. Ambrós, Carmen ...
# -----
# https://deculture.fandom.com/wiki/Sharon_Apple_Incident
#
# The events surrounding the installation of an illegal bio-neural chip into the virtual idol Sharon Apple in 2040. Once installed with the bio-neural chip, Sharon Apple began forming an artificial intelligence and subsequently a dangerous self-preservation psychology. took over control of the SDF-1 Macross, the X-9 Ghost fighter and hypnotically enthralled most of the population of ...
# -----
# https://macross.anime.net/wiki/Sharon_Apple
#
# Unlike the rest of the recent wave of idols however, Sharon is a computer-generated virtual reality singer. Despite this (or perhaps because of this), the virturoid's explosive popularity has grown on Earth and the planets colonized by Humans. ... Project Team: Project Debut: Plus Vol. 1 Voice: Mako Hyoudou, Mai Yamane, Gabriela ...
# -----
# https://anidb.net/character/48625
#
# Sharon Apple is an artificial idol and a computer-generated hologram in the anime Macross Plus. Learn about her name, gender, abilities, appearance, seiyuu, and anime relations on AniDB.
# -----
# https://en.namu.wiki/w/%EC%83%A4%EB%A1%A0%20%EC%95%A0%ED%94%8C
#
# シャロン・アップル Sharon Apple A cyber idol from Macross Plus.She is voiced by Mako Hyodo. At the beginning of Plus, Sharon's AI was incomplete, so the concert was conducted with the support of Mün Pan Ron, who masqueraded as her manager. As a result, Mune's deep consciousness was projected as it is, and in the second half, this twisted and caused an accident.
#
# {'text': 'Sharon Apple (シャロン・アップル, Sharon Appuru) is a virtual idol and one of the antagonists of the Macross Plus OVA and its theatrical adaptation, Macross Plus Movie Edition. She also appeared in the short-lived manga adaptation, Macross Plus: TAC Name. She is an artificially created idol singer whose voice and thoughts are provided by Myung Fang Lone. manifests herself ...', 'link': 'https://macross.fandom.com/wiki/Sharon_Apple'}
# {'text': 'The new alliance is tested when the crazed Sharon Apple takes control of the Ghost and attacks them. Guld however convinces Isamu to leave and rescue the imprisoned Myung. There he is faced by intense fire from the Macross itself, also controlled by Sharon Apple. Soon after, Isamu succumbs to hypnotizing powers and takes his fighter ...', 'link': 'https://en.wikipedia.org/wiki/List_of_Macross_characters'}
# {'text': 'Sharon Apple\'s songs are performed by a number of different artists - namely Gabriela Robin, Akino Arai, Mai Yamane, Melodie Sexton, Wuyontana and the Raiché Coutev Sisters. The most notable song in the series is Myung\'s song "Voices", which is performed by Arai and is the only Japanese-language song in the soundtrack. For the English dub of ...', 'link': 'https://en.wikipedia.org/wiki/Macross_Plus'}
# {'text': 'Virturoid is a computer-generated virtual reality singer created by the Macross Consortium in 2039. She became a popular idol, but also caused the Sharon Apple Incident in 2040 when she tried to take over the SDF-1 Macross.', 'link': 'http://www.macross2.net/m3/macrossplus/virturoid.htm'}
# {'text': "Sharon Apple (Macross) The first full-on virtual idol to really capture international attention however came ten years later with the arrival of Sharon Apple — from 1995's follow-up entry in the Macross anime series, the four-part video release called Macross Plus. She is very different from Minmay, in numerous ways!", 'link': 'https://blackgate.com/2020/01/25/her-masters-voice-the-world-of-virtual-idols-part-i/'}
# {'text': 'Sharon Apple Sharon Apple is an artificial idol. She exists as a computer which produces a hologram. While her producers say that she has an artificial intelligence that includes emotional programming. (Source: Wikipedia) Voice Actors. Hyoudou, Mako. Japanese. Harte, Melora. English. Federici, Roberta. Italian. Ambrós, Carmen ...', 'link': 'https://myanimelist.net/character/7888/Sharon_Apple/'}
# {'text': 'The events surrounding the installation of an illegal bio-neural chip into the virtual idol Sharon Apple in 2040. Once installed with the bio-neural chip, Sharon Apple began forming an artificial intelligence and subsequently a dangerous self-preservation psychology. took over control of the SDF-1 Macross, the X-9 Ghost fighter and hypnotically enthralled most of the population of ...', 'link': 'https://deculture.fandom.com/wiki/Sharon_Apple_Incident'}
# {'text': "Unlike the rest of the recent wave of idols however, Sharon is a computer-generated virtual reality singer. Despite this (or perhaps because of this), the virturoid's explosive popularity has grown on Earth and the planets colonized by Humans. ... Project Team: Project Debut: Plus Vol. 1 Voice: Mako Hyoudou, Mai Yamane, Gabriela ...", 'link': 'https://macross.anime.net/wiki/Sharon_Apple'}
# {'text': 'Sharon Apple is an artificial idol and a computer-generated hologram in the anime Macross Plus. Learn about her name, gender, abilities, appearance, seiyuu, and anime relations on AniDB.', 'link': 'https://anidb.net/character/48625'}
# {'text': "シャロン・アップル Sharon Apple A cyber idol from Macross Plus.She is voiced by Mako Hyodo. At the beginning of Plus, Sharon's AI was incomplete, so the concert was conducted with the support of Mün Pan Ron, who masqueraded as her manager. As a result, Mune's deep consciousness was projected as it is, and in the second half, this twisted and caused an accident.", 'link': 'https://en.namu.wiki/w/%EC%83%A4%EB%A1%A0%20%EC%95%A0%ED%94%8C'}
