# Based on:
#
# https://github.com/SillyTavern/SillyTavern-WebSearch-Selenium/blob/main/src/index.ts
#
# and its old Python implementation:
#
# https://github.com/SillyTavern/SillyTavern-Extras/blob/main/modules/websearch/script.py

__all__ = ["search_google", "search_duckduckgo"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import pathlib
import time
from typing import List, Optional
import urllib.parse

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

from . import config
from . import utils

# --------------------------------------------------------------------------------
# Bootup

# See `navigator.userAgent` in a web browser's JavaScript console (to access it, try pressing F12 or Ctrl+Shift+C)
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

dump_dir = pathlib.Path(config.config_base_dir).expanduser().resolve() / "websearch"
utils.create_directory(dump_dir)

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
        logger.info("get_driver: Chrome not found, using Firefox instead.")
        logger.info("get_driver: Initializing Firefox driver...")
        firefoxService = FirefoxService()
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.set_preference("intl.accept_languages", "en,en_US")
        options.set_preference("general.useragent.override", user_agent)  # https://stackoverflow.com/a/72465725
        return webdriver.Firefox(service=firefoxService, options=options)

driver = get_driver()
def quit_driver():
    driver.quit()
atexit.register(quit_driver)

# --------------------------------------------------------------------------------
# Utilities

def encodeURIComponent(text: str) -> str:
    # https://stackoverflow.com/questions/6431061/python-encoding-characters-with-urllib-quote
    return urllib.parse.quote(text, safe="!~*'()")

def wait_for_id(element_id: str, delay: float = 5.0) -> None:
    try:
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, element_id)))
    except Exception:
        logger.info(f"wait_for_id: Element with id '{element_id}' not found, proceeding without.")

def wait_for_selector(selector: str, delay: float = 5.0) -> None:
    try:
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
    except Exception:
        logger.info(f"wait_for_id: Element matching selector '{selector}' not found, proceeding without.")

def get_page_height() -> int:
    return driver.execute_script("return document.body.scrollHeight")

def wait_for_page_height_increase(old_height: int) -> int:
    for k in range(5):
        page_height = get_page_height()
        if page_height > old_height:
            return page_height
        time.sleep(1.0)
    return old_height

def click_element(element: WebElement) -> None:
    try:
        WebDriverWait(driver, 1.0).until(EC.element_to_be_clickable(element))
    except exceptions.TimeoutException:
        return
    else:
        element.click()

def find_first_element_by_id(element_id: str) -> Optional[WebElement]:
    it = driver.find_elements(By.ID, element_id)
    try:
        element = next(it)
    except StopIteration:
        return None
    else:
        return element

def get_content_by_selector(selector: str) -> List[str]:
    collected = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            collected.append(f"{el.text}")
    return collected

def get_attr_by_selector(selector: str, attr: str) -> List[str]:
    collected = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            collected.append(el.get_attribute(attr))
    return collected

def debug_dump():
    with open(dump_filename, "w", encoding='utf-8') as html_file:
        html_file.write(driver.page_source)

# --------------------------------------------------------------------------------
# API
#
# We use memoization to cache results for each unique query during the same session.

def format_results(texts: List[str],
                   titles: Optional[List[str]] = None,
                   links: Optional[List[str]] = None) -> str:
    results = [{"text": text} for text in texts]
    if links:
        for result, link in zip(results, links):
            result["link"] = link
    if titles:
        for result, title in zip(results, titles):
            result["title"] = title

    def format_result(result):
        if "title" in result and "link" in result:
            heading = f"{result['title']}\n{result['link']}"
        elif "title" in result:
            heading = result["title"]
        elif "link" in result:
            heading = result["link"]
        else:
            return f"{result['text']}\n"
        return f"{heading}\n\n{result['text']}\n"

    preformatted_text = "-----\n".join(format_result(result) for result in results)
    return preformatted_text, results

@memoize
def search_google(query: str, max_links: int = 10) -> (List[str], List[str]):
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
def search_duckduckgo(query: str, max_links: int = 10) -> (List[str], List[str]):
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
# def search_startpage(query: str, max_links: int = 10) -> (List[str], List[str]):
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

# --------------------------------------------------------------------------------
# Example

def main():
    preformatted_text, results = search_duckduckgo("sharon apple")
    print(preformatted_text)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
