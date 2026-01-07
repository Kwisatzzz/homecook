"""AniaGotuje-specific parsing utilities."""

from html.parser import HTMLParser
from typing import Optional

from ..scraper import ScrapeResult, Scraper, ScraperError


class _IngredientHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "div":
            return
        for key, value in attrs:
            if key == "id" and value == "recipeIngredients":
                self._capture = True
                break

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._capture:
            self._capture = False

    def handle_data(self, data: str) -> None:
        if self._capture:
            text = data.strip()
            if text:
                self._parts.append(text)

    def extract(self) -> str:
        return " ".join(self._parts)


def parse_ingredients(html: str) -> str:
    parser = _IngredientHTMLParser()
    parser.feed(html)
    return parser.extract()


def fetch_ingredients(scraper: Scraper, url: str) -> str:
    try:
        result = scraper.fetch(url)
    except ScraperError as exc:
        raise ScraperError(f"Failed to fetch ingredients from {url}: {exc}") from exc
    return parse_ingredients(result.content)


__all__ = ["ScrapeResult", "Scraper", "ScraperError", "fetch_ingredients", "parse_ingredients"]
