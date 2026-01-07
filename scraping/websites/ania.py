"""AniaGotuje-specific parsing utilities."""

import argparse
import json
import sys
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract AniaGotuje recipe ingredients.")
    parser.add_argument("url", help="Recipe page URL")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    scraper = Scraper()
    try:
        ingredients = fetch_ingredients(scraper, args.url)
    except ScraperError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    payload = {"url": args.url, "ingredients": ingredients}
    indent = 2 if args.pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))
    return 0


__all__ = [
    "ScrapeResult",
    "Scraper",
    "ScraperError",
    "build_parser",
    "fetch_ingredients",
    "main",
    "parse_ingredients",
]


if __name__ == "__main__":
    raise SystemExit(main())
