#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ScraperError(Exception):
    """Raised when scraping fails in a recoverable way."""

@dataclass(frozen=True)
class ScrapeResult:
    url: str
    status: int
    content: str


class Scraper:
    def __init__(self, user_agent: Optional[str] = None, timeout_s: int = 30) -> None:
        self._user_agent = user_agent or "Mozilla/5.0 (compatible; HomecookScraper/1.0)"
        self._timeout_s = timeout_s

    def fetch(self, url: str) -> ScrapeResult:
        request = Request(url, headers={"User-Agent": self._user_agent})
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read().decode(charset, errors="ignore")
                status = response.status
        except (HTTPError, URLError) as exc:
            raise ScraperError(f"Failed to fetch {url}: {exc}") from exc
        return ScrapeResult(url=url, status=status, content=payload)


__all__ = ["Scraper", "ScraperError", "ScrapeResult"]
