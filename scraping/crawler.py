import argparse
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


def setup_logging(level: str, log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("crawler")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logger.level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.debug("Logger initialized")
    return logger


@dataclass(frozen=True)
class CrawlConfig:
    start_url: str
    max_pages: int
    max_depth: int
    delay: float
    timeout: float
    same_domain_only: bool
    include_query: bool
    user_agent: str
    file_save: Optional[str]
    log_level: str
    log_file: Optional[str]
    flush_every: int
    dedupe_from_file: bool


def default_output_path(start_url: str) -> str:
    parsed = urlparse(start_url)
    name = (parsed.netloc or "site").lower().replace("www.", "")
    return os.path.join("data", "sites", name)


def canonicalize_url(url: str, *, include_query: bool) -> Optional[str]:
    if not url:
        return None

    url, _frag = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query = parsed.query
    if not include_query:
        query = ""
    else:
        q = parse_qsl(query, keep_blank_values=True)
        query = urlencode(sorted(q))

    return urlunparse((scheme, netloc, path, "", query, ""))


def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.netloc.lower() == pb.netloc.lower()


def extract_links(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        out.add(urljoin(base_url, href))

    return out


def fetch(
    logger: logging.Logger, session: requests.Session, url: str, timeout: float
) -> Tuple[Optional[str], Optional[str], Optional[int], float]:
    t0 = time.monotonic()
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        elapsed = time.monotonic() - t0
    except requests.RequestException:
        elapsed = time.monotonic() - t0
        logger.exception("REQUEST ERROR url=%s elapsed=%.3fs", url, elapsed)
        return None, None, None, elapsed

    status = resp.status_code
    ctype = (resp.headers.get("Content-Type") or "").lower()
    logger.debug(
        "RESPONSE status=%s ctype=%s elapsed=%.3fs final_url=%s",
        status,
        ctype,
        elapsed,
        resp.url,
    )

    if status >= 400:
        return None, ctype, status, elapsed

    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        return None, ctype, status, elapsed

    resp.encoding = resp.encoding or "utf-8"
    return resp.text, ctype, status, elapsed


def load_existing_urls(logger: logging.Logger, path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    urls: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    urls.add(u)
        logger.info("DEDUP loaded_existing=%d from=%s", len(urls), path)
    except Exception:
        logger.exception("DEDUP failed reading existing file: %s", path)
    return urls


def append_urls(logger: logging.Logger, path: str, urls: Set[str]) -> int:
    if not urls:
        return 0
    try:
        with open(path, "a", encoding="utf-8") as f:
            for u in sorted(urls):
                f.write(u + "\n")
            f.flush()
        return len(urls)
    except Exception:
        logger.exception("WRITE failed path=%s", path)
        return 0


def crawl(logger: logging.Logger, cfg: CrawlConfig, save_path: str) -> Set[str]:
    start = canonicalize_url(cfg.start_url, include_query=cfg.include_query)
    if not start:
        raise ValueError(f"Nieprawidłowy start_url: {cfg.start_url}")

    visited: Set[str] = set()
    discovered: Set[str] = {start}
    q = deque([(start, 0)])

    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    pages_fetched = 0
    t_start = time.monotonic()

    logger.info("INIT start_url=%s", start)
    logger.info("INIT output=%s", os.path.abspath(save_path))
    logger.info(
        "INIT limits: max_pages=%d max_depth=%d delay=%.2fs timeout=%.2fs flush_every=%d",
        cfg.max_pages,
        cfg.max_depth,
        cfg.delay,
        cfg.timeout,
        cfg.flush_every,
    )
    logger.info(
        "INIT same_domain_only=%s include_query=%s",
        cfg.same_domain_only,
        cfg.include_query,
    )

    saved_urls: Set[str] = set()
    if cfg.dedupe_from_file:
        saved_urls = load_existing_urls(logger, save_path)

    newly_to_write: Set[str] = set()
    if start not in saved_urls:
        newly_to_write.add(start)
        saved_urls.add(start)
        wrote = append_urls(logger, save_path, newly_to_write)
        logger.info("WRITE initial_appended=%d", wrote)

    newly_to_write.clear()

    while q and pages_fetched < cfg.max_pages:
        url, depth = q.popleft()

        if url in visited:
            logger.debug("SKIP already_visited depth=%d url=%s", depth, url)
            continue
        if depth > cfg.max_depth:
            logger.debug("SKIP depth_exceeded depth=%d url=%s", depth, url)
            continue

        visited.add(url)

        logger.info(
            "FETCH #%d depth=%d queue=%d visited=%d discovered=%d url=%s",
            pages_fetched + 1,
            depth,
            len(q),
            len(visited),
            len(discovered),
            url,
        )

        html, ctype, status, elapsed = fetch(logger, session, url, cfg.timeout)
        pages_fetched += 1

        if cfg.delay > 0:
            logger.debug("SLEEP delay=%.2fs", cfg.delay)
            time.sleep(cfg.delay)

        if html is None:
            logger.info(
                "NOHTML status=%s ctype=%s elapsed=%.3fs url=%s",
                status,
                ctype,
                elapsed,
                url,
            )
            if pages_fetched % cfg.flush_every == 0:
                wrote = append_urls(logger, save_path, newly_to_write)
                if wrote:
                    logger.info(
                        "WRITE appended=%d total_saved=%d", wrote, len(saved_urls)
                    )
                newly_to_write.clear()
            continue

        raw_links = extract_links(html, url)
        logger.info("PARSE extracted_links=%d url=%s", len(raw_links), url)

        new_added = 0
        filtered_external = 0
        filtered_bad = 0
        dedup_saved = 0

        for raw in raw_links:
            can = canonicalize_url(raw, include_query=cfg.include_query)
            if not can:
                filtered_bad += 1
                continue

            if cfg.same_domain_only and not same_domain(start, can):
                filtered_external += 1
                continue

            if can not in discovered:
                discovered.add(can)
                new_added += 1
                if can not in visited and depth + 1 <= cfg.max_depth:
                    q.append((can, depth + 1))

            if can in saved_urls:
                dedup_saved += 1
            else:
                saved_urls.add(can)
                newly_to_write.add(can)

        logger.info(
            "STATS page_added=%d filtered_external=%d filtered_bad=%d dedup_saved=%d queue=%d discovered=%d visited=%d pending_write=%d",
            new_added,
            filtered_external,
            filtered_bad,
            dedup_saved,
            len(q),
            len(discovered),
            len(visited),
            len(newly_to_write),
        )

        if pages_fetched % cfg.flush_every == 0:
            wrote = append_urls(logger, save_path, newly_to_write)
            if wrote:
                logger.info("WRITE appended=%d total_saved=%d", wrote, len(saved_urls))
            newly_to_write.clear()

    wrote = append_urls(logger, save_path, newly_to_write)
    if wrote:
        logger.info("WRITE final_appended=%d total_saved=%d", wrote, len(saved_urls))

    total = time.monotonic() - t_start
    logger.info(
        "DONE fetched=%d discovered=%d visited=%d duration=%.2fs",
        pages_fetched,
        len(discovered),
        len(visited),
        total,
    )
    return discovered


def parse_args(argv: Optional[Iterable[str]] = None) -> CrawlConfig:
    p = argparse.ArgumentParser(
        description="Crawler: zbiera sub-URL + pełne logi + zapis inkrementalny (append)."
    )
    p.add_argument(
        "--url",
        default="https://aniagotuje.pl/",
        help='Start URL (domyślnie: "https://aniagotuje.pl/").',
    )
    p.add_argument("--max-pages", type=int, default=20000)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument(
        "--no-same-domain-only",
        action="store_true",
        help="Jeśli ustawisz, crawler NIE będzie ograniczony do tej samej domeny (domyślnie jest).",
    )
    p.add_argument("--include-query", action="store_true")
    p.add_argument("--user-agent", default="SimpleSubURLCrawler/2.1")
    p.add_argument(
        "--file-save",
        default=None,
        help="Ścieżka do pliku wynikowego. Jeśli brak -> data/sites/<nazwa_url>. Append w trakcie działania.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    p.add_argument(
        "--log-file", default=None, help="Opcjonalny plik logów (np. logs/crawler.log)."
    )

    p.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Co ile pobranych stron dopisywać URL-e do pliku (domyślnie 1 = po każdej stronie).",
    )
    p.add_argument(
        "--dedupe-from-file",
        action="store_true",
        help="Jeśli ustawisz, wczyta istniejący plik i nie będzie dopisywał URL-i już obecnych.",
    )

    a = p.parse_args(argv)

    return CrawlConfig(
        start_url=a.url,
        max_pages=max(1, a.max_pages),
        max_depth=max(0, a.max_depth),
        delay=max(0.0, a.delay),
        timeout=max(1.0, a.timeout),
        same_domain_only=not a.no_same_domain_only,
        include_query=bool(a.include_query),
        user_agent=a.user_agent,
        file_save=a.file_save,
        log_level=a.log_level,
        log_file=a.log_file,
        flush_every=max(1, a.flush_every),
        dedupe_from_file=bool(a.dedupe_from_file),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)
    logger = setup_logging(cfg.log_level, cfg.log_file)

    save_path = cfg.file_save if cfg.file_save else default_output_path(cfg.start_url)
    save_path_abs = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path_abs), exist_ok=True)

    logger.info("PROGRAM START pid=%s python=%s", os.getpid(), sys.version.split()[0])
    logger.info("SAVE path=%s", save_path_abs)
    logger.info("CWD=%s", os.getcwd())

    try:
        crawl(logger, cfg, save_path_abs)
    except Exception:
        logger.exception("FATAL ERROR")
        return 2

    logger.info("PROGRAM END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
