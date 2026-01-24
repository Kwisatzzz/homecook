import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from llm.ingredient_parser import IngredientParser
from llm.unit_unifier import UnitUnifier
from scraping.websites.ania import AniaGotujeIngredientExtractor


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("process_aniagotuje")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    h.setFormatter(fmt)
    h.setLevel(logger.level)
    logger.addHandler(h)
    return logger


def fmt_quantity(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s


def fmt_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    if sec < 3600:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}m {s:.1f}s"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}h {m}m {s:.0f}s"


def aggregate_ingredients(unified_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    acc: Dict[Tuple[str, str], float] = defaultdict(float)

    normalized: Dict[str, Dict[str, Any]] = unified_payload.get("normalized", {})
    for name, info in normalized.items():
        values = info.get("values", [])
        for v in values:
            try:
                unit = v["unit"]
                val = float(v["value"])
            except Exception:
                continue
            if unit is None:
                continue
            acc[(name, unit)] += val

    out: List[Dict[str, str]] = []
    for (name, unit), total in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append({"quantity": fmt_quantity(total), "unit": unit, "name": name})
    return out


class JsonArrayWriter:
    def __init__(self, path: str, ensure_ascii: bool = False) -> None:
        self.path = path
        self.ensure_ascii = ensure_ascii
        self._f = None
        self._first = True

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self._f.write("[\n")
        self._f.flush()
        return self

    def write_obj(self, obj: Dict[str, Any]) -> None:
        if self._f is None:
            raise RuntimeError("Writer not opened")
        if not self._first:
            self._f.write(",\n")
        self._first = False
        self._f.write(json.dumps(obj, ensure_ascii=self.ensure_ascii, indent=2))
        self._f.flush()

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.write("\n]\n")
            self._f.flush()
            self._f.close()
        self._f = None


@dataclass
class ProcessConfig:
    input_urls_path: str = "data/sites/aniagotuje.txt"
    output_path: str = "data/processed_sites/aniagotuje.json"
    log_level: str = "INFO"
    max_urls: int = 0


def process_all(cfg: ProcessConfig) -> None:
    logger = setup_logger(cfg.log_level)

    if not os.path.exists(cfg.input_urls_path):
        raise FileNotFoundError(f"Input file not found: {cfg.input_urls_path}")

    with open(cfg.input_urls_path, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f.read().splitlines() if u.strip()]

    if cfg.max_urls and cfg.max_urls > 0:
        urls = urls[: cfg.max_urls]

    logger.info("Loaded %d URLs from %s", len(urls), cfg.input_urls_path)
    logger.info("Output: %s", cfg.output_path)

    extractor = AniaGotujeIngredientExtractor()
    llm_parser = IngredientParser()
    unifier = UnitUnifier(units_path="data/units.json", logging_level=cfg.log_level)

    ok = 0
    failed = 0

    t_program_start = time.perf_counter()

    with JsonArrayWriter(cfg.output_path) as writer:
        for i, url in enumerate(
            tqdm(urls, desc="Processing recipes", unit="url"), start=1
        ):
            t_url_start = time.perf_counter()
            logger.info("(%d/%d) START %s", i, len(urls), url)

            try:
                scraped = extractor.fetch_ingredients(url)
                raw_text = scraped["ingredients"]

                parsed: Dict[str, List[str]] = llm_parser.parse(raw_text)

                unified = unifier.unify(parsed)

                ingredients_list = aggregate_ingredients(unified)

                recipe_obj = {
                    "url": url,
                    "ingredients": ingredients_list,
                }
                writer.write_obj(recipe_obj)

                t_url = time.perf_counter() - t_url_start
                ok += 1

                logger.info(
                    "DONE url=%s | ingredients=%d | time=%s | units_db_updated=%s",
                    url,
                    len(ingredients_list),
                    fmt_seconds(t_url),
                    unified.get("units_db_updated"),
                )

            except Exception as e:
                t_url = time.perf_counter() - t_url_start
                failed += 1
                logger.exception(
                    "FAILED url=%s | time=%s | error=%s",
                    url,
                    fmt_seconds(t_url),
                    e,
                )

    t_total = time.perf_counter() - t_program_start

    logger.info(
        "FINISHED ok=%d failed=%d total=%d | total_time=%s | avg_per_url=%s",
        ok,
        failed,
        len(urls),
        fmt_seconds(t_total),
        fmt_seconds(t_total / max(1, len(urls))),
    )


if __name__ == "__main__":
    cfg = ProcessConfig(
        input_urls_path="data/sites/aniagotuje.txt",
        output_path="data/processed_sites/aniagotuje.json",
        log_level="INFO",
        max_urls=100,
    )
    process_all(cfg)
