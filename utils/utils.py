import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Union

import yaml


def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("__main__")
    if not logger.hasHandlers():
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=0, encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def split_into_chunks(text: str, tokenizer, max_tokens) -> List[str]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    div = (len(encoded) // max_tokens) + 1
    part_len = (len(encoded) // div) + 1
    chunks = [encoded[i : i + part_len] for i in range(0, len(encoded), part_len)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_config(config_path: Path) -> Dict[str, Union[str, int]]:
    with config_path.open("r") as file:
        return yaml.safe_load(file)
