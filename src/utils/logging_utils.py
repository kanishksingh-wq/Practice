import logging
import os
from typing import Optional


def setup_logging(logs_dir: str, level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))

    fh = logging.FileHandler(os.path.join(logs_dir, 'pipeline.log'))
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
