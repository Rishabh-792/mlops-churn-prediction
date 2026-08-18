"""
Fetch the raw training dataset.

The dataset is not committed to the repository. It is downloaded from its
upstream source and verified against a SHA-256 recorded in configs/config.json,
so every run trains on byte-identical input. If the checksum does not match,
the download is rejected rather than silently used.

Usage:
    python -m scripts.download_data
    python -m scripts.download_data --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from utils.core_utils import ensure_dir, get_logger
from utils.settings_manager import SettingsManager

logger = get_logger("download_data", log_dir=None)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(config_path: str = "configs/config.json", force: bool = False) -> Path:
    settings = SettingsManager(config_path).load()
    target = Path(settings.data.raw_path)

    if target.exists() and not force:
        actual = sha256_of(target)
        if actual == settings.data.sha256:
            logger.info("Dataset already present and verified: %s", target)
            return target
        logger.warning("Existing file checksum mismatch; re-downloading.")

    ensure_dir(str(target.parent))
    logger.info("Downloading %s", settings.data.source_url)
    urllib.request.urlretrieve(settings.data.source_url, target)

    actual = sha256_of(target)
    if actual != settings.data.sha256:
        target.unlink(missing_ok=True)
        raise SystemExit(
            f"Checksum mismatch for {settings.data.source_url}\n"
            f"  expected {settings.data.sha256}\n"
            f"  actual   {actual}\n"
            "Refusing to train on unverified data."
        )

    size_kb = target.stat().st_size / 1024
    logger.info("Verified %s (%.0f KB, sha256 %s...)", target, size_kb, actual[:12])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(args.config, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
