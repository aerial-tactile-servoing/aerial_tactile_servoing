"""Repository-relative log directories shared by the host and simulator."""

import datetime
import re
from pathlib import Path


def ensure_log_dir(repo_root: Path, relative_log_path: str) -> Path:
    """Create a log directory in the checkout mounted by the Docker launcher."""
    log_path = Path(repo_root) / relative_log_path
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def make_log_paths(repo_root: Path, tag: str, *, default_tag: str):
    """Create a trial directory while keeping GenoM filenames within 64 bytes."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag).strip("_")
    longest_filename = "pom-measurements.log"
    prefix = f"results/logs_{now}_"
    max_tag_len = 64 - len(prefix) - 1 - len(longest_filename)
    if max_tag_len < 1:
        raise RuntimeError("Cannot build TeleKyb-safe log path: prefix is too long.")
    tag = (tag or default_tag)[:max_tag_len]
    relative_log_path = prefix + tag
    log_path = ensure_log_dir(repo_root, relative_log_path)
    return relative_log_path, log_path


def print_log_paths(repo_root: Path, relative_log_path: str):
    """Report the persistent host-side trial directory."""
    print(f"Logs: {Path(repo_root) / relative_log_path}")
