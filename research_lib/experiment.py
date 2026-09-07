"""Experiment registry - reproducibility for every research run.

Every research run records: experiment id, git commit, data version/range, library versions,
parameters, seed. A result must be reproducible from its experiment id.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_lib.constants import PROJECT_ROOT


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for lib in ("pandas", "numpy", "scipy", "sklearn", "xgboost", "hmmlearn", "freqtrade"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[lib] = "missing"
    versions["python"] = sys.version.split()[0]
    return versions


@dataclass
class Experiment:
    """Record for one reproducible research run."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    data_version: str = "research-v1"
    data_range: str = "2022-01-01..2025-06-01"
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str = field(default_factory=_git_commit)
    libs: dict[str, str] = field(default_factory=_library_versions)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            payload = {
                "name": self.name,
                "params": self.params,
                "data_version": self.data_version,
                "data_range": self.data_range,
                "git_commit": self.git_commit,
            }
            self.id = hashlib.sha1(  # noqa: S324 - not security sensitive
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, reports_dir: Path | None = None) -> Path:
        reports_dir = reports_dir or PROJECT_ROOT / "research" / "experiments"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"{self.id}_{self.name.replace(' ', '_')}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return out
