"""
Structured logging for Dreams pipeline runs on LUMI.

Produces:
  - manifest.json: One-time run metadata (git hash, config, node info)
  - results_rank{rank}.jsonl: Per-rank delta>0 results
  - positives_rank{rank}.jsonl: Escalated positive hits
  - metrics_rank{rank}.jsonl: Timing and performance metrics
"""

import json
import os
import hashlib
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List


@dataclass
class RunManifest:
    """Run-level metadata, saved once per run."""
    run_id: str
    timestamp: str
    git_commit: str
    config_hash: str
    node_name: str
    job_id: str
    rocm_version: str
    python_version: str
    n_ranks: int
    n_gpus_per_node: int
    config: Dict[str, Any]
    modules: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_rocm_version() -> str:
    """Get ROCm version string."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showversion"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "version" in line.lower():
                    return line.strip()
        # Fallback: check /opt/rocm/.info/version
        version_file = Path("/opt/rocm/.info/version")
        if version_file.exists():
            return version_file.read_text().strip()
    except Exception:
        pass
    return "unknown"


def _config_hash(config: Dict[str, Any]) -> str:
    """Deterministic hash of config dict."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def create_manifest(
    run_id: str,
    config: Dict[str, Any],
    n_ranks: int = 1,
    n_gpus_per_node: int = 8,
) -> RunManifest:
    """Create a RunManifest with auto-detected metadata."""
    import platform
    import sys

    return RunManifest(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=_get_git_commit(),
        config_hash=_config_hash(config),
        node_name=os.environ.get("SLURMD_NODENAME", platform.node()),
        job_id=os.environ.get("SLURM_JOB_ID", "local"),
        rocm_version=_get_rocm_version(),
        python_version=sys.version,
        n_ranks=n_ranks,
        n_gpus_per_node=n_gpus_per_node,
        config=config,
        modules=_get_loaded_modules(),
    )


def _get_loaded_modules() -> List[str]:
    """Get list of loaded Lmod modules."""
    try:
        mods = os.environ.get("LOADEDMODULES", "")
        return [m for m in mods.split(":") if m]
    except Exception:
        return []


class RunLogger:
    """Structured JSONL logger for one MPI rank.

    Writes three files:
      - results_rank{rank}.jsonl   (delta > threshold only)
      - positives_rank{rank}.jsonl (escalated positives)
      - metrics_rank{rank}.jsonl   (timing / perf data)
    """

    def __init__(self, output_dir: Path, rank: int = 0,
                 delta_threshold: float = 0.0):
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.delta_threshold = delta_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._results_path = self.output_dir / f"results_rank{rank}.jsonl"
        self._positives_path = self.output_dir / f"positives_rank{rank}.jsonl"
        self._metrics_path = self.output_dir / f"metrics_rank{rank}.jsonl"

        # Open files (append mode for checkpoint resume)
        self._results_f = open(self._results_path, 'a')
        self._positives_f = open(self._positives_path, 'a')
        self._metrics_f = open(self._metrics_path, 'a')

        self._results_count = 0
        self._positives_count = 0

    def log_result(self, record: Dict[str, Any]):
        """Log a trajectory-level result (only if delta > threshold)."""
        delta = record.get("delta", -1e10)
        if delta > self.delta_threshold:
            record["rank"] = self.rank
            record["timestamp"] = time.time()
            self._results_f.write(json.dumps(record, default=str) + "\n")
            self._results_count += 1

            # Flush periodically
            if self._results_count % 100 == 0:
                self._results_f.flush()

    def log_positive(self, record: Dict[str, Any]):
        """Log an escalated positive result."""
        record["rank"] = self.rank
        record["timestamp"] = time.time()
        self._positives_f.write(json.dumps(record, default=str) + "\n")
        self._positives_f.flush()
        self._positives_count += 1

    def log_metrics(self, record: Dict[str, Any]):
        """Log timing / performance metrics."""
        record["rank"] = self.rank
        record["timestamp"] = time.time()
        self._metrics_f.write(json.dumps(record, default=str) + "\n")
        self._metrics_f.flush()

    def close(self):
        """Flush and close all log files."""
        for f in [self._results_f, self._positives_f, self._metrics_f]:
            f.flush()
            f.close()

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "results_logged": self._results_count,
            "positives_logged": self._positives_count,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
