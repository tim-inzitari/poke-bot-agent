from __future__ import annotations

from pathlib import Path

# =============================================================================
# Output layout — edit OUTPUT_ROOT here if you want a different base directory.
# =============================================================================

OUTPUT_ROOT = "outputs"
CHECKPOINTS_DIR = "checkpoints"
REPORTS_DIR = "reports"
LOGS_DIR = "logs"
ROLLOUTS_DIR = "rollouts"
SUBMISSIONS_DIR = "submissions"

LEGACY_CHECKPOINT_PATH = "out/value_model.pt"


def output_root(root: Path) -> Path:
    return root / OUTPUT_ROOT


def checkpoints_dir(root: Path) -> Path:
    return output_root(root) / CHECKPOINTS_DIR


def reports_dir(root: Path) -> Path:
    return output_root(root) / REPORTS_DIR


def logs_dir(root: Path) -> Path:
    return output_root(root) / LOGS_DIR


def rollouts_dir(root: Path) -> Path:
    return output_root(root) / ROLLOUTS_DIR


def submissions_dir(root: Path) -> Path:
    return output_root(root) / SUBMISSIONS_DIR


def checkpoint_path(root: Path, model_id: str) -> Path:
    return checkpoints_dir(root) / f"{model_id}.pt"


def report_path(root: Path, model_id: str) -> Path:
    return reports_dir(root) / f"{model_id}.json"


def log_path(root: Path, name: str) -> Path:
    stem = name.removesuffix(".log")
    return logs_dir(root) / f"{stem}.log"


def rollout_path(root: Path, name: str) -> Path:
    stem = name.removesuffix(".jsonl")
    return rollouts_dir(root) / f"{stem}.jsonl"


def resolve_checkpoint_path(root: Path, *, model_id: str | None = None, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    if model_id:
        return checkpoint_path(root, model_id)
    legacy = root / LEGACY_CHECKPOINT_PATH
    if legacy.exists():
        return legacy
    return checkpoint_path(root, "temporal_current")


def resolve_checkpoint_for_load(
    root: Path,
    *,
    model_id: str | None = None,
    explicit: str | Path | None = None,
) -> Path:
    """Pick an existing checkpoint for reading, with legacy fallback."""
    candidates: list[Path] = []
    if explicit is not None:
        path = Path(explicit)
        candidates.append(path if path.is_absolute() else root / path)
    if model_id:
        candidates.append(checkpoint_path(root, model_id))
    candidates.append(root / LEGACY_CHECKPOINT_PATH)
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return candidates[0]


def ensure_output_layout(root: Path) -> None:
    for path in (
        checkpoints_dir(root),
        reports_dir(root),
        logs_dir(root),
        rollouts_dir(root),
        submissions_dir(root),
    ):
        path.mkdir(parents=True, exist_ok=True)


def describe_layout(root: Path) -> dict[str, str]:
    return {
        "root": str(output_root(root)),
        "checkpoints": str(checkpoints_dir(root)),
        "reports": str(reports_dir(root)),
        "logs": str(logs_dir(root)),
        "rollouts": str(rollouts_dir(root)),
        "submissions": str(submissions_dir(root)),
    }
