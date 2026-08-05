import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from poke_bot.expert_pilot_importance import TARGET_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_expert_pilot_importance.py"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_uses_exact_acting_seat_team_name_and_keeps_missing_unverifiable(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "episodes.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "nested/episode-a.json",
            json.dumps({"info": {"TeamNames": ["Top Pilot", "Other"]}}),
        )
    targets = {
        "schema": TARGET_SCHEMA,
        "train_rows": [
            {"episode_id": "episode-a", "seat": 0},
            {"episode_id": "episode-missing", "seat": 1},
        ],
        "validation_rows": [],
        "source_archives": [
            {"name": archive.name, "sha256": _sha256(archive)}
        ],
    }
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets), encoding="utf-8")
    output = tmp_path / "pilot-map.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "extract",
            "--targets",
            str(target_path),
            "--archive-dir",
            str(archive_dir),
            "--output",
            str(output),
            "--verify-archive-digests",
            "--workers",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"] == [
        {"episode_id": "episode-a", "seat": 0, "team_name": "Top Pilot"}
    ]
    assert payload["unverifiable"] == [
        {"episode_id": "episode-missing", "seat": 1}
    ]
