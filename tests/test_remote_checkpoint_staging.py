"""Elmo checkpoint staging must be digest-addressed (no basename collisions)."""

from __future__ import annotations

from pathlib import Path

from poke_bot.checkpoint import checkpoint_digest
from poke_bot.remote_jobs import digest_addressed_basename


def test_digest_addressed_basename_embeds_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "iter_00000.pt"
    path.write_bytes(b"weights-a")
    dig_a = checkpoint_digest(path)
    name_a = digest_addressed_basename(path, digest=dig_a)
    assert name_a.startswith("iter_00000.")
    assert name_a.endswith(".pt")
    assert dig_a.split(":", 1)[-1][:16] in name_a

    path.write_bytes(b"weights-b-different")
    dig_b = checkpoint_digest(path)
    name_b = digest_addressed_basename(path, digest=dig_b)
    assert name_a != name_b
    assert dig_b.split(":", 1)[-1][:16] in name_b
    # Same logical trainer filename, distinct remote objects.
    assert name_a.split(".", 1)[0] == name_b.split(".", 1)[0]
