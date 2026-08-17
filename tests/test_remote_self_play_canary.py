from __future__ import annotations

from scripts.check_remote_self_play import _exact_resident_checkpoint


def _health(digest: str) -> dict[str, object]:
    return {
        "ok": True,
        "controller_healthy": True,
        "leaf_alive": True,
        "leaf_identity_ok": True,
        "checkpoint_digest": digest,
        "checkpoint_version": 15,
        "pinned_digests": [digest],
        "leaves": [
            {"healthy": True, "checkpoint_digest": digest},
            {"healthy": True, "checkpoint_digest": digest},
        ],
    }


def test_exact_resident_checkpoint_accepts_complete_same_digest_health() -> None:
    digest = "sha256:exact"
    assert _exact_resident_checkpoint(_health(digest), digest) is True


def test_exact_resident_checkpoint_fails_closed_on_any_leaf_mismatch() -> None:
    digest = "sha256:exact"
    health = _health(digest)
    health["leaves"][1]["checkpoint_digest"] = "sha256:stale"  # type: ignore[index]

    assert _exact_resident_checkpoint(health, digest) is False


def test_exact_resident_checkpoint_requires_pinned_digest() -> None:
    digest = "sha256:exact"
    health = _health(digest)
    health["pinned_digests"] = []

    assert _exact_resident_checkpoint(health, digest) is False
