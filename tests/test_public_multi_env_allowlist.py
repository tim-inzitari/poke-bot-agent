"""Public LibcgMultiEnv packing must never strand the r175 gate cells.

The historical generic public ``self_play_multi`` path was especially harmful
to ten legacy strong-public packages: they looked like ordinary completed
remote jobs but failed to contribute retained trajectories.  This suite pins
the deliberately narrow replacement policy:

* safe portable public jobs may use a four-game LibcgMultiEnv pack;
* the ten proven-unsafe legacy packages always retain the ordinary ``play``
  transport; and
* a recordless packed result stays an exact missing cell, rather than gaining
  success credit and disappearing from the replacement queue.
"""

from __future__ import annotations

import copy
import importlib.util
import threading
from pathlib import Path
from typing import Any

import pytest

from poke_bot import public_multi_env_safety, remote_jobs
from poke_bot.remote_jobs import (
    RemoteWorkerInfo,
    iter_additive_results,
    iter_scheduled_additive_results,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts/train_pure_rl.py"
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_pure_rl_public_multi_env_allowlist",
    TRAINER_PATH,
)
assert TRAINER_SPEC is not None and TRAINER_SPEC.loader is not None
trainer = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(trainer)


# Exact default-deny r182 transport admission.  This test deliberately
# duplicates the canonical manifest instead of generating fixtures from it:
# broadening the manifest or silently changing one accepted package byte must
# fail a regression run.  The remaining public packages are singleton-only.
SAFE_PUBLIC_PAIRS = {
    "aman-crustle-fighting": (
        "sha256:96bc2bf6e6c957b2d2d6af383787d14f6850ad299d8cb92f19ce157f0f98900f"
    ),
    "archaludon-ex": (
        "sha256:b5421d03b3c0f7860f7ad13da8b92f73e7eb689f4508e89208474385bd9a8cbe"
    ),
    "biohack-day2-new": (
        "sha256:c0664535a35439d55a30d3142fc7488affc8344437348a225bfe64f9c003ba53"
    ),
    "biohack-meta-20260707": (
        "sha256:83e484fc77e74833d17b07485a3c25eb164eef1c7d9cbe00365e14e1edb67664"
    ),
    "cynthia-garchomp-ex": (
        "sha256:1a972e37fc981efdbdc770d7c7427df15ad8245d8b4233852beda9a500f70966"
    ),
    "dedquoc-rule-based": (
        "sha256:1ffc33069d0e09836412f9464a741387291f7da94b304d6fa39f83a04957deba"
    ),
    "generic-heuristic": (
        "sha256:6f89fa7b6c9734731809fc9782374e94e1d0eb75cde989b0c37ac5b4b4f15e65"
    ),
    "heuristic-baseline": (
        "sha256:52b699ebf51beaf29767748d18f037c3227a38abb321dbe8c8a6b2dbfaa24de2"
    ),
    "kokinn-lucario-search-915": (
        "sha256:c387deb1b8f9a7beb584b07352c97827cc02b6cc8263b5da7e89843a1b284dbd"
    ),
    "lucifer19-battlecore": (
        "sha256:31a1d7453a80379b41797902fb973d4a41c0538decc0005aef444bbca8b3a336"
    ),
    "makthanithin-1084-5": (
        "sha256:b10a7c2c3cb8662a1fa1f3f13867cd0b842993c7ae9217a025cad606faa7e98f"
    ),
    "penguin-public-scores-915": (
        "sha256:04e495a8b21c5fc8f40fc0f2f8853e89e60e777a92cd63444a3da8e76d66aee9"
    ),
    "pilkwang-meta-20260708": (
        "sha256:7120bc67415e06c1cf69d64574f1a41545fd4c2fd084a029d77c5e43a357957f"
    ),
    "plamen06-pokemon-steel": (
        "sha256:c841a5870f00dcfafee88705cd7868a744d0f26e178ed2d96f135f960058767e"
    ),
    "rauff-ptcg-advanced": (
        "sha256:55c0429e788164a6fed2e22cb65dcecf96400f47c764ea82aade77d9a373de72"
    ),
    "roman-baseline-v10": (
        "sha256:83e484fc77e74833d17b07485a3c25eb164eef1c7d9cbe00365e14e1edb67664"
    ),
    "roman-v7-crustle-lucario": (
        "sha256:a7900a94f8b5a2e822454438e852d795fcdc32a3b46ad4f4cb4ff4212445669d"
    ),
    "rv1922-ai-battle": (
        "sha256:aabaf3cc1af680b51435a739a812acea5aaed711a4786b52bd52ba74e72be048"
    ),
    "ryota-alakazam-best5": (
        "sha256:344062386f72b0c66254daf7b7205e1dc937a070e8e62aba49fa55c03a04677f"
    ),
    "specialist-alakazam-owner-accepted-iter39-roster18-v5": (
        "sha256:b46c00c9cea2a1ecc1761550b07fdd2485130b100c5bf841179b45311757e79b"
    ),
    "specialist-crustle-final-format-h10-7efd8d4113e7": (
        "sha256:359e3b4fed00502e58be4631576501b6f63523226ec92f2d75446df085b19afa"
    ),
    "specialist-hops-trevenant-gate-iter10-462f201f8de6-roster18-v5": (
        "sha256:d6be51c9a463527a83e65a27bba5613872a796b1711ada23c9aefab4117d0639"
    ),
    "specialist-lucario-gate-iter10-ffa401242b2c-roster18-v5": (
        "sha256:c60e5b4455b869d8e6b438eea7d97fa54f4aba6f6cdbef208de300060a7f1c78"
    ),
    "specialist-marnie-final-format-h10-f20efb20f5c3": (
        "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
    ),
    "specialist-starmie-gate-iter10-51ed1cc6ffe6-roster18-v5": (
        "sha256:c5bd8c305f79be64626e065f4ae4dd7a2291e029a66258dce176ad74d9206c93"
    ),
    "yaminh-ai-challenge": (
        "sha256:9013cbd06dbc730961544a1bf49202f7b1c80b940d3de7940e29ad554c9bbefb"
    ),
    "yaroslav-lucario-v2-crustle": (
        "sha256:2738a2e4394155b0122eeaa68cec9bbe0cc7dbb4b79f5d055827778444b68bb3"
    ),
}

SAFE_STRONG_PUBLIC_IDS = frozenset(
    {
        "archaludon-ex",
        "lucifer19-battlecore",
        "pilkwang-meta-20260708",
        "specialist-alakazam-owner-accepted-iter39-roster18-v5",
        "specialist-hops-trevenant-gate-iter10-462f201f8de6-roster18-v5",
        "specialist-lucario-gate-iter10-ffa401242b2c-roster18-v5",
        "specialist-starmie-gate-iter10-51ed1cc6ffe6-roster18-v5",
    }
)
SAFE_PUBLIC_TRAINING_GROUPS = {
    opponent_id: (
        "strong_public_practice"
        if opponent_id in SAFE_STRONG_PUBLIC_IDS
        else "diverse_public"
    )
    for opponent_id in SAFE_PUBLIC_PAIRS
}

# The exact ten packages that were missing in attempt_0038 and lost even more
# often when all public rows were routed through run_play_multi in attempt_0040.
LEGACY_PUBLIC_MULTI_ENV_PAIRS = {
    "specialist-archaludon-ex-gate-iter5-251298117902": (
        "sha256:32f00f128d988fdaee2af6d7b1666898f9f1ebe7b38e6a9bf5e26a0a78e7a3e7"
    ),
    "specialist-dragapult-dusknoir-gate-iter15-b6996ed641b1": (
        "sha256:af12432fbb7c96c24d06bf36e4d21d98481793e69aad279a2c7256da42687461"
    ),
    "specialist-dudunsparce-gate-iter15-a1e944fcb4c4": (
        "sha256:2bbfe7218875160036d278312087ac89e4e19ec006363ee7477c23a57cb784e4"
    ),
    "specialist-garchomp-gate-iter5-61fbb254944f": (
        "sha256:66937a20893561c6decaed6a280f40f616bf7601faabdd342cac809c0d7e8968"
    ),
    "specialist-hammer-pult-gate-iter15-c256a0ababee": (
        "sha256:34868aa2232e80665b6a6a6eb0c6488b7390b1b3a01cec3c4e2be9dc9a305ee9"
    ),
    "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98": (
        "sha256:ae9f3c31e2705a955aa1c51b79fbeffcab0d93dfe65da23572b18b9a52d8e8f6"
    ),
    "specialist-rockets-mewtwo-gate-iter5-fc2f9a525a86": (
        "sha256:c16aacc578af56c676d94de3bed8744f73ba4187300c5e3b868a63562a72cf54"
    ),
    "specialist-teal-mask-ogerpon-ex-gate-iter14-5c74cfb63626": (
        "sha256:d5b5432ec25aafc68f74e96c2f4a6538b66f6fcc41698e3dd70269ce7d8f6dee"
    ),
    "specialist-team-rockets-spidops-gate-iter5-4ab63dc94d5a": (
        "sha256:05a06b3640eec555e1f8c37aa06e5556a7d2a1972f34a9b65c575539b0d7073a"
    ),
    "specialist-thwackey-gate-iter5-0435f335fde6": (
        "sha256:fda5d1cff88a6b4eb2d5ff8043b930ec1077751a3134bd3320515f3cb1dcdddf"
    ),
}
LEGACY_PUBLIC_MULTI_ENV_IDS = frozenset(LEGACY_PUBLIC_MULTI_ENV_PAIRS)
SAFE_PUBLIC_IDS = tuple(SAFE_PUBLIC_PAIRS)


def _public_job(
    job_index: int,
    opponent_id: str,
    *,
    content_digest: str | None = None,
    spec_id: str | None = None,
    provenance_opponent_id: str | None = None,
    provenance_digest: str | None = None,
    training_group: str | None = None,
    portable: bool = True,
    contract_schema: str = "poke_bot.portable_baseline_spec/v1",
) -> dict[str, Any]:
    """Return the collection-relevant portion of a portable public job."""

    resolved_spec_id = str(spec_id or opponent_id)
    digest = str(
        content_digest
        or SAFE_PUBLIC_PAIRS.get(opponent_id)
        or LEGACY_PUBLIC_MULTI_ENV_PAIRS.get(opponent_id)
        or ""
    )
    resolved_training_group = str(
        training_group
        or SAFE_PUBLIC_TRAINING_GROUPS.get(opponent_id)
        or "diverse_public"
    )
    return {
        "job_index": int(job_index),
        "opponent_id": str(opponent_id),
        "require_portable_baseline_contract": bool(portable),
        "our_seat": int(job_index % 2),
        "seed": 50_000 + int(job_index),
        "checkpoint": "/tmp/r175-candidate.pt",
        "checkpoint_digest": "sha256:" + "a" * 64,
        "game_timeout_s": 1,
        "spec": {
            "id": resolved_spec_id,
            "name": resolved_spec_id,
            "dir_name": resolved_spec_id,
            "group": "roster",
            "source": "test/public-multi-env",
            "path": f"/tmp/baselines/roster/{resolved_spec_id}",
            "contract_schema": contract_schema,
            "content_digest": digest,
        },
        "target_provenance": {
            "opponent_id": str(provenance_opponent_id or opponent_id),
            "opponent_content_digest": str(provenance_digest or digest),
            "opponent_training_group": resolved_training_group,
        },
    }


class _Pool:
    def imap_unordered(self, fn, jobs):
        yield from (fn(job) for job in jobs)

    def apply(self, fn, job):
        return fn(job)


class _Decision:
    local_share = 0.0
    remote_share = 1.0
    # Keep the initial packet large enough that this test exercises a mixed
    # remote-owned queue, rather than accidentally passing due to input chunks.
    remote_chunk = 32
    remote_demand = {"public-pack.test:8765": 1}


class _Scheduler:
    min_local_frac = 0.0
    prefer_local_frac = 0.0
    min_remote_frac = 1.0
    max_remote_frac = 1.0

    def decision(self):
        return _Decision()

    def bind_remote_endpoints(self, _clients) -> None:
        return None

    def maybe_tick(self, **_kwargs):
        return None

    def note_completed(self, **_kwargs) -> None:
        return None

    def remote_demand(self):
        return dict(_Decision.remote_demand)


class _PublicPackRemote:
    host = "public-pack.test"
    port = 8765
    endpoint = "public-pack.test:8765"

    def __init__(
        self,
        *,
        record_json: object = None,
        public_multi_env_capable: bool = True,
    ) -> None:
        self.record_json = record_json
        self.single_calls: list[tuple[str, dict[str, Any]]] = []
        self.multi_calls: list[list[dict[str, Any]]] = []
        self.info = RemoteWorkerInfo(
            endpoint=self.endpoint,
            workers=1,
            leaf_servers=0,
            gpu_name="",
            device="cpu",
            checkpoint_digest=None,
            hostname=self.host,
            max_workers=1,
            default_workers=1,
            job_kinds=("play", "self_play", "self_play_multi"),
            capabilities=(
                ("public_multi_env_allowlist_r182_v1",)
                if public_multi_env_capable
                else ()
            ),
        )

    def submit_job(self, job: dict[str, Any], *, kind: str = "play") -> dict[str, Any]:
        self.single_calls.append((kind, dict(job)))
        return self._result(job)

    def submit_self_play_multi(
        self, jobs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        snapshot = [dict(job) for job in jobs]
        self.multi_calls.append(snapshot)
        # A real RemoteJobClient restores request order after strict identity
        # validation.  Returning reverse order here ensures the collector is
        # still keyed by each explicit job index rather than packet position.
        return [self._result(job) for job in reversed(snapshot)]

    def _result(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_index": int(job["job_index"]),
            "opponent_id": str(job["opponent_id"]),
            "our_seat": int(job["our_seat"]),
            "seed": int(job["seed"]),
            "checkpoint_digest": str(job["checkpoint_digest"]),
            "winner": int(job["our_seat"]),
            "record_json": self.record_json,
            "self_play": False,
            "multi_env": True,
        }

    def reconnect(self):
        return self.info

    def close(self) -> None:
        return None


class _Writer:
    def __init__(self) -> None:
        self.games: list[object] = []

    @property
    def n_decisions(self) -> int:
        return sum(len(game.decisions) for game in self.games)

    def write_game(self, game) -> None:
        self.games.append(game)


def _stats() -> dict[str, Any]:
    return {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
    }


def _scheduled_remote_rows(
    monkeypatch: pytest.MonkeyPatch,
    remote: _PublicPackRemote,
    jobs: list[dict[str, Any]],
    *,
    executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")
    monkeypatch.setenv("POKEBOT_REMOTE_DEMAND_QUEUE", "1")
    # Test only the packet split.  Do not inherit a real service's public
    # reserve setting and attempt clone sockets to this fake hostname.
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH_MAX", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "1")
    return list(
        iter_scheduled_additive_results(
            local_pool=_Pool(),
            local_fn=lambda job: {**job, "source": "local"},
            jobs=jobs,
            remote_clients=[remote],  # type: ignore[list-item]
            kind="play",
            scheduler=_Scheduler(),
            local_workers=1,
            remote_workers=1,
            on_execution=(executions.append if executions is not None else None),
        )
    )


def _additive_remote_rows(
    monkeypatch: pytest.MonkeyPatch,
    remote: _PublicPackRemote,
    jobs: list[dict[str, Any]],
    *,
    executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the legacy/additive dispatcher against an isolated fake remote."""

    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")
    # Keep the fake endpoint at one socket: this test is about the legacy
    # dispatcher's packet partition, not the separately covered queue reserve.
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "1")
    return list(
        iter_additive_results(
            local_pool=_Pool(),
            local_fn=lambda job: {**job, "source": "local"},
            jobs=jobs,
            remote_clients=[remote],  # type: ignore[list-item]
            kind="play",
            local_workers=0,
            remote_workers=1,
            on_execution=(executions.append if executions is not None else None),
        )
    )


def test_public_multi_env_allowlist_is_exact_and_rejects_identity_smuggling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact r182 id+digest+group identities may enter a pack."""

    monkeypatch.setenv(
        "POKEBOT_PUBLIC_MULTI_ENV_SAFETY_MANIFEST", "/tmp/forged-r182.json"
    )
    assert public_multi_env_safety.public_multi_env_safety_manifest_path().name == (
        "alakazam-public-multi-env-split-r182.json"
    )
    pairs, groups, legacy, schema = public_multi_env_safety._manifest()
    assert pairs == SAFE_PUBLIC_PAIRS
    assert groups == SAFE_PUBLIC_TRAINING_GROUPS
    assert schema == "poke_bot.portable_baseline_spec/v1"
    assert legacy == LEGACY_PUBLIC_MULTI_ENV_IDS
    assert remote_jobs.PUBLIC_MULTI_ENV_LEGACY_OPPONENT_IDS == legacy

    for index, (opponent_id, digest) in enumerate(SAFE_PUBLIC_PAIRS.items()):
        assert remote_jobs.public_multi_env_safe_job(
            _public_job(
                index,
                opponent_id,
                content_digest=digest,
                training_group=SAFE_PUBLIC_TRAINING_GROUPS[opponent_id],
            )
        )

    valid = _public_job(
        900,
        "archaludon-ex",
        training_group=SAFE_PUBLIC_TRAINING_GROUPS["archaludon-ex"],
    )
    malformed_cases: list[dict[str, Any]] = []
    missing_portable = copy.deepcopy(valid)
    missing_portable["require_portable_baseline_contract"] = False
    malformed_cases.append(missing_portable)
    wrong_spec_id = copy.deepcopy(valid)
    wrong_spec_id["spec"]["id"] = "lucifer19-battlecore"
    malformed_cases.append(wrong_spec_id)
    wrong_provenance_id = copy.deepcopy(valid)
    wrong_provenance_id["target_provenance"]["opponent_id"] = (
        "lucifer19-battlecore"
    )
    malformed_cases.append(wrong_provenance_id)
    wrong_spec_digest = copy.deepcopy(valid)
    wrong_spec_digest["spec"]["content_digest"] = SAFE_PUBLIC_PAIRS[
        "lucifer19-battlecore"
    ]
    malformed_cases.append(wrong_spec_digest)
    wrong_provenance_digest = copy.deepcopy(valid)
    wrong_provenance_digest["target_provenance"]["opponent_content_digest"] = (
        SAFE_PUBLIC_PAIRS["lucifer19-battlecore"]
    )
    malformed_cases.append(wrong_provenance_digest)
    wrong_group = copy.deepcopy(valid)
    wrong_group["target_provenance"]["opponent_training_group"] = "formal_eval"
    malformed_cases.append(wrong_group)
    wrong_schema = copy.deepcopy(valid)
    wrong_schema["spec"]["contract_schema"] = "other-schema"
    malformed_cases.append(wrong_schema)
    missing_spec = copy.deepcopy(valid)
    missing_spec.pop("spec")
    malformed_cases.append(missing_spec)

    assert not any(
        remote_jobs.public_multi_env_safe_job(candidate)
        for candidate in malformed_cases
    )
    for index, opponent_id in enumerate(SAFE_PUBLIC_IDS, start=950):
        other_group = (
            "strong_public_practice"
            if SAFE_PUBLIC_TRAINING_GROUPS[opponent_id] == "diverse_public"
            else "diverse_public"
        )
        assert not remote_jobs.public_multi_env_safe_job(
            _public_job(index, opponent_id, training_group=other_group)
        )
    # All ten have their genuine package identities in this fixture.  Their
    # exclusion therefore proves default-deny/legacy routing rather than a
    # weak malformed-payload fallback.
    for index, (opponent_id, digest) in enumerate(
        LEGACY_PUBLIC_MULTI_ENV_PAIRS.items(), start=1_000
    ):
        assert not remote_jobs.public_multi_env_safe_job(
            _public_job(index, opponent_id, content_digest=digest)
        )


def test_scheduled_public_queue_packs_safe_rows_and_keeps_legacy_rows_single(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mixed remote-owned queue must split—not make an unsafe mixed pack."""

    safe_jobs = [
        _public_job(
            index,
            opponent_id,
            training_group=SAFE_PUBLIC_TRAINING_GROUPS[opponent_id],
        )
        for index, opponent_id in enumerate(SAFE_PUBLIC_IDS[:8])
    ]
    legacy_jobs = [
        _public_job(100 + index, opponent_id, content_digest=digest)
        for index, (opponent_id, digest) in enumerate(
            LEGACY_PUBLIC_MULTI_ENV_PAIRS.items()
        )
    ]
    # Safe cells bracket the exact ten legacy cells.  The dispatch must retain
    # the two four-game safe packets while all legacy cells stay on `play`.
    jobs = [*safe_jobs[:4], *legacy_jobs, *safe_jobs[4:]]
    remote = _PublicPackRemote()
    executions: list[dict[str, Any]] = []

    monkeypatch.setenv("POKEBOT_REMOTE_ENDPOINT_CHUNKS", "public-pack.test:8765=64")
    rows = _scheduled_remote_rows(monkeypatch, remote, jobs, executions=executions)

    assert sorted(int(row["job_index"]) for row in rows) == sorted(
        int(job["job_index"]) for job in jobs
    )
    assert len(rows) == len(jobs)
    assert [len(packet) for packet in remote.multi_calls] == [4, 4]
    packed_indices = {
        int(job["job_index"])
        for packet in remote.multi_calls
        for job in packet
    }
    assert packed_indices == {int(job["job_index"]) for job in safe_jobs}
    assert all(
        remote_jobs.public_multi_env_safe_job(job)
        for packet in remote.multi_calls
        for job in packet
    )
    assert all(
        not (
            {str(job["opponent_id"]) for job in packet}
            & LEGACY_PUBLIC_MULTI_ENV_IDS
        )
        for packet in remote.multi_calls
    )

    assert [kind for kind, _job in remote.single_calls] == ["play"] * len(
        legacy_jobs
    )
    assert {
        str(job["opponent_id"]) for _kind, job in remote.single_calls
    } == LEGACY_PUBLIC_MULTI_ENV_IDS
    assert {
        int(job["job_index"]) for _kind, job in remote.single_calls
    } == {int(job["job_index"]) for job in legacy_jobs}

    # A transport packet is not a new logical collection kind.  This protects
    # strict remote proof/accounting and gives each child exactly one credit.
    assert len(executions) == len(jobs)
    assert {event["kind"] for event in executions} == {"play"}
    assert sum(int(event["pack_games"]) for event in executions) == len(jobs)
    assert (
        sum(
            event["transport_kind"] == "self_play_multi"
            for event in executions
        )
        == 8
    )
    assert sum(event["transport_kind"] == "play" for event in executions) == 10

    # One bounded completion receipt makes the production log independently
    # auditable: it distinguishes true four-way LibcgMultiEnv work from the
    # ten intentional one-game legacy transports without counting each packed
    # request as a single retained logical cell.
    output = capsys.readouterr().out
    assert (
        "public_multi_env_plan logical_games=18 safe_allowlisted=8 "
        "legacy_id_singleton=10 default_singleton=0"
    ) in output
    assert "endpoint_pack_cap={'public-pack.test:8765': 4}" in output
    assert "public_multi_env_transport complete " in output
    assert (
        "public-pack.test:8765:safe_multi=8/2req,"
        "safe_singleton=0/0req,legacy_id_singleton=10/10req,"
        "default_singleton=0/0req"
    ) in output


def test_additive_public_queue_packs_safe_rows_and_keeps_legacy_rows_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-scheduled dispatcher must apply the same r182 partition."""

    safe_jobs = [
        _public_job(
            index,
            opponent_id,
            training_group=SAFE_PUBLIC_TRAINING_GROUPS[opponent_id],
        )
        for index, opponent_id in enumerate(SAFE_PUBLIC_IDS[:8])
    ]
    legacy_jobs = [
        _public_job(100 + index, opponent_id, content_digest=digest)
        for index, (opponent_id, digest) in enumerate(
            LEGACY_PUBLIC_MULTI_ENV_PAIRS.items()
        )
    ]
    jobs = [*safe_jobs[:4], *legacy_jobs, *safe_jobs[4:]]
    remote = _PublicPackRemote()
    executions: list[dict[str, Any]] = []

    rows = _additive_remote_rows(monkeypatch, remote, jobs, executions=executions)

    assert sorted(int(row["job_index"]) for row in rows) == sorted(
        int(job["job_index"]) for job in jobs
    )
    assert [len(packet) for packet in remote.multi_calls] == [4, 4]
    assert {
        int(job["job_index"])
        for packet in remote.multi_calls
        for job in packet
    } == {int(job["job_index"]) for job in safe_jobs}
    assert all(
        remote_jobs.public_multi_env_safe_job(job)
        for packet in remote.multi_calls
        for job in packet
    )
    assert [kind for kind, _job in remote.single_calls] == ["play"] * 10
    assert {
        str(job["opponent_id"]) for _kind, job in remote.single_calls
    } == LEGACY_PUBLIC_MULTI_ENV_IDS
    assert len(executions) == len(jobs)
    assert {event["kind"] for event in executions} == {"play"}
    assert sum(int(event["pack_games"]) for event in executions) == len(jobs)
    assert sum(
        event["transport_kind"] == "self_play_multi" for event in executions
    ) == 8
    assert sum(event["transport_kind"] == "play" for event in executions) == 10


def test_scheduled_public_prefetch_does_not_hide_lookahead_in_private_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All public sockets start one pack before an earlier pack can return.

    The r182 controller may scan/reorder queued logical jobs to find four safe
    children through a legacy interleave, but one emitter must not *own* more
    than that one packet.  Otherwise a public prefetch wave opens eight
    sockets for four workers while only the first one or two can reach the
    remote.
    """

    host = "public-prefetch.test"
    endpoint = f"{host}:8765"
    release = threading.Event()
    all_packets_started = threading.Event()
    clients_lock = threading.Lock()
    clients: list[Any] = []
    entered_by_client: set[str] = set()
    submitted_packets: list[tuple[str, tuple[int, ...]]] = []

    class _QueueRemote:
        port = 8765

        def __init__(self, label: str) -> None:
            self.label = label
            self.host = host
            self.endpoint = endpoint
            self.info = RemoteWorkerInfo(
                endpoint=self.endpoint,
                workers=4,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=4,
                default_workers=4,
                job_kinds=("play", "self_play", "self_play_multi"),
                capabilities=("public_multi_env_allowlist_r182_v1",),
            )

        @staticmethod
        def _result(job: dict[str, Any]) -> dict[str, Any]:
            return {
                "job_index": int(job["job_index"]),
                "opponent_id": str(job["opponent_id"]),
                "our_seat": int(job["our_seat"]),
                "seed": int(job["seed"]),
                "checkpoint_digest": str(job["checkpoint_digest"]),
                "winner": int(job["our_seat"]),
                "record_json": None,
            }

        def _wait_for_release(self, jobs: list[dict[str, Any]]) -> None:
            with clients_lock:
                entered_by_client.add(self.label)
                submitted_packets.append(
                    (self.label, tuple(int(job["job_index"]) for job in jobs))
                )
                if len(entered_by_client) == 8:
                    all_packets_started.set()
            assert release.wait(timeout=5.0), "test did not release remote pack"

        def submit_self_play_multi(
            self, jobs: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            self._wait_for_release(jobs)
            return [self._result(job) for job in jobs]

        def submit_job(self, job: dict[str, Any], *, kind: str = "play") -> dict[str, Any]:
            assert kind == "play"
            self._wait_for_release([job])
            return self._result(job)

        def reconnect(self):
            return self.info

        def close(self) -> None:
            return None

    def _new_client(label: str) -> _QueueRemote:
        client = _QueueRemote(label)
        with clients_lock:
            clients.append(client)
        return client

    template = _new_client("template")
    clone_count = 0

    def _clone(_template: object) -> _QueueRemote:
        nonlocal clone_count
        with clients_lock:
            clone_count += 1
            label = f"clone-{clone_count}"
        return _new_client(label)

    class _QueueDecision:
        local_share = 0.0
        remote_share = 1.0
        remote_chunk = 64
        remote_demand = {endpoint: 4}

    class _QueueScheduler:
        min_local_frac = 0.0
        prefer_local_frac = 0.0
        min_remote_frac = 1.0
        max_remote_frac = 1.0

        def decision(self):
            return _QueueDecision()

        def bind_remote_endpoints(self, _clients) -> None:
            return None

        def maybe_tick(self, **_kwargs):
            return None

        def note_completed(self, **_kwargs) -> None:
            return None

    jobs = [
        _public_job(
            index,
            SAFE_PUBLIC_IDS[index % len(SAFE_PUBLIC_IDS)],
            training_group=SAFE_PUBLIC_TRAINING_GROUPS[
                SAFE_PUBLIC_IDS[index % len(SAFE_PUBLIC_IDS)]
            ],
        )
        for index in range(32)
    ]
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_DEMAND_QUEUE", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH", "2")
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH_MAX", "2")
    monkeypatch.setenv("POKEBOT_REMOTE_ENDPOINT_CHUNKS", f"{endpoint}=32")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "30")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "1")
    monkeypatch.setattr(remote_jobs, "_clone_remote_client", _clone)

    rows: list[dict[str, Any]] = []
    failures: list[BaseException] = []

    def _run_dispatch() -> None:
        try:
            rows.extend(
                iter_scheduled_additive_results(
                    local_pool=_Pool(),
                    local_fn=lambda job: {**job, "source": "local"},
                    jobs=jobs,
                    remote_clients=[template],  # type: ignore[list-item]
                    kind="play",
                    scheduler=_QueueScheduler(),
                    local_workers=0,
                    remote_workers=4,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - report thread failure
            failures.append(exc)

    dispatch = threading.Thread(target=_run_dispatch, daemon=True)
    dispatch.start()
    try:
        assert all_packets_started.wait(timeout=3.0), (
            "public prefetch opened eight request sockets but only "
            f"{len(entered_by_client)} entered a packet: {submitted_packets!r}"
        )
        assert len(clients) == 8
        assert len(entered_by_client) == 8
        assert all(len(packet) == 4 for _client, packet in submitted_packets)
    finally:
        release.set()
        dispatch.join(timeout=8.0)

    assert not dispatch.is_alive()
    assert not failures
    assert sorted(int(row["job_index"]) for row in rows) == list(range(32))


def test_scheduled_public_queue_without_r182_capability_stays_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old self-play-multi worker must never receive a public packet."""

    jobs = [
        _public_job(
            index,
            opponent_id,
            training_group=SAFE_PUBLIC_TRAINING_GROUPS[opponent_id],
        )
        for index, opponent_id in enumerate(SAFE_PUBLIC_IDS[:4])
    ]
    remote = _PublicPackRemote(public_multi_env_capable=False)
    executions: list[dict[str, Any]] = []

    rows = _scheduled_remote_rows(monkeypatch, remote, jobs, executions=executions)

    assert sorted(int(row["job_index"]) for row in rows) == list(range(4))
    assert remote.multi_calls == []
    assert [kind for kind, _job in remote.single_calls] == ["play"] * 4
    assert {event["transport_kind"] for event in executions} == {"play"}
    assert {event["kind"] for event in executions} == {"play"}


def test_recordless_safe_public_pack_keeps_every_exact_cell_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-record packet child must not silently satisfy a strong-public cell."""

    jobs = [
        _public_job(
            index,
            opponent_id,
            training_group=SAFE_PUBLIC_TRAINING_GROUPS[opponent_id],
        )
        for index, opponent_id in enumerate(sorted(SAFE_STRONG_PUBLIC_IDS)[:4])
    ]
    remote = _PublicPackRemote(record_json=None)
    rows = _scheduled_remote_rows(monkeypatch, remote, jobs)
    assert [len(packet) for packet in remote.multi_calls] == [4]
    assert remote.single_calls == []

    writer = _Writer()
    stats = _stats()
    seen: set[int] = set()
    successful: set[int] = set()
    written: set[int] = set()
    contracts = {
        int(job["job_index"]): {
            "opponent_id": str(job["opponent_id"]),
            "opponent_archetype_id": "test-public",
            "active_gate_id": "alakazam-strong-public-roster-v1",
            "our_seat": str(job["our_seat"]),
        }
        for job in jobs
    }

    trainer._consume_results(
        rows,
        writer,
        [],
        stats,
        practice_record_contracts=contracts,
        practice_seen_indices=seen,
        practice_successful_indices=successful,
        practice_written_indices=written,
    )

    expected = set(contracts)
    assert seen == expected
    assert successful == written == set()
    assert writer.games == []
    assert stats["strong_public_practice_recordless_results"] == len(jobs)
