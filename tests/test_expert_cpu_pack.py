from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot.device_corpus import (
    DEVICE_CORPUS_PACKING_SCHEMA_VERSION,
    DeviceResidentBootstrapCorpus,
)
from poke_bot.pure_rl.expert_cpu_pack import (
    ExpertCpuPackCache,
    ExpertCpuPackError,
    ExpertCpuPackKey,
    validate_cpu_corpus,
)
from poke_bot.pure_rl.expert_rehearsal import (
    ExpertManifestIdentity,
    ResidentExpertCorpusCache,
)


def _tiny_corpus() -> DeviceResidentBootstrapCorpus:
    tensors = {
        "board_index": torch.empty(0, dtype=torch.int32),
        "board_value": torch.empty(0, dtype=torch.float32),
        "board_offset": torch.zeros(2 * 24 + 1, dtype=torch.int32),
        "option_index": torch.empty(0, dtype=torch.int32),
        "option_value": torch.empty(0, dtype=torch.float32),
        "option_offset": torch.zeros(3, dtype=torch.int32),
        "sample_board": torch.tensor([0, 1], dtype=torch.int32),
        "option_word_start": torch.tensor([0, 1], dtype=torch.int32),
        "n_options": torch.tensor([1, 1], dtype=torch.int16),
        "target_index": torch.tensor([0, 0], dtype=torch.int16),
        "value_target": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "guide_target_index": torch.tensor([-1, -1], dtype=torch.int16),
        "guide_confidence": torch.zeros(2, dtype=torch.float32),
        "action_index": torch.empty(0, dtype=torch.int32),
        "action_value": torch.empty(0, dtype=torch.float32),
        "action_offset": torch.zeros(3, dtype=torch.int32),
        "game_decision_offset": torch.tensor([0, 2], dtype=torch.int32),
        "game_sample_offset": torch.tensor([0, 2], dtype=torch.int32),
    }
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in tensors.values()
    )
    return DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=tensors,
        scalars={
            "train_samples": 2,
            "val_samples": 0,
            "train_games": 1,
            "val_games": 0,
            "decisions": 2,
            "input_bytes": tensor_bytes,
            "build_seconds": 0.25,
            "belief_card_vocab": 0,
        },
    )


def _key(**overrides) -> ExpertCpuPackKey:
    values = {
        "manifest_digest": "sha256:" + "a" * 64,
        "split_seed": 17,
        "val_frac": 0.10,
        "max_context": 320,
    }
    values.update(overrides)
    return ExpertCpuPackKey(**values)


def test_cpu_pack_round_trip_is_durable_and_skips_builder(tmp_path: Path) -> None:
    cache = ExpertCpuPackCache(tmp_path)
    calls = 0

    def build() -> DeviceResidentBootstrapCorpus:
        nonlocal calls
        calls += 1
        return _tiny_corpus()

    first, first_info = cache.load_or_build(_key(), build)
    assert calls == 1
    assert first_info["cache_hit"] is False
    validate_cpu_corpus(first)
    del first

    def must_not_build() -> DeviceResidentBootstrapCorpus:
        raise AssertionError("durable cache hit rebuilt the source corpus")

    second, second_info = ExpertCpuPackCache(tmp_path).load_or_build(
        _key(), must_not_build
    )
    assert second_info["cache_hit"] is True
    assert second.tensor_bytes == _tiny_corpus().tensor_bytes
    assert second.game_decision_offset is not None
    assert second.game_decision_offset.tolist() == [0, 2]
    active = json.loads((tmp_path / "active.json").read_text())
    assert active["key"] == _key().digest
    assert not list(tmp_path.glob(".*.partial.*"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"manifest_digest": "sha256:" + "b" * 64},
        {"split_seed": 18},
        {"val_frac": 0.2},
        {"max_context": 160},
        {"packing_schema": DEVICE_CORPUS_PACKING_SCHEMA_VERSION + 1},
    ],
)
def test_cpu_pack_key_covers_every_packing_input(overrides: dict) -> None:
    assert _key(**overrides).digest != _key().digest


def test_corrupt_payload_rebuilds_fail_closed(tmp_path: Path) -> None:
    cache = ExpertCpuPackCache(tmp_path)
    _, info = cache.load_or_build(_key(), _tiny_corpus)
    payload = Path(info["payload"])
    with payload.open("r+b") as handle:
        handle.seek(max(0, payload.stat().st_size // 2))
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(b"z" if original != b"z" else b"y")

    calls = 0

    def rebuild() -> DeviceResidentBootstrapCorpus:
        nonlocal calls
        calls += 1
        return _tiny_corpus()

    corpus, rebuilt = ExpertCpuPackCache(tmp_path).load_or_build(_key(), rebuild)
    assert calls == 1
    assert rebuilt["cache_hit"] is False
    validate_cpu_corpus(corpus)
    assert not list(tmp_path.glob(".*.partial.*"))


def test_only_new_active_pack_is_retained(tmp_path: Path) -> None:
    cache = ExpertCpuPackCache(tmp_path)
    cache.load_or_build(_key(), _tiny_corpus)
    newer = _key(split_seed=99)
    cache.load_or_build(newer, _tiny_corpus)

    assert {path.name for path in tmp_path.glob("expert-pack-*.pt")} == {
        f"expert-pack-{newer.digest}.pt"
    }
    assert {path.name for path in tmp_path.glob("expert-pack-*.json")} == {
        f"expert-pack-{newer.digest}.json"
    }
    assert json.loads((tmp_path / "active.json").read_text())["key"] == newer.digest


def test_structurally_invalid_pack_is_never_written(tmp_path: Path) -> None:
    bad = _tiny_corpus()
    bad.sample_board[1] = 99
    with pytest.raises(ExpertCpuPackError, match="invalid board"):
        ExpertCpuPackCache(tmp_path).load_or_build(_key(), lambda: bad)
    assert not list(tmp_path.glob("expert-pack-*"))
    assert not list(tmp_path.glob(".*.partial.*"))


def test_resident_expert_cache_restart_bypasses_source_load_and_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_bot.pure_rl.expert_feature_stream import (
        EpisodeGroupedFeatureManifest,
    )

    identity = ExpertManifestIdentity(
        path=str(tmp_path / "manifest.json"),
        digest="sha256:" + "c" * 64,
        dates=("2026-07-20",),
        decisions=2,
        records=1,
    )
    source_opens = 0
    packs = 0

    class FakePlan:
        decisions = 2
        max_context = None
        truncated_sequences = 0

        @staticmethod
        def splits():
            return [object()], []

    def open_manifest(*_args, **_kwargs):
        nonlocal source_opens
        source_opens += 1
        return FakePlan()

    def pack(_cls, *_args, **_kwargs):
        nonlocal packs
        packs += 1
        return _tiny_corpus()

    monkeypatch.setattr(
        EpisodeGroupedFeatureManifest,
        "open",
        classmethod(open_manifest),
    )
    monkeypatch.setattr(
        DeviceResidentBootstrapCorpus, "from_splits", classmethod(pack)
    )
    first = ResidentExpertCorpusCache(cpu_pack_root=tmp_path / "pack")
    first.prepare(
        identity,
        device=torch.device("cpu"),
        seed=123,
        val_frac=0.1,
        max_context=None,
    )
    assert source_opens == 1 and packs == 1
    first.release()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache restart touched the source feature corpus")

    monkeypatch.setattr(
        EpisodeGroupedFeatureManifest,
        "open",
        classmethod(forbidden),
    )
    monkeypatch.setattr(
        DeviceResidentBootstrapCorpus,
        "from_splits",
        classmethod(lambda *_args, **_kwargs: forbidden()),
    )
    restarted = ResidentExpertCorpusCache(cpu_pack_root=tmp_path / "pack")
    corpus = restarted.prepare(
        identity,
        device=torch.device("cpu"),
        seed=123,
        val_frac=0.1,
        max_context=None,
    )
    assert corpus.train_samples == 2
    assert restarted.pack_info is not None
    assert restarted.pack_info["cache_hit"] is True
    assert source_opens == 1 and packs == 1
