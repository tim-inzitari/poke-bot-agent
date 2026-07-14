import json
from pathlib import Path

from poke_bot import dataset, features
from poke_bot.dataset import DecisionSample


def _sv(words: int = 1) -> features.SparseVector:
    sv = features.SparseVector()
    for _ in range(words):
        sv.word_start()
        sv.add(0, 1.0)
    return sv


def _fake_featurize(step, _deck, *, verify_info_set=True):
    del verify_info_set
    return DecisionSample(
        board=_sv(features.NUM_BOARD_TOKENS),
        options=_sv(1),
        action=[0],
        action_combo_index=0,
        action_combos=[[0]],
        env_step=int(step["env_step"]),
    )


def _record(n_steps: int = 5) -> dict:
    return {
        "episode_id": "episode",
        "seat": 0,
        "archetype": "a",
        "opp_archetype": "b",
        "deck": [1] * 60,
        "value": 1.0,
        "steps": [{"env_step": i} for i in range(n_steps)],
        "policy_targets": [[float(i)] for i in range(n_steps)],
        "info_set_ok": True,
    }


def test_context_and_policy_targets_truncate_together(monkeypatch) -> None:
    monkeypatch.setattr(dataset, "featurize_step", _fake_featurize)
    seq, reason, details = dataset.convert_record(_record(), max_context=2)
    assert reason is None
    assert [d.env_step for d in seq.decisions] == [3, 4]
    assert seq.policy_targets == [[3.0], [4.0]]
    assert details["decisions_truncated"] == 3

    short = _record()
    short["policy_targets"] = [[0.0], [1.0], [2.0]]
    seq2, reason2, details2 = dataset.convert_record(short, max_context=2)
    assert reason2 is None
    assert seq2.policy_targets == [None, None]
    assert details2["policy_targets_padded"] == 2


def test_dataset_reports_conversion_drop_reasons(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset, "featurize_step", _fake_featurize)
    invalid_deck = _record(1)
    invalid_deck["deck"] = [1]
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(_record(1))
        + "\n"
        + "{invalid json\n"
        + json.dumps(invalid_deck)
        + "\n",
        encoding="utf-8",
    )
    ds = dataset.BootstrapDataset.from_jsonl(path, use_cache=False)
    stats = ds.summary()["conversion"]
    assert stats["records_total"] == 3
    assert stats["records_kept"] == 1
    assert stats["records_dropped"] == 2
    assert stats["drop_reasons"] == {
        "invalid_json": 1,
        "invalid_deck_length": 1,
    }


def test_cache_key_includes_feature_and_dataset_schema(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    key1 = dataset._cache_key(path, 8, True, 0)
    monkeypatch.setattr(features, "FEATURE_SCHEMA_VERSION", 999)
    key2 = dataset._cache_key(path, 8, True, 0)
    assert key1 != key2
    assert dataset._cache_key(path, 8, True, 1) != key2
