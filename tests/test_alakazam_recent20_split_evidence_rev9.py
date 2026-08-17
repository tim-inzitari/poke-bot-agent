from poke_bot.alakazam_recent20_split_evidence_rev9 import (
    EVALUATION_DAYS,
    TRAIN_DAYS,
    VALIDATION_DAYS,
    _int_field,
    _text_field,
)


def test_split_days_are_exact_and_disjoint() -> None:
    parts = [set(TRAIN_DAYS), set(VALIDATION_DAYS), set(EVALUATION_DAYS)]
    assert [len(part) for part in parts] == [14, 3, 3]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]
    assert min(parts[0] | parts[1] | parts[2]) == "2026-07-23"
    assert max(parts[0] | parts[1] | parts[2]) == "2026-08-11"


def test_compact_canonical_source_fields_extract() -> None:
    fragment = (
        b'{"acting_seat":1,"episode_id":"abc","env_step":22,'
        b'"source_archive_date":"2026-08-11","factorized_stage":3}'
    )
    assert _int_field(fragment, b'"acting_seat":') == 1
    assert _int_field(fragment, b'"env_step":') == 22
    assert _int_field(fragment, b'"factorized_stage":') == 3
    assert _text_field(fragment, b'"episode_id":"') == "abc"
