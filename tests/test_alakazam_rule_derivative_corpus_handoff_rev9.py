from poke_bot.alakazam_rule_derivative_corpus_handoff_rev9 import (
    GOAL_REVISION,
    SCHEMA,
    canonical_bytes,
)


def test_closed_schema_and_canonical_encoding() -> None:
    assert GOAL_REVISION == 9
    assert SCHEMA == "poke_bot.alakazam_rule_derivative_inzi_corpus_receipt/v1"
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
