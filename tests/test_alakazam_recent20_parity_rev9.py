from poke_bot.alakazam_recent20_parity_rev9 import build_manifest, build_parity_receipt, canonical_sha256


def _day(day: str, host: str, index: int):
    digest = "sha256:" + f"{index + 1:064x}"
    receipt = {
        "schema": "poke_bot.alakazam_recent20_intraday_refeature_day_receipt/v1",
        "status": "complete", "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "hostname": "truenas" if host == "elmo" else "inzi-MS-7C35", "utc_day": day,
        "record_count": 10,
        "shard_manifest": {"shard_count": 1, "total_bytes": 100, "record_count": 10,
            "shards": [{"utc_day": day, "filename": f"sha256-{digest[7:]}.jsonl", "sha256": digest, "size_bytes": 100, "record_count": 10}]},
    }
    return receipt, f"/{host}/{day}/COMPLETE.json", "sha256:" + f"{100 + index:064x}"


def test_recent20_manifest_and_parity_close_exact_inventory():
    days = [f"2026-07-{d:02d}" for d in range(23, 32)] + [f"2026-08-{d:02d}" for d in range(1, 12)]
    rows = [_day(day, "inzi" if i % 2 == 0 else "elmo", i) for i, day in enumerate(days)]
    manifest = build_manifest(day_receipts=rows, elmo_host_receipt_path="/e", elmo_host_receipt_sha256="sha256:e", inzi_host_receipt_path="/i", inzi_host_receipt_sha256="sha256:i", raw_manifest_sha256="sha256:r", sealed_at_utc="now")
    transfers = []
    observations = {}
    for row in manifest["shards"]:
        observations[row["sha256"]] = {"sha256": row["sha256"], "size_bytes": 100}
        if row["source_host"] == "elmo":
            transfers.append(({"schema": "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v2", "goal_revision": 9, "goal_contract_sha256": manifest["goal_contract_sha256"], "source_sha256": row["sha256"], "destination_validation_passed": True, "inzi_loading_or_execution_authority": False}, "sha256:t" + row["sha256"][8:]))
    parity = build_parity_receipt(source_manifest=manifest, transfer_receipts=transfers, destination_observations=observations)
    assert parity["source_manifest_sha256"] == canonical_sha256(manifest)
    assert parity["source_and_destination_object_count"] == 20
    assert parity["source_remote_parity_passed"] is True
