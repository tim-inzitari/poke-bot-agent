#!/usr/bin/env python3
"""Seal the pinned r236 public card catalog for the isolated r298 derivative.

The competition simulator is the mechanics authority.  ``EN_Card_Data.csv``
is supplied only to the existing provenance-checked identity join in
``poke_bot.card_metadata``; card prose is never parsed into new rules here.

This command is deliberately create-only and does nothing without
``--execute --elmo-ack``.  It neither loads a policy checkpoint nor touches a
service, selector, production artifact, or Inzi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CATALOG_SCHEMA = "poke_bot.alakazam_public_catalog_r298/v1"
VECTOR_SCHEMA = "poke_bot.alakazam_structured_rule_vectors_r298/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_public_catalog_r298_receipt/v1"
READY_SCHEMA = "poke_bot.alakazam_public_catalog_r298_ready/v1"

GOAL_PATH = REPO_ROOT / "goals/alakazam-elmo-rule-derivative/GOAL.md"
CONTRACT_PATH = REPO_ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
EXPECTED_GOAL_SHA256 = "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
EXPECTED_CONTRACT_SHA256 = "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
EXPECTED_LIBCG_SHA256 = "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
EXPECTED_LIBCG_SIZE = 1_342_400
EXPECTED_CSV_SHA256 = "sha256:408bc978661c8b0628e5f17b27693dc8da9c732472168f5574999be4774031c1"
EXPECTED_ELMO_HOSTNAME = "truenas"


class SealError(RuntimeError):
    """The catalog cannot be sealed against the required immutable inputs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _verify_exact_source(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise SealError(f"{label} identity does not match the r298 contract")


def _source_digest() -> str:
    return _sha256_file(Path(__file__).resolve())


def _peak_rss_bytes() -> int:
    """Return this isolated sealing process's observed peak resident bytes.

    Elmo is Linux, where ``ru_maxrss`` is reported in KiB.  The hostname gate
    above prevents this receipt from silently acquiring macOS byte semantics.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) * 1024


def _structured_catalog(catalog: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project the validated engine catalog to non-text structured mechanics.

    Existing Card2Vec and its learned/text-derived inputs remain completely
    untouched.  This projection exists only for the new additive rule
    residual, so names, prose, and text hashes are deliberately absent.
    """

    parents = {
        int(child): [int(value) for value in values]
        for child, values in catalog.evolution_parents.items()
    }
    cards: list[dict[str, Any]] = []
    for source in catalog.cards:
        card_id = int(source["cardId"])
        cards.append(
            {
                "cardId": card_id,
                "cardType": int(source["cardType"]),
                "retreatCost": int(source["retreatCost"]),
                "hp": int(source["hp"]),
                "weakness": source["weakness"],
                "resistance": source["resistance"],
                "energyType": int(source["energyType"]),
                "basic": bool(source["basic"]),
                "stage1": bool(source["stage1"]),
                "stage2": bool(source["stage2"]),
                "ex": bool(source["ex"]),
                "megaEx": bool(source["megaEx"]),
                "tera": bool(source["tera"]),
                "aceSpec": bool(source["aceSpec"]),
                "evolutionParentCardIds": parents.get(card_id, []),
                "skillCount": len(source["skills"]),
                "attacks": [int(value) for value in source["attacks"]],
            }
        )
    attacks = [
        {
            "attackId": int(source["attackId"]),
            "damage": int(source["damage"]),
            "energies": [int(value) for value in source["energies"]],
        }
        for source in catalog.attacks
    ]
    return cards, attacks


def _structured_vectors(catalog: Any, torch: Any) -> tuple[Any, Any, str]:
    """Return only fixed structured columns for the new residual.

    ``card_metadata`` stores structured card columns in ``[:64]`` and begins
    its signed text hashes at column 64.  Attack text hashes begin at column
    20; column 16 is merely a text-presence indicator and is explicitly zeroed
    here.  This does not alter the legacy catalog or Card2Vec tensors.
    """

    card = catalog.card_features[:, :64].detach().cpu().clone()
    attack = catalog.attack_features[:, :20].detach().cpu().clone()
    attack[:, 16:] = 0.0
    digest = hashlib.sha256()
    for tensor in (card, attack):
        value = tensor.to(dtype=torch.float32).contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return card, attack, "sha256:" + digest.hexdigest()


def _seal(args: argparse.Namespace) -> Path:
    started_monotonic = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    host = socket.gethostname().split(".", 1)[0].casefold()
    if host != EXPECTED_ELMO_HOSTNAME:
        raise SealError(f"catalog sealing is Elmo-only (host={host!r})")
    if not args.elmo_ack:
        raise SealError("--elmo-ack is required")

    runtime = args.runtime.resolve()
    library = runtime / "cg/libcg.so"
    csv_path = args.csv.resolve()
    _verify_exact_source(GOAL_PATH, EXPECTED_GOAL_SHA256, "dedicated goal")
    _verify_exact_source(CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "dedicated contract")
    _verify_exact_source(library, EXPECTED_LIBCG_SHA256, "canonical libcg")
    if library.stat().st_size != EXPECTED_LIBCG_SIZE:
        raise SealError("canonical libcg size drifted")
    _verify_exact_source(csv_path, EXPECTED_CSV_SHA256, "card CSV identity join")

    # Set this only after the exact binary identity has passed.  Importing cg
    # earlier could bind a host-specific or production runtime by accident.
    os.environ["CG_LIB_PATH"] = str(runtime)
    import torch

    from poke_bot import cg_env
    from poke_bot.card_metadata import build_metadata_catalog

    catalog = build_metadata_catalog(
        cg_env.all_card_data(),
        cg_env.all_attack(),
        csv_path=csv_path,
    )
    provenance = dict(catalog.provenance)
    if provenance.get("csv_sha256") != EXPECTED_CSV_SHA256:
        raise SealError("metadata builder did not retain the exact CSV identity")
    structured_cards, structured_attacks = _structured_catalog(catalog)
    structured_card_vectors, structured_attack_vectors, structured_vectors_sha = (
        _structured_vectors(catalog, torch)
    )

    output = args.output_dir.resolve()
    if output.exists():
        raise SealError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))

    catalog_payload: dict[str, Any] = {
        "schema": CATALOG_SCHEMA,
        "revision": 298,
        "status": "sealed_public_simulator_catalog",
        "authority": {
            "mechanics": "pinned_official_libcg_r236",
            "csv_role": "provenance_checked_identity_join_only",
            "card_text_rules_parser": False,
            "card2vec_preserved_unchanged": True,
            "legacy_text_derived_embeddings_preserved_but_not_rule_authority": True,
            "new_residual_vector_source": "structured_engine_fields_only",
            "policy_or_runtime_authority": False,
        },
        "source": {
            "goal_sha256": EXPECTED_GOAL_SHA256,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "libcg_sha256": EXPECTED_LIBCG_SHA256,
            "libcg_size_bytes": EXPECTED_LIBCG_SIZE,
            "csv_sha256": EXPECTED_CSV_SHA256,
            "sealer_source_sha256": _source_digest(),
        },
        "provenance": {
            **provenance,
            "legacy_fixed_vectors_sha256_not_authorized_for_r298_rules": provenance.get(
                "fixed_vectors_sha256"
            ),
            "structured_rule_vectors_sha256": structured_vectors_sha,
            "structured_rule_vector_schema": VECTOR_SCHEMA,
        },
        "cards": structured_cards,
        "attacks": structured_attacks,
    }
    catalog_path = staging / "catalog.json"
    vectors_path = staging / "fixed-vectors.pt"
    _write_json_create_only(catalog_path, catalog_payload)
    with vectors_path.open("xb") as stream:
        torch.save(
            {
                "schema": VECTOR_SCHEMA,
                "provenance": {
                    **provenance,
                    "structured_rule_vectors_sha256": structured_vectors_sha,
                    "text_or_name_hash_columns_included": False,
                    "card2vec_tensor_mutation": False,
                },
                "card_features": structured_card_vectors,
                "attack_features": structured_attack_vectors,
            },
            stream,
        )

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "revision": 298,
        "status": "passed_elmo_only_nonproduction",
        "execution_host_role": "elmo",
        "execution_hostname": host,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "goal_sha256": EXPECTED_GOAL_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "libcg_sha256": EXPECTED_LIBCG_SHA256,
        "libcg_size_bytes": EXPECTED_LIBCG_SIZE,
        "csv_sha256": EXPECTED_CSV_SHA256,
        "catalog_schema": CATALOG_SCHEMA,
        "catalog_semantic_sha256": _canonical_sha256(catalog_payload),
        "catalog_file_sha256": _sha256_file(catalog_path),
        "catalog_file_size_bytes": catalog_path.stat().st_size,
        "fixed_vectors_file_sha256": _sha256_file(vectors_path),
        "fixed_vectors_file_size_bytes": vectors_path.stat().st_size,
        "metadata_provenance": catalog_payload["provenance"],
        "facts": {
            "card_count": len(catalog.cards),
            "attack_count": len(catalog.attacks),
            "mechanics_from_pinned_engine": True,
            "csv_used_as_mechanics_authority": False,
            "text_hash_used_as_rules_parser": False,
            "text_or_name_hash_in_new_residual_vectors": False,
            "card2vec_preserved_unchanged": True,
            "structured_rule_vectors_sha256": structured_vectors_sha,
        },
        "resource_peak_receipt": {
            "schema": "poke_bot.alakazam_elmo_rule_derivative_resource_peak/v1",
            "host": host,
            "started_at_utc": started_at_utc,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "experiment_ram_measurement_method": "linux_getrusage_ru_maxrss_single_isolated_process_no_children",
            "experiment_ram_peak_bytes": _peak_rss_bytes(),
            "aggregate_experiment_ram_limit_bytes": 96 * 1024**3,
            "cpu_process_or_utilization_peak": {
                "measurement": "single_sealing_process",
                "process_count_peak": 1,
            },
            "gpu_memory_peak_bytes_per_device_or_not_applicable": "not_applicable_cpu_catalog_seal",
            "gpu_utilization_peak_per_device_or_not_applicable": "not_applicable_cpu_catalog_seal",
            "zfs_arc_or_l2arc_tuning_performed": False,
        },
        "authority": {
            "elmo_only": True,
            "create_only": True,
            "training_eligibility_by_itself": False,
            "production_or_inzi": False,
            "runtime_activation": False,
        },
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    receipt_path = staging / "receipt.json"
    _write_json_create_only(receipt_path, receipt)
    ready: dict[str, Any] = {
        "schema": READY_SCHEMA,
        "status": "ready_for_r298_schema_freeze_and_refeaturization",
        "catalog_file_sha256": receipt["catalog_file_sha256"],
        "vectors_file_sha256": receipt["fixed_vectors_file_sha256"],
        "receipt_file_sha256": _sha256_file(receipt_path),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "training_or_runtime_authority": False,
    }
    _write_json_create_only(staging / "READY.json", ready)
    staging.rename(output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--elmo-ack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        print("default-off: pass --execute --elmo-ack to seal the catalog")
        return 0
    try:
        result = _seal(args)
    except (OSError, ValueError, SealError) as exc:
        print(f"FAIL CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
