"""Focused create-only checks for the r241 H10 receipt producer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/generate_r241_marnie_direct_policy_adapter_receipt.py"


def _module():
    spec = importlib.util.spec_from_file_location("r241_h10_receipt_producer", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "h10"
    package.mkdir()
    (package / "model.pt").write_bytes(b"model")
    (package / "matchup_tree.json").write_text("{}", encoding="utf-8")
    cg_root = tmp_path / "cg-r236"
    cg_root.mkdir()
    output = tmp_path / "runtime" / "marnie-h10-direct-policy-adapter-r251-v8.json"
    output.parent.mkdir()
    return package, cg_root, output


def test_h10_receipt_producer_validates_temp_before_create_only_publish(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    package, cg_root, output = _paths(tmp_path)
    observed: list[Path] = []

    def validate(spec, *, environment):
        receipt = Path(environment[module.runtime.R241_DIRECT_POLICY_RECEIPT_ENV])
        observed.append(receipt)
        assert receipt.is_file()
        return (
            (package / "model.pt").resolve(),
            [1] * 60,
            (package / "matchup_tree.json").resolve(),
            receipt.resolve(),
        )

    monkeypatch.setattr(module.adapter, "validate_r241_marnie_direct_policy_adapter", validate)
    published = module._create_only_validate_and_publish(
        payload={"receipt": "test"},
        output=output,
        package_root=package,
        cg_root=cg_root,
    )

    assert output.is_file()
    assert output.stat().st_mode & 0o777 == 0o444
    assert json.loads(output.read_text(encoding="utf-8")) == {"receipt": "test"}
    assert observed[0] != output
    assert observed[-1] == output
    assert published["path"] == str(output)

    with pytest.raises(module.R241MarnieAdapterReceiptError, match="already exists"):
        module._create_only_validate_and_publish(
            payload={"receipt": "different"},
            output=output,
            package_root=package,
            cg_root=cg_root,
        )


def test_h10_receipt_producer_leaves_no_final_receipt_after_validation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    package, cg_root, output = _paths(tmp_path)

    def reject(*_args, **_kwargs):
        raise RuntimeError("data-only validation failed")

    monkeypatch.setattr(module.adapter, "validate_r241_marnie_direct_policy_adapter", reject)
    with pytest.raises(RuntimeError, match="data-only validation failed"):
        module._create_only_validate_and_publish(
            payload={"receipt": "test"},
            output=output,
            package_root=package,
            cg_root=cg_root,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "inactive_filename",
    (
        "marnie-h10-direct-policy-adapter-r251.json",
        "marnie-h10-direct-policy-adapter-r251-v2.json",
        "marnie-h10-direct-policy-adapter-r251-v3.json",
        "marnie-h10-direct-policy-adapter-r251-v4.json",
        "marnie-h10-direct-policy-adapter-r251-v5.json",
        "marnie-h10-direct-policy-adapter-r251-v6.json",
        "marnie-h10-direct-policy-adapter-r251-v7.json",
    ),
)
def test_h10_receipt_producer_rejects_the_inactive_predecessor_filename(
    tmp_path: Path, inactive_filename: str,
) -> None:
    module = _module()

    with pytest.raises(
        module.R241MarnieAdapterReceiptError,
        match="predeclared successor path",
    ):
        module.generate_receipt(
            source_snapshot_root=tmp_path,
            source_snapshot_manifest=tmp_path / "manifest.json",
            cg_lib_path=tmp_path,
            package_root=tmp_path,
            output=tmp_path / inactive_filename,
        )
