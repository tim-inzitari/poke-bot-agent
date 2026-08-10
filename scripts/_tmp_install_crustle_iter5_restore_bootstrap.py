#!/usr/bin/env python3
"""Install iter5 corpus restore hooks into the Crustle r167 bootstrap overlay."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

OVERLAY = Path("/home/inzi/poke-bot-agent/overlays/crustle-adapter-auth-r167")
SITE = Path(
    "/home/inzi/miniconda3/envs/poke-bot-agent/lib/python3.11/site-packages"
)
SRC_PATCH = Path(
    "/home/inzi/poke-bot-agent/scripts/_tmp_crustle_iter5_corpus_restore_r167_patch.py"
)


def main() -> int:
    overlay_patch = OVERLAY / "crustle_iter5_corpus_restore_r167_patch.py"
    shutil.copy2(SRC_PATCH, overlay_patch)
    shutil.copy2(overlay_patch, SITE / overlay_patch.name)

    bootstrap = OVERLAY / "crustle_adapter_auth_r167_bootstrap.py"
    bak = OVERLAY / "crustle_adapter_auth_r167_bootstrap.py.bak_pre_iter5_restore"
    if not bak.exists():
        shutil.copy2(bootstrap, bak)
    text = bootstrap.read_text(encoding="utf-8")

    if "crustle_iter5_corpus_restore_r167_patch.py" not in text:
        needle = (
            '# Adapter-auth repair (ActivationReceipt-safe rebind).\n'
            '_run_overlay("crustle_adapter_auth_r167_patch.py")\n'
        )
        insert = (
            needle
            + "\n# Iter5 corpus restore/resume (no recollection).\n"
            + '_run_overlay("crustle_iter5_corpus_restore_r167_patch.py")\n'
        )
        if needle not in text:
            raise SystemExit("bootstrap adapter-auth needle missing")
        text = text.replace(needle, insert, 1)

    old_fn_lines = [
        "def _apply_promote_patch_to_mapping(mapping) -> bool:",
        "    if not isinstance(mapping, dict):",
        "        return False",
        '    file = str(mapping.get("__file__", "") or "")',
        '    if "train_pure_rl.py" not in file.replace("\\\\", "/"):',
        "        return False",
        '    if "_replacement_schedule_contract_from_result" not in mapping:',
        "        return False",
        "    fn = _load_replacement_fn()",
        "    if fn is None:",
        "        return False",
        '    current = mapping.get("_replacement_schedule_contract_from_result")',
        '    if getattr(current, "__name__", "") == "replacement_schedule_contract_from_result":',
        "        return True",
        '    mapping["_replacement_schedule_contract_from_result"] = fn',
        "    return True",
    ]
    # Source file contains a single escaped backslash in the replace() call.
    old_fn = "\n".join(old_fn_lines).replace('replace("\\\\\\\\", "/")', 'replace("\\\\", "/")')
    new_fn = '''def _apply_promote_patch_to_mapping(mapping) -> bool:
    if not isinstance(mapping, dict):
        return False
    file = str(mapping.get("__file__", "") or "")
    if "train_pure_rl.py" not in file.replace("\\\\", "/"):
        return False
    try:
        from crustle_iter5_corpus_restore_r167_patch import (
            apply_iter5_restore_patches,
        )

        apply_iter5_restore_patches(mapping)
    except Exception:
        try:
            ns = runpy.run_path(
                str(_OVERLAY / "crustle_iter5_corpus_restore_r167_patch.py"),
                run_name="crustle_iter5_corpus_restore_r167_patch",
            )
            fn_restore = ns.get("apply_iter5_restore_patches")
            if callable(fn_restore):
                fn_restore(mapping)
        except Exception:
            pass
    if "_replacement_schedule_contract_from_result" not in mapping:
        return False
    fn = _load_replacement_fn()
    if fn is None:
        return False
    current = mapping.get("_replacement_schedule_contract_from_result")
    if getattr(current, "__name__", "") != "replacement_schedule_contract_from_result":
        mapping["_replacement_schedule_contract_from_result"] = fn
    return True
'''.replace('replace("\\\\\\\\", "/")', 'replace("\\\\", "/")')
    if "apply_iter5_restore_patches" not in text:
        if old_fn not in text:
            raise SystemExit(
                "could not locate _apply_promote_patch_to_mapping block\n"
                + repr(text[text.find("def _apply_promote_patch_to_mapping") : text.find("def _apply_promote_patch_to_mapping") + 500])
            )
        text = text.replace(old_fn, new_fn, 1)

    bootstrap.write_text(text, encoding="utf-8")
    shutil.copy2(bootstrap, SITE / bootstrap.name)
    # Invalidate pyc
    for pyc in (SITE / "__pycache__").glob("crustle_adapter_auth_r167_bootstrap*.pyc"):
        pyc.unlink()
    for pyc in (OVERLAY / "__pycache__").glob("crustle_iter5_corpus_restore_r167_patch*.pyc"):
        pyc.unlink()
    for pyc in (SITE / "__pycache__").glob("crustle_iter5_corpus_restore_r167_patch*.pyc"):
        pyc.unlink()

    digest = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    print("bootstrap_sha256", digest)
    print("has_restore_overlay", "crustle_iter5_corpus_restore_r167_patch.py" in text)
    print("has_apply_restore", "apply_iter5_restore_patches" in text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
