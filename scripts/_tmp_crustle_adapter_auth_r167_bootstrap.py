"""Bootstrap Crustle r167 overlays without dirtying the hashed deploy tree."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys
import importlib
import importlib.machinery

_OVERLAY = Path("/home/inzi/poke-bot-agent/overlays/crustle-adapter-auth-r167")


def _run_overlay(name: str) -> None:
    patch = _OVERLAY / name
    if patch.is_file():
        runpy.run_path(str(patch), run_name=name.replace(".py", ""))


# Adapter-auth repair (ActivationReceipt-safe rebind).
_run_overlay("crustle_adapter_auth_r167_patch.py")


def _apply_promote_patch() -> None:
    patch = _OVERLAY / "crustle_promote_contract_r167_patch.py"
    if not patch.is_file():
        return
    overlay = str(_OVERLAY)
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    try:
        from crustle_promote_contract_r167_patch import (
            apply_train_pure_rl_promote_patch,
            replacement_schedule_contract_from_result,
        )
    except Exception:
        ns = runpy.run_path(str(patch), run_name="crustle_promote_contract_r167_patch")
        apply_train_pure_rl_promote_patch = ns.get("apply_train_pure_rl_promote_patch")
        replacement_schedule_contract_from_result = ns.get(
            "replacement_schedule_contract_from_result"
        )
        if not callable(apply_train_pure_rl_promote_patch):
            return
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        file = str(getattr(mod, "__file__", "") or "")
        if "train_pure_rl.py" not in file.replace("\\", "/"):
            continue
        if hasattr(mod, "_replacement_schedule_contract_from_result"):
            mod._replacement_schedule_contract_from_result = (
                replacement_schedule_contract_from_result
            )
    try:
        apply_train_pure_rl_promote_patch()
    except Exception:
        pass


_orig_import_module = importlib.import_module


def _import_module(name, package=None):
    mod = _orig_import_module(name, package=package)
    if name == "scripts.train_pure_rl":
        _apply_promote_patch()
    return mod


if getattr(importlib.import_module, "_crustle_r167_promote", False) is not True:
    _import_module._crustle_r167_promote = True
    importlib.import_module = _import_module


_orig_exec_module = importlib.machinery.SourceFileLoader.exec_module


def _exec_module(self, module):
    _orig_exec_module(self, module)
    file = str(getattr(module, "__file__", "") or "")
    if "train_pure_rl.py" in file.replace("\\", "/"):
        _apply_promote_patch()


if getattr(importlib.machinery.SourceFileLoader.exec_module, "_crustle_r167_promote", False) is not True:
    _exec_module._crustle_r167_promote = True
    importlib.machinery.SourceFileLoader.exec_module = _exec_module
