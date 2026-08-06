"""Validate Crustle r167 promote patch applies via builtins.exec (script launch)."""
from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

SCRIPT = Path(
    "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136"
    "/scripts/train_pure_rl.py"
)


def _smoke(fn_name: str, opp: str) -> None:
    if fn_name != "replacement_schedule_contract_from_result":
        raise SystemExit(f"unexpected fn name: {fn_name}")
    if opp != "marnie-s-grimmsnarl-ex":
        raise SystemExit(f"unexpected opp_archetype: {opp}")


def _call(fn):
    return fn(
        {
            "our_seat": 0,
            "opponent_id": "specialist-marnie-final-format-h10-x",
            "archetype": "crustle",
        },
        [
            {
                "archetype": "crustle",
                "opp_archetype": "specialist-marnie-final-format-h10-x",
                "target_provenance": {
                    "opponent_id": "specialist-marnie-final-format-h10-x",
                    "opponent_archetype_id": "marnie-s-grimmsnarl-ex",
                    "matchup_runtime_audit": {
                        "active_archetype_id": "marnie-s-grimmsnarl-ex"
                    },
                    "opponent_checkpoint_digest": "a",
                    "opponent_content_digest": "b",
                    "opponent_training_group": "public",
                },
            }
        ],
    )["opp_archetype"]


def main() -> int:
    if getattr(builtins.exec, "_crustle_r167_promote", False) is not True:
        raise SystemExit(f"builtins.exec not wrapped: {builtins.exec!r}")

    mini = textwrap.dedent(
        """
        def _replacement_schedule_contract_from_result(runtime_audit_row, records):
            return {"opp_archetype": "ORIGINAL"}
        """
    )
    code = compile(mini, str(SCRIPT), "exec")
    mod = types.ModuleType("__main__")
    mod.__file__ = str(SCRIPT)
    mod.__builtins__ = builtins
    sys.modules["__main__"] = mod
    builtins.exec(code, mod.__dict__)
    fn = mod.__dict__["_replacement_schedule_contract_from_result"]
    inline_opp = _call(fn)
    print("INLINE", fn.__name__, inline_opp)
    _smoke(fn.__name__, inline_opp)

    runner = textwrap.dedent(
        f"""
        import builtins, sys, types, textwrap
        assert getattr(builtins.exec, "_crustle_r167_promote", False) is True
        mini = textwrap.dedent('''
        def _replacement_schedule_contract_from_result(runtime_audit_row, records):
            return {{"opp_archetype": "ORIGINAL"}}
        ''')
        code = compile(mini, {str(SCRIPT)!r}, "exec")
        mod = types.ModuleType("__main__")
        mod.__file__ = {str(SCRIPT)!r}
        mod.__builtins__ = builtins
        sys.modules["__main__"] = mod
        builtins.exec(code, mod.__dict__)
        fn = mod.__dict__["_replacement_schedule_contract_from_result"]
        row = {{
            "our_seat": 0,
            "opponent_id": "specialist-marnie-final-format-h10-x",
            "archetype": "crustle",
        }}
        rec = [{{
            "archetype": "crustle",
            "opp_archetype": "specialist-marnie-final-format-h10-x",
            "target_provenance": {{
                "opponent_id": "specialist-marnie-final-format-h10-x",
                "opponent_archetype_id": "marnie-s-grimmsnarl-ex",
                "matchup_runtime_audit": {{
                    "active_archetype_id": "marnie-s-grimmsnarl-ex"
                }},
                "opponent_checkpoint_digest": "a",
                "opponent_content_digest": "b",
                "opponent_training_group": "public",
            }},
        }}]
        print("SUBPROCESS", fn.__name__, fn(row, rec)["opp_archetype"])
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136"
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    if "SUBPROCESS replacement_schedule_contract_from_result marnie-s-grimmsnarl-ex" not in proc.stdout:
        raise SystemExit("subprocess script-mode validation failed")
    print("BUILTINS_EXEC_HOOK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
