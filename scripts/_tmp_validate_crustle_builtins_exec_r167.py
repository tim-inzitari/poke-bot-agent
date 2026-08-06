"""Validate Crustle r167 promote patch applies during `python train_pure_rl.py`."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "train_pure_rl.py"
        fake.write_text(
            textwrap.dedent(
                """
                def _replacement_schedule_contract_from_result(runtime_audit_row, records):
                    return {"opp_archetype": "ORIGINAL"}

                # Must already be rebound before script body continues.
                _fn = _replacement_schedule_contract_from_result
                print("MID_EXEC", _fn.__name__)
                row = {
                    "our_seat": 0,
                    "opponent_id": "specialist-marnie-final-format-h10-x",
                    "archetype": "crustle",
                }
                rec = [{
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
                }]
                print("MID_EXEC_VAL", _fn(row, rec)["opp_archetype"])
                """
            ),
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136"
        )
        proc = subprocess.run(
            [sys.executable, str(fake)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            return proc.returncode
        if "MID_EXEC replacement_schedule_contract_from_result" not in proc.stdout:
            raise SystemExit("mid-exec name validation failed (settrace path)")
        if "MID_EXEC_VAL marnie-s-grimmsnarl-ex" not in proc.stdout:
            raise SystemExit("mid-exec normalize validation failed")
    print("SCRIPT_TRACE_HOOK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
