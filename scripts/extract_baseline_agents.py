#!/usr/bin/env python3
"""Extract main.py + deck.csv from pulled Kaggle baseline notebooks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "baselines" / "kernels"
DECKS = ROOT / "baselines" / "decks"
OUT = ROOT / "baselines" / "official"

AGENT_DIRS = ("iono", "dragapult-ex", "mega-abomasnow-ex", "mega-lucario-ex")


def extract_main_py(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        src = "".join(cell.get("source", []))
        if "%%writefile main.py" in src:
            return src.split("%%writefile main.py", 1)[1].lstrip("\n")
    raise ValueError(f"no %%writefile main.py cell in {notebook_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for agent_dir in AGENT_DIRS:
        kernel_dir = KERNELS / agent_dir
        notebooks = sorted(kernel_dir.glob("*.ipynb"))
        if not notebooks:
            raise FileNotFoundError(f"missing notebook under {kernel_dir}; run setup_baseline_agents.sh")
        main_py = extract_main_py(notebooks[0])
        deck_src = DECKS / agent_dir / "deck.csv"
        if not deck_src.is_file():
            raise FileNotFoundError(f"missing deck.csv at {deck_src}")

        target = OUT / agent_dir
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        (target / "main.py").write_text(main_py, encoding="utf-8")
        shutil.copy2(deck_src, target / "deck.csv")
        print(f"wrote {target}/main.py + deck.csv ({len(main_py):,} bytes)")


if __name__ == "__main__":
    main()
