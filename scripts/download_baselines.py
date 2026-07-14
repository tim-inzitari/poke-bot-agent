#!/usr/bin/env python3
"""Download baseline agents listed in baselines/manifest.json from Kaggle.

Idempotent: skips agents that already have main.py + deck.csv unless --force.
Resumable: continues after partial failures; prints a summary at the end.

Install layout (gitignored payloads)::

    baselines/official/<dir>/{main.py,deck.csv}
    baselines/community/<dir>/{main.py,deck.csv}
    baselines/roster/<dir>/{main.py,deck.csv}
    baselines/decks/<dir>/deck.csv
    baselines/kernels/<dir>/  (notebook + kernel-metadata when available)

Tracked files remain: baselines/README.md, baselines/manifest.json.

Requires a working Kaggle CLI (``kaggle`` on PATH, or ``.venv/bin/kaggle``).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "manifest.json"
BASELINES = ROOT / "baselines"

GROUP_DIR = {
    "official": BASELINES / "official",
    "community": BASELINES / "community",
    "roster": BASELINES / "roster",
}


def _find_kaggle() -> str:
    for cand in (
        ROOT / ".venv" / "bin" / "kaggle",
        Path(shutil.which("kaggle") or ""),
    ):
        if cand and cand.is_file():
            return str(cand)
    raise SystemExit("kaggle CLI not found (tried .venv/bin/kaggle and PATH)")


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _cell_src(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)


def _extract_writefile(cells: list[str], filename: str) -> str | None:
    pat = re.compile(
        rf"^(?:%%writefile|#\s*%%writefile)\s+{re.escape(filename)}\s*\n(.*)$",
        re.S | re.M,
    )
    for src in cells:
        m = pat.search(src)
        if m:
            return m.group(1).lstrip("\n")
    return None


def _expand_count_dict(full: str) -> list[int] | None:
    for var in ("DECK_COUNTS", "DECKLIST", "DECK"):
        m = re.search(rf"{var}\s*=\s*(\{{.*?\n\}})", full, re.S)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        deck: list[int] = []
        for cid, n in d.items():
            deck.extend([int(cid)] * int(n))
        if 55 <= len(deck) <= 65:
            return deck
    return None


def _extract_deck_list(cells: list[str]) -> list[int] | None:
    full = "\n\n".join(cells)
    counts = _expand_count_dict(full)
    if counts:
        return counts
    for m in re.finditer(
        r"(?:^|\n)(?:DECK|MY_DECK|deck_list|deck_ids|DECK_IDS)\s*=\s*\[(.*?)\]",
        full,
        re.S,
    ):
        nums = [int(x) for x in re.findall(r"\b\d+\b", m.group(1))]
        if 55 <= len(nums) <= 65:
            return nums
    deck_csv = _extract_writefile(cells, "deck.csv")
    if deck_csv:
        return _parse_deck_text(deck_csv)
    return None


def _parse_deck_text(text: str) -> list[int] | None:
    nums: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("id"):
            continue
        part = line.split(",")[0].strip() if "," in line else line.split()[0]
        if part.isdigit():
            nums.append(int(part))
    if len(nums) < 55:
        nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
    return nums if 55 <= len(nums) <= 65 else None


def _extract_main(cells: list[str]) -> str | None:
    main = _extract_writefile(cells, "main.py")
    if main and "def agent" in main:
        return main
    best = None
    for src in cells:
        if re.search(r"^\s*def\s+agent\s*\(", src, re.M) or (
            "from cg" in src and "def agent" in src
        ):
            cleaned = re.sub(r"^%%\w+.*\n", "", src)
            if best is None or len(cleaned) > len(best):
                best = cleaned
    return best if best and "def agent" in best else None


def _extract_payload(cells: list[str]) -> tuple[str | None, list[int] | None]:
    full = "\n".join(cells)
    m = re.search(r"AGENT_PAYLOADS\s*=\s*json\.loads\('((?:\\.|[^'\\])*)'\)", full)
    if not m:
        return None, None
    try:
        raw = m.group(1).encode("utf-8").decode("unicode_escape")
        payloads = json.loads(raw)
    except Exception:
        return None, None
    key = "A"
    dm = re.search(r'"default_selection":\s*"([^"]+)"', full)
    if dm and dm.group(1) in payloads:
        key = dm.group(1)
    elif key not in payloads:
        key = next(iter(payloads))
    p = payloads[key]
    main_py = p.get("main_py")
    deck_csv = p.get("deck_csv")
    deck = _parse_deck_text(deck_csv) if deck_csv else None
    if not deck and deck_csv:
        nums = [int(x) for x in deck_csv.split() if x.strip().isdigit()]
        if 55 <= len(nums) <= 65:
            deck = nums
    return main_py, deck


def _install_pair(
    agent: dict,
    main_src: str,
    deck_ids: list[int],
    nb_path: Path | None = None,
    meta_path: Path | None = None,
) -> Path:
    group = agent.get("group", "community")
    parent = GROUP_DIR.get(group, BASELINES / "community")
    out = parent / agent["dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text(main_src if main_src.endswith("\n") else main_src + "\n")
    deck_text = "\n".join(str(i) for i in deck_ids) + "\n"
    (out / "deck.csv").write_text(deck_text)

    deck_copy = BASELINES / "decks" / agent["dir"]
    deck_copy.mkdir(parents=True, exist_ok=True)
    (deck_copy / "deck.csv").write_text(deck_text)

    if nb_path and nb_path.is_file():
        kdir = BASELINES / "kernels" / agent["dir"]
        kdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb_path, kdir / nb_path.name)
        if meta_path and meta_path.is_file():
            shutil.copy2(meta_path, kdir / "kernel-metadata.json")
    return out


def _already_installed(agent: dict) -> bool:
    group = agent.get("group", "community")
    parent = GROUP_DIR.get(group, BASELINES / "community")
    out = parent / agent["dir"]
    return (out / "main.py").is_file() and (out / "deck.csv").is_file()


def _from_local_deck_source(agent: dict) -> list[int] | None:
    rel = agent.get("deck_source")
    if not rel:
        return None
    path = ROOT / rel
    if not path.is_file():
        return None
    nums: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("id"):
            continue
        part = line.split(",")[0].strip()
        if part.isdigit():
            nums.append(int(part))
    return nums if 55 <= len(nums) <= 65 else None


def _try_notebook(kaggle: str, agent: dict, work: Path) -> tuple[str | None, list[int] | None, Path | None, Path | None]:
    source = agent["source"]
    dest = work / "kernel"
    dest.mkdir(parents=True, exist_ok=True)
    r = _run([kaggle, "kernels", "pull", source, "-p", str(dest), "-m"])
    if r.returncode != 0:
        return None, None, None, None
    nbs = list(dest.glob("*.ipynb"))
    if not nbs:
        return None, None, None, None
    nb = json.loads(nbs[0].read_text())
    cells = [_cell_src(c) for c in nb.get("cells", [])]
    main = _extract_main(cells)
    deck = _extract_deck_list(cells)
    if (not main or not deck) and any("AGENT_PAYLOADS" in c for c in cells):
        pm, pd = _extract_payload(cells)
        main = main or pm
        deck = deck or pd
    meta = dest / "kernel-metadata.json"
    return main, deck, nbs[0], meta if meta.is_file() else None


def _try_kernel_output(kaggle: str, agent: dict, work: Path) -> tuple[str | None, list[int] | None]:
    source = agent["source"]
    dest = work / "output"
    dest.mkdir(parents=True, exist_ok=True)
    r = _run([kaggle, "kernels", "output", source, "-p", str(dest)])
    if r.returncode != 0:
        return None, None

    mains = [p for p in dest.rglob("main.py") if "cg" not in p.parts]
    decks = [p for p in dest.rglob("deck.csv") if "cg" not in p.parts]
    if mains and decks:
        return mains[0].read_text(), _parse_deck_text(decks[0].read_text())

    for tar_path in dest.rglob("submission.tar.gz"):
        extract = work / "tar"
        extract.mkdir(exist_ok=True)
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                for m in tar.getmembers():
                    name = Path(m.name).name
                    if name in {"main.py", "deck.csv"} and m.isfile():
                        f = tar.extractfile(m)
                        if f is not None:
                            (extract / name).write_bytes(f.read())
        except Exception:
            continue
        if (extract / "main.py").is_file() and (extract / "deck.csv").is_file():
            return (
                (extract / "main.py").read_text(),
                _parse_deck_text((extract / "deck.csv").read_text()),
            )
    return None, None


def _try_deck_dataset(kaggle: str, agent: dict, work: Path) -> list[int] | None:
    ds = agent.get("deck_dataset")
    if not ds:
        return None
    dest = work / "dataset"
    dest.mkdir(parents=True, exist_ok=True)
    r = _run([kaggle, "datasets", "download", "-d", ds, "-p", str(dest), "--unzip"])
    if r.returncode != 0:
        return None
    for p in dest.rglob("*.csv"):
        deck = _parse_deck_text(p.read_text())
        if deck:
            return deck
    return None


def download_one(kaggle: str, agent: dict, force: bool) -> str:
    """Return status: ok | skip | fail:<reason>."""
    if _already_installed(agent) and not force:
        return "skip"

    with tempfile.TemporaryDirectory(prefix=f"bl-{agent['id']}-") as tmp:
        work = Path(tmp)
        main, deck, nb, meta = _try_notebook(kaggle, agent, work)

        local_deck = _from_local_deck_source(agent)
        if local_deck and not deck:
            deck = local_deck

        if not deck:
            deck = _try_deck_dataset(kaggle, agent, work / "ds")

        if not main or not deck:
            om, od = _try_kernel_output(kaggle, agent, work / "out")
            main = main or om
            deck = deck or od

        if not main or "def agent" not in main:
            return "fail:no_main"
        if not deck:
            return "fail:no_deck"

        # Normalize length quirks (some lists are 59/61); keep if near 60.
        if not (55 <= len(deck) <= 65):
            return f"fail:bad_deck_len={len(deck)}"

        _install_pair(agent, main, deck, nb, meta)
        return "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Re-download even if already present")
    ap.add_argument("--group", choices=["official", "community", "roster"], help="Only this group")
    ap.add_argument("--only", nargs="+", help="Only these agent ids")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: missing manifest {args.manifest}", file=sys.stderr)
        return 1

    data = json.loads(args.manifest.read_text())
    agents = data.get("agents", [])
    # Never re-download agents that a prior cleanup pass flagged as broken.
    excluded = set(data.get("excluded_broken", []))
    if excluded:
        before = len(agents)
        agents = [a for a in agents if a["id"] not in excluded]
        if len(agents) < before:
            print(
                f"excluded_broken: skipping {before - len(agents)} agent(s): "
                f"{', '.join(sorted(excluded))}"
            )
    if args.group:
        agents = [a for a in agents if a.get("group") == args.group]
    if args.only:
        wanted = set(args.only)
        agents = [a for a in agents if a["id"] in wanted]

    kaggle = _find_kaggle()
    print(f"kaggle={kaggle}")
    print(f"manifest={args.manifest} agents={len(agents)}")

    from tqdm.auto import tqdm

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[tuple[str, str]] = []

    for i, agent in enumerate(tqdm(agents, desc="baselines", unit="agent"), start=1):
        status = download_one(kaggle, agent, force=args.force)
        tag = status.split(":", 1)[0]
        counts[tag] = counts.get(tag, 0) + 1
        tqdm.write(
            f"  [{i}/{len(agents)}] [{status:12}] {agent['id']}  "
            f"({agent.get('group')})  {agent['source']}"
        )
        if status.startswith("fail"):
            failures.append((agent["id"], status))

    print("\nSummary:", counts)
    if failures:
        print("Failures:")
        for aid, st in failures:
            print(f"  {aid}: {st}")
        notes = data.get("field_notes", {}).get("inaccessible_403", [])
        if notes:
            print("Known inaccessible (403) sources in manifest field_notes:")
            for s in notes:
                print(f"  - {s}")
        return 1 if counts["ok"] == 0 and counts["skip"] == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
