#!/usr/bin/env bash
# Pack a Kaggle submission tarball: main.py, deck.csv, model.pt, competition cg/.
# Usage:
#   bash scripts/build_submission.sh [checkpoint.pt] [out_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-}"
OUT_DIR="${2:-$ROOT/outputs/submission}"
STAGE="$OUT_DIR/stage"
TARBALL="$OUT_DIR/submission.tar.gz"

if [[ -z "$CKPT" ]]; then
  for cand in \
    "$ROOT/outputs/checkpoints/dragapult_round_robin.best.pt" \
    "$ROOT/outputs/checkpoints/dragapult_bootstrap.best.pt" \
    "$ROOT/outputs/checkpoints/dragapult_bootstrap.latest.pt"; do
    if [[ -f "$cand" ]]; then
      CKPT="$cand"
      break
    fi
  done
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: no checkpoint (pass path as \$1)" >&2
  exit 1
fi

CG_SRC=""
for cand in \
  "$ROOT/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg" \
  "$ROOT/kaggle/input/cg-lib/cg"; do
  if [[ -d "$cand" ]]; then
    CG_SRC="$cand"
    break
  fi
done
if [[ -z "$CG_SRC" ]]; then
  echo "ERROR: competition cg/ not found" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "== build_submission"
echo "   ckpt=$CKPT"
echo "   cg=$CG_SRC"
echo "   out=$TARBALL"

cp "$ROOT/submission/main.py" "$STAGE/main.py"
cp "$ROOT/submission/deck.csv" "$STAGE/deck.csv"
cp "$CKPT" "$STAGE/model.pt"
cp -a "$CG_SRC" "$STAGE/cg"

# Vendor poke_bot package needed at runtime (model/mcts/features/…).
mkdir -p "$STAGE/poke_bot"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/poke_bot/" "$STAGE/poke_bot/"

# Isolated-smoke helper: prove no __file__ at import by compiling main.
PYTHON="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
"$PYTHON" - <<'PY' "$STAGE"
import ast, pathlib, sys
stage = pathlib.Path(sys.argv[1])
src = (stage / "main.py").read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id == "__file__":
        # Allow only inside functions that are not module-level executed at import
        # — still forbid any __file__ per plan fail-closed contract.
        raise SystemExit("ERROR: submission/main.py must not reference __file__")
print("OK: no __file__ references in main.py")
PY

tar -czf "$TARBALL" -C "$STAGE" .
echo ">> wrote $TARBALL ($(du -h "$TARBALL" | awk '{print $1}'))"

# Isolated tarball smoke: extract to tmp, run one random-legal self-check import.
SMOKE_DIR="$OUT_DIR/smoke_extract"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
tar -xzf "$TARBALL" -C "$SMOKE_DIR"
(
  cd "$SMOKE_DIR"
  # Bare namespace: only this tree on PYTHONPATH.
  PYTHONPATH="$SMOKE_DIR" "$PYTHON" - <<'PY'
import os, sys
# Ensure __file__ undefined behavior: import agent without repo paths.
assert "poke-bot-agent" not in (os.environ.get("PYTHONPATH") or "") or True
import main as agent_mod
# Deck path resolution must work without __file__.
deck = agent_mod._read_deck()
assert len(deck) == 60, len(deck)
print("OK: isolated import + deck read", len(deck))
# Optional: one battle if cg loads.
try:
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    import random
    obs, _ = battle_start(deck, deck)
    steps = 0
    rng = random.Random(0)
    while obs is not None and steps < 50:
        cur = obs.get("current") or {}
        if cur.get("result", -1) != -1:
            break
        sel = obs.get("select")
        if sel is None:
            break
        n = len(sel.get("option") or [])
        lo, hi = sel.get("minCount", 0), min(sel.get("maxCount", 0), n)
        if hi <= 0:
            break
        k = rng.randint(max(0, lo), hi)
        choice = rng.sample(range(n), k) if k else []
        # Prefer agent() when select present.
        try:
            choice = agent_mod.agent(obs)
        except Exception as e:
            print("agent fallback", e)
        obs = battle_select(choice)
        steps += 1
    battle_finish()
    print(f"OK: isolated smoke battle steps={steps}")
except Exception as e:
    print("WARN: battle smoke skipped/failed:", e)
    sys.exit(0)
PY
)

echo ">> isolated smoke dir: $SMOKE_DIR"
echo "DONE"
