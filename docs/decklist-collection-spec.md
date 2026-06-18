# Competitive Decklist Collection — Spec & Runbook

**Task owner:** Matis (assisting tim-inzitari)
**Branch:** `data/card-pool-and-decklists`
**Status:** approved design; ready to execute via an autonomous `/goal` session.

## 1. Objective

Build a training dataset of real competitive Pokémon TCG decklists, converted into the
exact format the competition simulator consumes, restricted to the competition's legal
card pool, and split into two performance tiers.

## 2. Output format (non-negotiable)

- **One deck = one file.**
- **Each file has exactly 60 lines.**
- **Each line is a single numeric Card ID** (the simulator's ID — see the card master in §3).
- **Duplicates are repeated**: 4 copies of a card = that ID on 4 separate lines.
- This is byte-identical to the existing `decks/submission.csv` and to what
  `submission/main.py` (`read_deck_csv`) reads. No headers, no names, no blank lines.

## 3. Card master (source of truth for legality + IDs)

- File: `cards/EN_Card_Data.csv` (UTF-8 with BOM; read with `encoding="utf-8-sig"`).
- 2,022 rows → **1,267 unique Card IDs (1–1267)**. There are multiple rows per card
  (one per attack/move); dedupe on `Card ID`.
- Columns we use: `Card ID`, `Card Name`, `Expansion`, `Collection No.`, `Category`.
- **A Card ID is legal iff it appears in this file.** Nothing else is in the pool.
- Note: card *names are not unique* across IDs (e.g. "Charcadet" = 204/319/796/935).
  The simulator keys on the numeric ID, which is why output is IDs, not names.
- Note: the `Stage…/Type` and effect-text columns contain a mojibake `Pok�mon`; ignore
  it. The `Card Name`, `Expansion`, and `Collection No.` columns are clean.

## 4. Resolver (name/set/number → Card ID)

Build these in-memory indexes from the master:
1. **Primary key — `(Expansion, Collection No.) → Card ID`.** This is the reliable,
   unambiguous mapping. Limitless exports use the same set-abbreviation + number scheme,
   so most cards resolve here directly.
2. **Fallback — normalized name → {Card IDs}.** Normalize by lowercasing, trimming,
   collapsing whitespace, and unifying apostrophes/quotes (`’` → `'`) and accents. Use
   only when (set, number) fails AND the normalized name maps to exactly one Card ID.
3. **Basic Energy** (`Card ID` 1–8, Category `Basic Energy`): resolve by energy type;
   these are exempt from the 4-copy rule.

If a card line cannot be resolved to exactly one in-pool Card ID, the **whole deck is
rejected** (see §6). Never guess, never substitute a different printing, never pad.

## 5. Source

- **Limitless TCG** (`limitlesstcg.com`) only, for this first pass.
- Target the most **recent Standard-format** events (the pool maps to ~mid-2026 Standard,
  sets through MEG/PFL/ASC/POR). Older lists use rotated cards absent from the pool and
  will mostly be rejected — that's expected.
- Decklists are available as fetchable text (PTCGL export: `qty name SET number`). If a
  specific page genuinely cannot be fetched, **log it and move on — do not fabricate a
  list**. Honest coverage beats invented data.

## 6. Validation & rejection

A deck is written only if ALL hold:
- exactly 60 cards total,
- every card resolves to an in-pool Card ID,
- ≤ 4 copies of any card **by name** (basic Energy exempt).

Any deck failing validation (including any unresolved/out-of-pool card) is **skipped** and
appended to `decks/competitive/rejected.log` with: source URL, event, placement, and the
specific reason (e.g. `unresolved: "Iono" PAL 185 not in pool`). This log is how we
measure pool coverage honestly.

## 7. Deduplication

Two decks with the identical 60-ID multiset are the same deck. Keep one (the
best-placing / largest-event instance) and record collisions in `rejected.log` as
`duplicate of <filename>`.

## 8. Performance tiering (which subfolder)

A deck goes in **`high_performing/`** if ANY of:
- it **won or was runner-up** of *any* event, OR
- big event (≥256 players): finished **top 32** (~top 12%), OR
- mid event (64–255 players): finished **top 8**, OR
- small event (<64 players): **winner / finalist only**.

Everything else valid → **`the_rest/`**. (Cutoffs are a deliberate heuristic and easy to
re-tune later.)

## 9. Output layout

```
decks/competitive/
  high_performing/        # 60-line ID files, strong finishes
  the_rest/               # 60-line ID files, all other valid legal lists
  index.csv               # provenance, one row per written deck
  rejected.log            # skipped/duplicate decks + reasons
  progress.json           # processed event/deck URLs, for resumability
```
- **Filename:** `YYYY-MM_<event-slug>_<placement>_<archetype-slug>.csv`
  (e.g. `2026-05_road-to-lisbon_top4_charizard-ex.csv`). Keep slugs ascii/kebab-case.
- **`index.csv` columns:** `filename, tier, event, event_date, field_size, placement,
  archetype, source_url`.
- **`progress.json`:** list of already-processed event + deck URLs so a re-run resumes
  instead of re-fetching.

## 10. Working constraints

- Work only on branch `data/card-pool-and-decklists`. **Do not touch `main`.**
- Commit in batches with clear messages (e.g. `add 40 high_performing decks from <event>`).
- **Do not `git push` and do not open a PR** — Matis/inzi handle review and integration.
- Be polite to Limitless (reasonable request pacing). Log every source URL touched.

## 11. Done condition (verifiable)

`decks/competitive/high_performing/` and `decks/competitive/the_rest/` are populated with
every distinct, fully-resolvable current-Standard competitive decklist obtainable from
Limitless TCG, each a valid 60-line Card-ID file (all IDs in `cards/EN_Card_Data.csv`,
≤4 per card by name, exactly 60 lines), correctly tiered per §8, with `index.csv`,
`rejected.log`, and `progress.json` complete and consistent (every written file has an
`index.csv` row; counts reconcile).
