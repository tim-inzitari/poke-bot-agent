# Archetype Metagame Breakdown

Derived from the competitive decklist collection: **1,747 decks** across **10 Standard events**
(Korean League S4, NAIC New Orleans, Regionals: Campinas, Indianapolis, LA, Melbourne, Prague,
Utrecht; SE Lima, SE Turin — all 2026).

Archetype = the archetype slug encoded in each decklist filename
(`YYYY-MM_<event-slug>_<placement>_<archetype-slug>.csv`, per the collection spec).
Share = decks of that archetype / 1,747. 1% threshold ≈ 17.5 decks.

## Top 15 archetypes (≥1% metagame share)

| # | Archetype | Decks | Meta share |
|---|-----------|------:|-----------:|
| 1 | Dragapult | 222 | 12.71% |
| 2 | Dragapult / Dusknoir | 147 | 8.41% |
| 3 | Festival Lead | 109 | 6.24% |
| 4 | Alakazam / Dudunsparce | 105 | 6.01% |
| 5 | Raging Bolt / Ogerpon | 102 | 5.84% |
| 6 | Dragapult / Blaziken | 99 | 5.67% |
| 7 | Dragapult / Dudunsparce | 89 | 5.09% |
| 8 | Lopunny / Dudunsparce | 81 | 4.64% |
| 9 | N's Zoroark | 79 | 4.52% |
| 10 | Rockets' Mewtwo | 77 | 4.41% |
| 11 | Hydrapple | 75 | 4.29% |
| 12 | Cynthia's Garchomp | 68 | 3.89% |
| 13 | Slowking | 67 | 3.84% |
| 14 | Ogerpon Box | 60 | 3.43% |
| 15 | Lucario / Hariyama | 46 | 2.63% |

**Top 15 cumulative share: 81.6%** (1,426 / 1,747 decks).

## Remaining archetypes also above 1% (ranks 16–22)

| # | Archetype | Decks | Meta share |
|---|-----------|------:|-----------:|
| 16 | Ogerpon / Meganium | 45 | 2.58% |
| 17 | Rockets' Honchkrow | 38 | 2.18% |
| 18 | Starmie / Froslass | 35 | 2.00% |
| 19 | Crustle | 28 | 1.60% |
| 20 | Mega Lucario | 26 | 1.49% |
| 21 | Okidogi / Barbaracle | 18 | 1.03% |
| 22 | Clefairy / Ogerpon | 18 | 1.03% |

**Top 22 cumulative share (all archetypes ≥1%): 93.53%** (1,634 / 1,747 decks).
Everything below rank 22 (Greninja at 16 decks / 0.92% and down) falls under the 1% line —
the remaining ~6.5% of the field is a long tail of sub-1% archetypes.

## Notes

- **Dragapult is the defining engine of the format.** Counting all four Dragapult-tagged
  variants (Dragapult, /Dusknoir, /Blaziken, /Dudunsparce) gives **557 decks = 31.9%** of the
  field. They are kept split here because the request was for the *more specific* breakdown;
  flag if you'd rather collapse them into one "Dragapult" bucket for training.
- "Festival Lead" is the archetype slug as labeled in the source data (rank 3, 6.24%).
- Shares are pooled across all 10 events equally (raw deck counts), not weighted by event size
  or recency. Say the word if you want per-event weighting or a recency tilt toward NAIC/latest.
