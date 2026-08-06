# Model and Parameter Budget

## Recommendation

Use **`base_384`** as the first serious PokeRLM model:

- approximately **35.7M shared parameters total** if the encoder is built to this profile;
- approximately **14.5M new planning parameters** if attached to an already compatible state encoder and embedding stack;
- approximately **0.4M–0.9M additional parameters per specialist adapter**;
- recursion depth 2 with **shared weights**, so recursion does not multiply parameter count.

The architecture does not need billions of parameters. Pokémon TCG state and action representations are structured, the plan vocabulary is finite, CABT handles exact rules, and most performance should come from target quality, counterfactual coverage, process supervision, and deployment-aligned training rather than language-scale memorization.

## Profiles

The estimates below assume standard pre-norm transformer blocks, FFN multiplier 4, a finite structured embedding table, shared recursive planner weights, four Q bootstrap heads, 32 quantiles, five value horizons, a compact successor head, and four latent-dynamics blocks for base/strong.

| Component | `pilot_256` | **`base_384`** | `strong_512` |
|---|---:|---:|---:|
| Width / encoder layers | 256 / 8 | **384 / 10** | 512 / 12 |
| Structured embeddings | 2.05M | **3.46M** | 5.12M |
| State encoder | 6.33M | **17.76M** | 37.85M |
| Two-layer action decoder | 2.11M | **4.74M** | 8.41M |
| Shared recursive planner | 2.11M | **7.10M** | 12.62M |
| Latent dynamics | 0.79M | **2.37M** | 4.21M |
| Plan/value/output heads | 0.17M | **0.26M** | 0.34M |
| **Approx. total** | **13.55M** | **35.68M** | **68.55M** |
| **Approx. new attachment** | **5.18M** | **14.47M** | **25.58M** |

“New attachment” includes the action decoder, recursive planner, dynamics, and heads, but excludes the state encoder and base structured embeddings.

## Parameter-memory floor

This is parameter storage only; activations, optimizer workspace, batches, sequence lengths, and framework overhead dominate training memory.

| Profile | BF16 weights | FP32 weights | Approx. 16 bytes/parameter training state* |
|---|---:|---:|---:|
| `pilot_256` | 27 MB | 54 MB | 0.22 GB |
| `base_384` | 71 MB | 143 MB | 0.57 GB |
| `strong_512` | 137 MB | 274 MB | 1.10 GB |

\*A rough planning figure for weights/master weights, gradients, and Adam moments. Exact optimizer and mixed-precision implementations vary.

On a 48 GB RTX PRO 5000 Blackwell, all three profiles are small in parameter storage. The practical constraint will be activation memory, batch size, token counts, planner candidate count, and latency. Rough development envelopes with BF16 and sane batching are:

- `pilot_256`: often 4–10 GB total training VRAM;
- `base_384`: often 8–20 GB;
- `strong_512`: often 14–32 GB.

These are engineering ranges, not guarantees. Measure the actual repository graph.

## Why the base profile is the likely sweet spot

`pilot_256` is valuable for proving:

- observation/action parity;
- typed plan compilation;
- legality and hidden-information safety;
- deployment graph and cache behavior;
- latency instrumentation;
- training-target correctness.

It may underfit deck interactions and plan-value calibration at scale.

`base_384` provides roughly 2.6× the parameters of the pilot while remaining compact enough for batched candidate planning and multiple neural calls per turn. It is the recommended strength/latency compromise.

`strong_512` should be gated on evidence. Scale only when:

- train and validation losses both remain capacity-limited;
- teacher-ranking regret improves with width/depth in controlled scaling runs;
- data and target quality are not the bottleneck;
- p95 deployment latency retains margin;
- held-out nonlinear decks benefit more than simple decks regress.

## Specialist parameter budget

Prefer small adapters instead of complete model copies during development.

A bottleneck adapter with width `D/4` costs about `0.5 * D²` parameters per insertion point. With adapters in selected upper encoder, action-decoder, and planner blocks:

| Profile | Typical adapter range per specialist |
|---|---:|
| `pilot_256` | 0.2M–0.5M |
| `base_384` | **0.4M–0.9M** |
| `strong_512` | 0.7M–1.5M |

Keep shared card embeddings, state encoder, most action comparison, latent dynamics, belief modeling, and plan grammar common. Specialize tactical priorities, plan-code calibration, and upper-block feature transformations.

## Approximation formulas

For width `D` and FFN multiplier 4:

```text
encoder layer    ≈ 12 D² + O(D)
decoder layer    ≈ 16 D² + O(D)   # self-attn + cross-attn + FFN
dynamics block   ≈  4 D² + O(D)   # D -> 2D -> D residual MLP
embedding stack  = embedding_rows * D
```

Total planning attachment:

```text
2 * decoder_layer(D)             # current legal-action decoder
+ planner_layers * decoder_layer(D)
+ dynamics_blocks * dynamics_block(D)
+ plan embeddings and output heads
```

The recursive planner layers are counted once because the same module is applied at each depth.

## Exact count procedure after repository audit

1. Map the real state encoder width, layers, embeddings, and tied weights.
2. Instantiate every profile with the repository's actual modules.
3. Print:

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

4. Report counts by module and adapter.
5. Save counts in checkpoint metadata and deployment logs.
6. Compare against `tools/estimate_poke_rlm_params.py`; explain large discrepancies.

## Command examples

```bash
python tools/estimate_poke_rlm_params.py --profile pilot_256
python tools/estimate_poke_rlm_params.py --profile base_384
python tools/estimate_poke_rlm_params.py --profile strong_512 --json
```
