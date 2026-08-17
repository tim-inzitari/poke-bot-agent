# Native expert featurizer (Bert development track)

This is an additive, fail-closed C++ acceleration track for the expert replay
pipeline. Production continues to use `authoritative_visual_trace.py`.

Milestones:

1. `replay_ingest_probe`: bounded-memory parallel ZIP decompression and JSON
   validation using libarchive + simdjson.
2. Emit canonical per-episode intermediate records and compare them with the
   Python converter on a fixed corpus.
3. Port feature packing/writing only after accepted/rejected episode sets,
   decision counts, target coverage, and output checksums reconcile.

No native output is eligible for training until the parity suite passes.

Build on Bert:

```bash
bash native/expert_featurizer/build_bert.sh
```

Run:

```bash
outputs/native/expert-featurizer/replay_ingest_probe ARCHIVE.zip 8
```
