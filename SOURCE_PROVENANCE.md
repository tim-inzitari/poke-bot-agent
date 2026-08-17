# Source provenance

This repository was exported on 2026-08-17 from private development commit:

```text
9b3225215a3e06c156916c52218fb16667914e33
```

It uses a new Git history. The private development history was not copied.

## Excluded from the public source boundary

- model checkpoints, feature/replay shards, generated output, and vendored
  wheels or native binaries;
- private task goals, receipts, evidence, dashboards, runtime state, and fleet
  service definitions;
- local deployment staging and machine-specific editor/controller files;
- third-party competition data and competition-only engine patch source;
- bulk tournament deck-list datasets (small project-authored archetype examples remain);
- one-off `_tmp*` repair scripts;
- scripts that obtained third-party service credentials from a site's published
  client configuration rather than explicit user-supplied credentials.

## Release-time portability changes

- package metadata and public documentation were added;
- the test wrappers default to the active `python` executable instead of a
  private absolute interpreter path;
- private fleet roots, aliases, and SSH targets in the remote worker module were
  replaced by `POKEBOT_*` environment-driven defaults;
- generated and private artifacts are denied by `.gitignore` and CI checks.

These changes do not alter policy, feature, training-target, or simulator
semantics.
