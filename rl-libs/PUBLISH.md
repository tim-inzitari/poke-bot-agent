# Standalone repository status

`rl-libs` has **its own git history** (not mixed with poke-bot commits).

## On GitHub today

| Item | Location |
|---|---|
| Branch (library at repo root) | [`lib/rl-libs`](https://github.com/tim-inzitari/poke-bot-agent/tree/lib/rl-libs) |
| Tag | `v0.2.0` |
| Bundle artifact | `rl-libs-v0.2.0.bundle` |

Clone the library-only tree:

```bash
git clone --branch lib/rl-libs --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git rl-libs
cd rl-libs
git checkout -B main
```

## Promote to its own GitHub repo

Cloud agent tokens cannot `createRepository` under `tim-inzitari`. From a machine
where you are logged into GitHub:

```bash
bash scripts/publish_standalone_repo.sh tim-inzitari/rl-libs --private
```

Or manually:

```bash
git clone --branch lib/rl-libs --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git rl-libs
cd rl-libs && git checkout -B main
gh repo create tim-inzitari/rl-libs --private --source=. --remote=origin --push
git push origin v0.2.0
```

Production poke-bot trainers/selectors are **not** wired to this library.
