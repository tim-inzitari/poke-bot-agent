# Standalone repository status

`wave_dispatch` has **its own git history** (not mixed with poke-bot commits).

## On GitHub today

| Item | Location |
|---|---|
| Branch (library at repo root) | [`lib/wave-dispatch`](https://github.com/tim-inzitari/poke-bot-agent/tree/lib/wave-dispatch) |
| Tag | `v0.3.1` |
| Bundle artifact | `wave-dispatch-v0.3.1.bundle` |

Clone the library-only tree:

```bash
git clone --branch lib/wave-dispatch --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git wave-dispatch
cd wave-dispatch
git checkout -B main
```

## Promote to its own GitHub repo

Cloud agent tokens cannot `createRepository` under `tim-inzitari`. From a machine
where you are logged into GitHub:

```bash
bash wave-dispatch/scripts/publish_standalone_repo.sh tim-inzitari/wave-dispatch --private
```

Or manually:

```bash
git clone --branch lib/wave-dispatch --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git wave-dispatch
cd wave-dispatch && git checkout -B main
gh repo create tim-inzitari/wave-dispatch --private --source=. --remote=origin --push
git push origin v0.3.1
```

Then optionally replace the in-tree copy in poke-bot-agent with a submodule:

```bash
git submodule add https://github.com/tim-inzitari/wave-dispatch.git wave-dispatch
```
