# Agent Skills compatibility

`update-durable-goal` follows the open Agent Skills specification. Its canonical
directory is:

```text
.agents/skills/update-durable-goal/
├── SKILL.md
└── agents/openai.yaml
```

The standard defines the skill directory and `SKILL.md` format, but individual
harnesses choose discovery locations.

## Codex

Codex discovers repository skills under `.agents/skills`, so the canonical
directory works directly. Invoke it explicitly as `$update-durable-goal` or use
a matching natural-language prompt.

## Prime Agent

Prime Agent also discovers project skills under `.agents/skills`, so the same
canonical directory works directly. Invoke it with
`/skill:update-durable-goal` when explicit selection is useful.

## Claude Code

Claude Code discovers project skills under `.claude/skills`. This repository
includes a relative symlink:

```text
.claude/skills/update-durable-goal
  -> ../../.agents/skills/update-durable-goal
```

Claude Code documents support for symlinked skill directories. Invoke the
skill as `/update-durable-goal` or let its description trigger automatically.

## Hermes Agent

Hermes stores installed skills under `~/.hermes/skills`. For a local checkout,
configure the canonical shared directory as an external skill directory:

```yaml
skills:
  external_dirs:
    - /absolute/path/to/durable-goals/.agents/skills
```

Alternatively, after the repository is published, install it through a Hermes
skill tap. The repository also exposes `skills/update-durable-goal` as a
symlink to the canonical directory for tap layouts that use the conventional
`skills/` subtree.

Hermes can modify writable external skills. Use read-only permissions if it
should consume but never self-edit the shared skill.

## Snowflake Cortex Code

Cortex Code discovers project skills in `.cortex/skills` and
`.claude/skills`. This repository includes a `.cortex/skills/update-durable-goal`
symlink to the same canonical standard skill, while the Claude compatibility
symlink provides a second supported discovery route. Invoke it as
`$update-durable-goal` and verify discovery with `/skill list` or `$$`.

After publication, Cortex Code can also install the conventional `skills/`
subtree from the Git repository with `/skill add`, or publish it through a
Snowflake stage or Snowflake Git repository. No Snowflake connection is needed
for the local goal-file operations themselves.

## Sources

- Agent Skills specification: <https://agentskills.io/specification>
- Codex skills: <https://developers.openai.com/codex/skills>
- Claude Code skills: <https://code.claude.com/docs/en/skills>
- Hermes skills: <https://hermes-agent.nousresearch.com/docs/user-guide/features/skills>
- Prime Agent skills: <https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/skills.md>
- Snowflake Cortex Code extensibility: <https://docs.snowflake.com/en/user-guide/cortex-code/extensibility>
