# Distribution model

The protocol is language-neutral. Its portable assets are `GOAL.md`, JSON,
JSONL, SHA-256 references, and JSON Schemas. Python is the reference verifier
and safe writer, not a requirement imposed on every agent or platform.

## Why Python

Python keeps the reference CLI small, readable, and easy to install in coding
and ML environments. C++ would be appropriate only for embedding the resolver
inside a C++ application with no Python runtime. It would not make goal files
more portable.

Another implementation can conform by resolving the same schemas and producing
the same active contract, desired contract, workflow node states, topological
order, and next prompts. A future conformance corpus should make implementations
in TypeScript, Rust, Go, or C++ straightforward.

## Sharing layers

1. **Any coding agent:** commit `GOAL.md` and its canonical records. Add the
   one-line “read `GOAL.md` completely” rule to the harness's existing project
   instructions.
2. **Python environments:** publish `durable-goals` as a wheel so `dgoal`
   validates and updates packages deterministically.
3. **Skill-aware agents:** ship `.agents/skills/update-durable-goal` with the
   repository. It teaches the agent when and how to use the writer safely.
4. **Codex and ChatGPT distribution:** package the skill in a plugin when it is
   ready for installation beyond one repository.
5. **Other languages:** consume the schemas directly or implement the small
   resolver algorithm without Python.

OpenAI's current skill documentation describes skills as folders containing a
required `SKILL.md` plus optional scripts and resources, recognizes
repository-scoped skills under `.agents/skills`, and recommends plugins for
broader distribution:
<https://developers.openai.com/codex/skills>.

## Intended release artifacts

```text
durable-goals repository
├── Python wheel                 # verifier and safe writer
├── schemas/                     # language-neutral protocol
├── templates/GOAL.md            # authoritative entry point
├── workflow.json                # optional prompt-loop DAG
├── .agents/skills/              # prompt-driven update workflow
└── examples/                    # conformance examples
```

The wheel and skill are complementary. The skill supplies judgment and
procedure; the CLI performs fragile mutations and verification.

See [`SKILL_COMPATIBILITY.md`](SKILL_COMPATIBILITY.md) for Codex, Claude Code,
Hermes Agent, and Prime Agent discovery and invocation details.
