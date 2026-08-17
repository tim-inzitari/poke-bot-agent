# References and Design Basis

## Recursive planning inspiration

- Zhang, Kraska, and Khattab, **Recursive Language Models**, arXiv:2512.24601. RLMs externalize a problem, decompose it, and recursively invoke a shared model. PokeRLM borrows decomposition, working memory, and adaptive computation—not the free-form text REPL.
- Roy et al., **The Y-Combinator for LLMs: Solving Long-Context Rot with λ-Calculus**, arXiv:2603.20105. Supports typed, bounded recursive runtimes with explicit termination and cost controls.
- Wang, **Think, But Don't Overthink: Reproducing Recursive Language Models**, arXiv:2603.02615. Motivates explicit depth and compute controls because deeper recursion can degrade quality and sharply increase latency.

## Learned planning and value modeling

- Ruoss et al., **Amortized Planning with Large-Scale Transformers / Grandmaster-Level Chess Without Search**, arXiv:2402.04494.
- Dabney et al., **Distributional Reinforcement Learning with Quantile Regression**, arXiv:1710.10044.
- Oh, Singh, and Lee, **Value Prediction Network**, arXiv:1707.03497.
- Farquhar et al., **TreeQN and ATreeC**, arXiv:1710.11417.
- Schrittwieser et al., **MuZero**, arXiv:1911.08265.

## Cursor integration

- Cursor official documentation: project rules under `.cursor/rules`, MDC rule metadata, and root `AGENTS.md` support.
  - https://docs.cursor.com/context/rules
  - https://docs.cursor.com/en/cli/using

## Existing project documents to preserve

- `RL_TRAINING_PROTOCOL.md`
- `config/rl_protocol.yaml`
- `state/specialists.yaml`
- `Pokemon_TCG_Deck_Agnostic_Amortized_Search_Full_Plan` or repository-equivalent design notes

References establish architectural feasibility. They do not prove the Pokémon-specific parameter counts, data volumes, or win-rate gains; those must be validated by the experiments in this kit.
