from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_agent.actions import legal_actions
from poke_agent.attack_plan import score_actions_with_attack_plan
from poke_agent.features import COARSE_FEATURE_DIM, encode_observation_step, stable_hash_index
from poke_agent.game_tracker import GameEventTracker
from poke_agent.models.temporal_transformer import TemporalTransformer


@dataclass
class PolicySession:
    """Per-player game memory for one loaded checkpoint."""

    history: list[np.ndarray] = field(default_factory=list)
    tracker: GameEventTracker = field(default_factory=GameEventTracker)

    def reset(self) -> None:
        self.history.clear()
        self.tracker.reset()


class PolicyRuntime:
    """Load one checkpoint; run many seats via separate PolicySession objects."""

    def __init__(self, checkpoint_path: Path, *, device: torch.device | str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or "cpu")
        self._loaded = False
        self._model: TemporalTransformer | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._window_size = 0
        self._policy_dim = 0
        self._state_hash_dim = 0
        self._coarse_feature_dim = COARSE_FEATURE_DIM

    def new_session(self) -> PolicySession:
        return PolicySession()

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"missing trained model checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg = checkpoint["model_config"]
        input_dim = int(checkpoint["input_dim"])
        policy_dim = int(checkpoint["policy_dim"])
        self._window_size = int(model_cfg["window_size"])
        self._policy_dim = policy_dim
        self._coarse_feature_dim = int(checkpoint.get("coarse_feature_dim", COARSE_FEATURE_DIM))
        self._state_hash_dim = input_dim - self._coarse_feature_dim

        self._model = TemporalTransformer(
            input_dim,
            policy_dim,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["heads"]),
            num_layers=int(model_cfg["layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            window_size=self._window_size,
            kan_grid_size=int(model_cfg.get("kan_grid_size", 8)),
            use_kan=bool(model_cfg.get("use_kan", False)),
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32).reshape(-1)
        self._feature_std = np.array(checkpoint["feature_std"], dtype=np.float32).reshape(-1)
        self._loaded = True

    def _encode_observation(
        self,
        obs_dict: dict[str, Any],
        session: PolicySession,
        *,
        our_deck: list[int] | None = None,
    ) -> np.ndarray:
        assert self._feature_mean is not None and self._feature_std is not None
        features = encode_observation_step(
            obs_dict,
            session.tracker,
            state_hash_dim=self._state_hash_dim,
            our_deck=our_deck,
        ).reshape(-1)
        return ((features - self._feature_mean) / self._feature_std).astype(np.float32)

    def _model_logits(self, session: PolicySession) -> np.ndarray:
        assert self._model is not None
        assert self._feature_mean is not None
        window = session.history[-self._window_size :]
        pad_count = self._window_size - len(window)
        if pad_count > 0:
            pad = np.zeros_like(window[0]) if window else np.zeros(self._feature_mean.shape[0], dtype=np.float32)
            window = [pad] * pad_count + window

        x = torch.tensor(np.stack(window), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones((1, self._window_size), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._model(x, mask)["policy_logits"].squeeze(0).cpu().numpy()
        return logits

    def _choose_from_policy_logits(
        self,
        logits: np.ndarray,
        actions: list[list[int]],
        obs_dict: dict[str, Any] | None = None,
    ) -> list[int]:
        best_action = actions[0]
        best_score = float("-inf")
        plan_scores = score_actions_with_attack_plan(obs_dict, actions) if obs_dict is not None else [0.0 for _ in actions]
        max_abs_plan = max((abs(score) for score in plan_scores), default=0.0)
        for index, action in enumerate(actions):
            action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
            action_class = stable_hash_index(action_key, self._policy_dim)
            score = float(logits[action_class])
            if max_abs_plan > 0:
                score += 0.35 * (plan_scores[index] / max_abs_plan)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def choose_action(
        self,
        obs_dict: dict[str, Any],
        session: PolicySession,
        *,
        our_deck: list[int] | None = None,
        use_beam: bool = False,
        beam_config: Any | None = None,
    ) -> list[int]:
        self._load()
        select = obs_dict.get("select") or {}
        options = select.get("option") or []
        min_count = int(select.get("minCount", 1))
        max_count = int(select.get("maxCount", 1))
        if not options:
            return []

        actions = legal_actions(len(options), min_count, max_count)
        if not actions:
            return []

        session.history.append(self._encode_observation(obs_dict, session, our_deck=our_deck))
        logits = self._model_logits(session)
        root_your_index = int((obs_dict.get("current") or {}).get("yourIndex", 0))

        if use_beam and our_deck is not None and obs_dict.get("search_begin_input"):
            from poke_agent.beam_search import BeamSearchConfig, run_beam_search, should_skip_beam_search

            config = beam_config or BeamSearchConfig(sim_mode=True, time_budget_ms=150)
            if not should_skip_beam_search(obs_dict, config):
                try:
                    return run_beam_search(
                        self,
                        obs_dict,
                        session,
                        our_deck,
                        actions,
                        root_your_index,
                        config,
                    )
                except Exception:
                    pass

        return self._choose_from_policy_logits(logits, actions, obs_dict)


def make_policy_fn(
    runtime: PolicyRuntime,
    session: PolicySession,
    deck: list[int],
    *,
    use_beam: bool,
) -> Any:
    def agent(obs_dict: dict[str, Any]) -> list[int]:
        obs = dict(obs_dict)
        if use_beam:
            obs["remainingOverageTime"] = max(float(obs.get("remainingOverageTime") or 0), 9999.0)
        return runtime.choose_action(obs, session, our_deck=deck if use_beam else None, use_beam=use_beam)

    return agent
