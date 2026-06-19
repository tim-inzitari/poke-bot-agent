from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from features import combine_features, stable_hash_index
from model import TransformerRLModel


def features_from_observation(obs: dict[str, Any]) -> list[float]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    p0 = players[0] if len(players) > 0 else {}
    p1 = players[1] if len(players) > 1 else {}
    select = obs.get("select") or {}
    return [
        float(current.get("turn", 0)),
        float(current.get("yourIndex", 0)),
        float(p0.get("deckCount", 0)),
        float(p0.get("handCount", 0)),
        float(len(p0.get("bench", []))),
        float(p1.get("deckCount", 0)),
        float(p1.get("handCount", 0)),
        float(len(p1.get("bench", []))),
        float(len(select.get("option", []))),
        float(select.get("maxCount", 0)),
    ]


def legal_actions(option_count: int, min_count: int, max_count: int) -> list[list[int]]:
    """Enumerate legal action index lists in the same shape used during rollout logging."""
    actions: list[list[int]] = []
    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(range(option_count), count):
            if count <= 1:
                actions.append(list(combo))
            else:
                actions.extend(list(perm) for perm in itertools.permutations(combo))
    return actions


class TrainedPolicyAgent:
    def __init__(self, checkpoint_path: Path, *, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or "cpu")
        self._loaded = False
        self._history: list[np.ndarray] = []
        self._model: TransformerRLModel | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._window_size = 0
        self._policy_dim = 0
        self._state_hash_dim = 0

    def reset(self) -> None:
        self._history.clear()

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
        self._state_hash_dim = input_dim - 10

        self._model = TransformerRLModel(
            input_dim,
            policy_dim,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["heads"]),
            num_layers=int(model_cfg["layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            window_size=self._window_size,
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
        self._feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
        self._loaded = True

    def _encode_observation(self, obs_dict: dict[str, Any]) -> np.ndarray:
        assert self._feature_mean is not None and self._feature_std is not None
        features = combine_features(
            features_from_observation(obs_dict),
            obs_dict,
            None,
            state_hash_dim=self._state_hash_dim,
        )
        return ((features - self._feature_mean) / self._feature_std).astype(np.float32)

    def _model_logits(self) -> np.ndarray:
        assert self._model is not None
        assert self._feature_mean is not None
        window = self._history[-self._window_size :]
        pad_count = self._window_size - len(window)
        if pad_count > 0:
            pad = np.zeros_like(window[0]) if window else np.zeros(self._feature_mean.shape[-1], dtype=np.float32)
            window = [pad] * pad_count + window

        x = torch.tensor(np.stack(window), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones((1, self._window_size), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._model(x, mask)["policy_logits"].squeeze(0).cpu().numpy()
        return logits

    def choose_action(self, obs_dict: dict[str, Any]) -> list[int]:
        self._load()
        select = obs_dict.get("select") or {}
        options = select.get("option") or []
        min_count = int(select.get("minCount", 1))
        max_count = int(select.get("maxCount", 1))
        if not options:
            return []

        self._history.append(self._encode_observation(obs_dict))
        logits = self._model_logits()

        best_action: list[int] | None = None
        best_score = float("-inf")
        for action in legal_actions(len(options), min_count, max_count):
            action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
            action_class = stable_hash_index(action_key, self._policy_dim)
            score = float(logits[action_class])
            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else legal_actions(len(options), min_count, max_count)[0]


_AGENT: TrainedPolicyAgent | None = None


def default_checkpoint_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "value_model.pt",
        Path(os.environ.get("VALUE_MODEL_PATH", "")),
        Path("/kaggle_simulations/agent/value_model.pt"),
    ]
    for path in candidates:
        if str(path) and path.exists():
            return path
    return here / "value_model.pt"


def get_policy_agent() -> TrainedPolicyAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = TrainedPolicyAgent(default_checkpoint_path())
    return _AGENT
