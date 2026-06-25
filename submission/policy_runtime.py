from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from features import (
    CARD_ID_SLOT_COUNT,
    COARSE_FEATURE_DIM,
    STRUCTURED_FEATURE_DIM,
    base_features_from_observation,
    combine_features,
    encode_observation_step,
    stable_hash_index,
)
from game_tracker import GameEventTracker
from model import TransformerRLModel

_BEAM_SEARCH_MODULE: Any | None = None


def _agent_dir() -> Path:
    kaggle = Path("/kaggle_simulations/agent")
    if (kaggle / "beam_search.py").is_file():
        return kaggle
    here = globals().get("__file__")
    if here:
        return Path(here).resolve().parent
    for candidate in (Path.cwd(), Path(".")):
        resolved = candidate.resolve()
        if (resolved / "beam_search.py").is_file():
            return resolved
    return Path.cwd()


def _load_beam_search() -> Any:
    """Load beam_search.py from the submission bundle (Kaggle cwd may omit agent dir)."""
    global _BEAM_SEARCH_MODULE
    if _BEAM_SEARCH_MODULE is not None:
        return _BEAM_SEARCH_MODULE

    agent_dir = _agent_dir()
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    beam_path = agent_dir / "beam_search.py"
    if not beam_path.is_file():
        raise ModuleNotFoundError(f"beam_search.py missing next to policy_runtime: {beam_path}")

    spec = importlib.util.spec_from_file_location("beam_search", beam_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"failed to load beam_search from {beam_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["beam_search"] = module
    spec.loader.exec_module(module)
    _BEAM_SEARCH_MODULE = module
    return module


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
        self._card_history: list[np.ndarray] = []
        self._tracker = GameEventTracker()
        self._model: TransformerRLModel | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._window_size = 0
        self._policy_dim = 0
        self._state_hash_dim = 0
        self._structured_feature_dim = STRUCTURED_FEATURE_DIM
        self._card_vocab_size = 2000

    def reset(self) -> None:
        self._history.clear()
        self._card_history.clear()
        self._tracker.reset()

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
        self._structured_feature_dim = int(checkpoint.get("coarse_feature_dim", STRUCTURED_FEATURE_DIM))
        self._state_hash_dim = input_dim - self._structured_feature_dim
        self._card_vocab_size = int(model_cfg.get("card_vocab_size", 2000))

        self._model = TransformerRLModel(
            input_dim,
            policy_dim,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["heads"]),
            num_layers=int(model_cfg["layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            window_size=self._window_size,
            card_vocab_size=self._card_vocab_size,
            card_embed_dim=int(model_cfg.get("card_embed_dim", 32)),
            card_slot_count=CARD_ID_SLOT_COUNT,
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32).reshape(-1)
        self._feature_std = np.array(checkpoint["feature_std"], dtype=np.float32).reshape(-1)
        self._loaded = True

    def _encode_observation(
        self,
        obs_dict: dict[str, Any],
        *,
        our_deck: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._feature_mean is not None and self._feature_std is not None
        if self._structured_feature_dim >= COARSE_FEATURE_DIM:
            features, card_ids = encode_observation_step(
                obs_dict,
                self._tracker,
                state_hash_dim=self._state_hash_dim,
                our_deck=our_deck,
                card_vocab_size=self._card_vocab_size,
            )
            features = features.reshape(-1)
        else:
            features, card_ids = combine_features(
                base_features_from_observation(obs_dict),
                obs_dict,
                None,
                state_hash_dim=self._state_hash_dim,
                our_deck=our_deck,
                card_vocab_size=self._card_vocab_size,
            )
            features = features.reshape(-1)
        normalized = ((features - self._feature_mean) / self._feature_std).astype(np.float32)
        return normalized, card_ids.astype(np.int64)

    def _choose_from_policy_logits(self, logits: np.ndarray, actions: list[list[int]]) -> list[int]:
        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            action_key = json.dumps(action, sort_keys=True, separators=(",", ":"))
            action_class = stable_hash_index(action_key, self._policy_dim)
            score = float(logits[action_class])
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _stack_model_window(
        self,
        *,
        leaf_features: np.ndarray | None = None,
        leaf_cards: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._model is not None
        assert self._feature_mean is not None
        history = list(self._history)
        card_history = list(self._card_history)
        if leaf_features is not None:
            history.append(leaf_features)
        if leaf_cards is not None:
            card_history.append(leaf_cards)

        window = history[-self._window_size :]
        card_window = card_history[-self._window_size :]
        pad_count = self._window_size - len(window)
        if pad_count > 0:
            pad = np.zeros_like(window[0]) if window else np.zeros(self._feature_mean.shape[0], dtype=np.float32)
            window = [pad] * pad_count + window
            card_pad = np.zeros(CARD_ID_SLOT_COUNT, dtype=np.int64)
            card_window = [card_pad] * pad_count + card_window

        x = torch.tensor(np.stack(window), dtype=torch.float32, device=self.device).unsqueeze(0)
        cards = torch.tensor(np.stack(card_window), dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.ones((1, self._window_size), dtype=torch.float32, device=self.device)
        return x, cards, mask

    def _model_logits(
        self,
        *,
        leaf_features: np.ndarray | None = None,
        leaf_cards: np.ndarray | None = None,
    ) -> np.ndarray:
        assert self._model is not None
        x, cards, mask = self._stack_model_window(leaf_features=leaf_features, leaf_cards=leaf_cards)
        with torch.no_grad():
            return self._model.forward_last(x, mask, card_ids=cards)["policy_logits"].squeeze(0).cpu().numpy()

    def _model_value(
        self,
        *,
        leaf_features: np.ndarray | None = None,
        leaf_cards: np.ndarray | None = None,
    ) -> float:
        assert self._model is not None
        x, cards, mask = self._stack_model_window(leaf_features=leaf_features, leaf_cards=leaf_cards)
        with torch.no_grad():
            return float(self._model.forward_last(x, mask, card_ids=cards)["value"].squeeze().cpu().item())

    def choose_action(self, obs_dict: dict[str, Any], *, our_deck: list[int] | None = None) -> list[int]:
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

        features, card_ids = self._encode_observation(obs_dict, our_deck=our_deck)
        self._history.append(features)
        self._card_history.append(card_ids)
        if self._window_size > 0 and len(self._history) > self._window_size:
            self._history = self._history[-self._window_size :]
            self._card_history = self._card_history[-self._window_size :]
        logits = self._model_logits()
        root_your_index = int((obs_dict.get("current") or {}).get("yourIndex", 0))

        use_beam = our_deck is not None and obs_dict.get("search_begin_input")
        if use_beam:
            beam = _load_beam_search()
            config = beam.BeamSearchConfig()
            if not beam.should_skip_beam_search(obs_dict, config):
                try:
                    return beam.run_beam_search(
                        self,
                        obs_dict,
                        our_deck,
                        actions,
                        root_your_index,
                        config,
                    )
                except Exception:
                    pass

        return self._choose_from_policy_logits(logits, actions)

    def choose_action_with_beam(self, obs_dict: dict[str, Any], our_deck: list[int]) -> list[int]:
        return self.choose_action(obs_dict, our_deck=our_deck)


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
