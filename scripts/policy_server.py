#!/usr/bin/env python3
"""Serve the neural policy from the Mac while Linux/Docker runs CABT games."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.deck import read_deck
from poke_agent.config import build_config
from poke_agent.device import torch_device
from poke_agent.policy_agent import PolicyRuntime


class PolicyServer:
    def __init__(self, checkpoint: Path, *, deck_path: Path | None = None, use_beam: bool = False):
        self.checkpoint = checkpoint
        self.device = torch_device()
        self.runtime = PolicyRuntime(checkpoint, device=self.device)
        overrides: dict[str, object] = {"model_output_path": str(checkpoint)}
        if deck_path is not None:
            overrides["agent_deck_path"] = str(deck_path)
        config = build_config(ROOT, overrides=overrides)
        self.deck, self.deck_path = read_deck(config, ROOT)
        self.use_beam = use_beam
        self.sessions: dict[str, Any] = {}

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self.sessions.clear()
            return
        self.sessions.pop(session_id, None)

    def act(self, payload: dict[str, Any]) -> list[int]:
        session_id = str(payload.get("session_id") or "default")
        obs = payload["observation"]
        session = self.sessions.get(session_id)
        if session is None:
            session = self.runtime.new_session()
            self.sessions[session_id] = session
        return self.runtime.choose_action(
            obs,
            session,
            our_deck=self.deck,
            use_beam=self.use_beam,
        )


def make_handler(policy: PolicyServer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "ok": True,
                        "device": str(policy.device),
                        "deck": str(policy.deck_path),
                        "checkpoint": str(policy.checkpoint),
                    },
                )
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == "/reset":
                    policy.reset(payload.get("session_id"))
                    self._send(200, {"ok": True})
                    return
                if self.path == "/act":
                    action = policy.act(payload)
                    self._send(200, {"action": action})
                    return
                self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac neural policy HTTP server for Docker CABT rollouts.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/temporal_current.pt")
    parser.add_argument("--deck", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--beam", action="store_true", help="enable beam; requires CABT search API on this host")
    args = parser.parse_args()

    policy = PolicyServer(args.checkpoint, deck_path=args.deck, use_beam=args.beam)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"policy server: http://{args.host}:{args.port}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {policy.device}")
    print(f"deck: {policy.deck_path} ({len(policy.deck)} cards)")
    print("ready", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
