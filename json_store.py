"""
Simple JSON History Logger
Appends every claim + result to data/history.json
No dependencies — pure Python.
"""

import json
import os
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "data" / "history.json"


def _ensure_file():
    """Create the data directory and history file if they don't exist."""
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]")


def save_to_json(claim: str, result: dict):
    """Append a single claim + result to history.json."""
    _ensure_file()

    entry = {
        "id": _next_id(),
        "timestamp": datetime.now().isoformat(),
        "claim": claim,
        "verdict": result.get("final_verdict", "Uncertain"),
        "explanation": result.get("final_explanation", ""),
        "scores": {
            "evidence_confidence": result.get("evidence_confidence", 0),
            "source_score": result.get("source_score", 0),
            "critic_score": result.get("critic_score", 0),
            "composite": round(
                result.get("evidence_confidence", 0) * 0.35
                + result.get("source_score", 0) * 0.35
                + result.get("critic_score", 0) * 0.30
            ),
        },
        "source_reliability": result.get("source_reliability", ""),
        "sources": result.get("sources", []),
        "evidence": result.get("evidence", ""),
        "critic_challenges": result.get("critic_challenges", ""),
        "latency_ms": result.get("latency_ms", 0),
    }

    # Read → append → write
    history = json.loads(HISTORY_FILE.read_text())
    history.append(entry)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

    return entry


def load_history() -> list[dict]:
    """Load all saved claims from history.json."""
    _ensure_file()
    return json.loads(HISTORY_FILE.read_text())


def _next_id() -> int:
    """Auto-increment ID based on current history length."""
    _ensure_file()
    history = json.loads(HISTORY_FILE.read_text())
    return len(history) + 1