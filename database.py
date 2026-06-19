"""
SQLite Database Layer
Stores every claim + result in a proper relational database.
Zero setup — SQLite is built into Python.
Swap sqlite:///... for postgresql://... later with zero code changes.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "fakenews.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory so rows behave like dicts."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # rows accessible as row["column"]
    return conn


def init_db():
    """
    Create tables if they don't exist.
    Call this once at app startup.
    """
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            claim               TEXT NOT NULL,
            verdict             TEXT NOT NULL,
            explanation         TEXT,
            evidence_confidence INTEGER,
            source_score        INTEGER,
            critic_score        INTEGER,
            composite_score     INTEGER,
            source_reliability  TEXT,
            sources             TEXT,   -- JSON string
            evidence            TEXT,
            critic_challenges   TEXT,
            latency_ms          INTEGER
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            claim_id            INTEGER,
            predicted_verdict   TEXT NOT NULL,
            correct_verdict     TEXT NOT NULL,
            is_correct          INTEGER NOT NULL,  -- 0 or 1
            notes               TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        );
    """)
    conn.commit()
    conn.close()


def save_to_db(claim: str, result: dict) -> int:
    """
    Insert a claim + result into the database.
    Returns the new row's ID.
    """
    composite = round(
        result.get("evidence_confidence", 0) * 0.35
        + result.get("source_score", 0) * 0.35
        + result.get("critic_score", 0) * 0.30
    )

    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO claims (
            timestamp, claim, verdict, explanation,
            evidence_confidence, source_score, critic_score, composite_score,
            source_reliability, sources, evidence, critic_challenges, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            claim,
            result.get("final_verdict", "Uncertain"),
            result.get("final_explanation", ""),
            result.get("evidence_confidence", 0),
            result.get("source_score", 0),
            result.get("critic_score", 0),
            composite,
            result.get("source_reliability", ""),
            json.dumps(result.get("sources", [])),   # list → JSON string
            result.get("evidence", ""),
            result.get("critic_challenges", ""),
            result.get("latency_ms", 0),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def save_feedback_to_db(claim_id: Optional[int], predicted: str,
                         correct: str, notes: str = "") -> int:
    """Save human feedback linked to a claim."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO feedback (timestamp, claim_id, predicted_verdict, correct_verdict, is_correct, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            claim_id,
            predicted,
            correct,
            int(predicted == correct),
            notes,
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def load_all_claims(limit: int = 100) -> list[dict]:
    """Fetch the most recent N claims as a list of dicts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        r = dict(row)
        r["sources"] = json.loads(r["sources"] or "[]")  # JSON string → list
        results.append(r)
    return results


def get_db_stats() -> dict:
    """Quick stats summary from the database."""
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    verdict_rows = conn.execute(
        "SELECT verdict, COUNT(*) as count FROM claims GROUP BY verdict"
    ).fetchall()
    avg_latency = conn.execute(
        "SELECT AVG(latency_ms) FROM claims"
    ).fetchone()[0]
    avg_composite = conn.execute(
        "SELECT AVG(composite_score) FROM claims"
    ).fetchone()[0]
    feedback_total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    feedback_correct = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE is_correct = 1"
    ).fetchone()[0]

    conn.close()

    return {
        "total_claims": total,
        "verdict_distribution": {row["verdict"]: row["count"] for row in verdict_rows},
        "avg_latency_ms": round(avg_latency or 0, 1),
        "avg_composite_score": round(avg_composite or 0, 1),
        "feedback": {
            "total": feedback_total,
            "correct": feedback_correct,
            "accuracy_pct": round(feedback_correct / feedback_total * 100, 1)
            if feedback_total > 0 else None,
        },
    }