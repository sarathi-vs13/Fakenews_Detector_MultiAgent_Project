"""
Lightweight in-process metrics.
No external dependency — stores counts/latencies in memory.
Swap out for Prometheus if you want Grafana dashboards later.
"""

import threading
import time
from collections import defaultdict
from typing import Any

# ── Thread-safe store ─────────────────────────────────────────────────────────
_lock = threading.Lock()

_counters: dict[str, int] = defaultdict(int)
_latencies: list[float] = []          # all latencies in ms
_verdict_counts: dict[str, int] = defaultdict(int)
_start_time: float = time.time()


# ── Writers ───────────────────────────────────────────────────────────────────

def record_request(success: bool = True):
    with _lock:
        _counters["total_requests"] += 1
        if success:
            _counters["successful_requests"] += 1
        else:
            _counters["failed_requests"] += 1


def record_latency(latency_ms: float):
    with _lock:
        _latencies.append(latency_ms)
        # Keep last 1000 only to avoid unbounded memory
        if len(_latencies) > 1000:
            _latencies.pop(0)


def record_verdict(verdict: str):
    with _lock:
        _verdict_counts[verdict] += 1


# ── Readers ───────────────────────────────────────────────────────────────────

def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return round(sorted_data[min(index, len(sorted_data) - 1)], 2)


def get_metrics_summary() -> dict[str, Any]:
    with _lock:
        latency_snapshot = list(_latencies)
        counters_snapshot = dict(_counters)
        verdict_snapshot = dict(_verdict_counts)

    uptime_seconds = round(time.time() - _start_time)

    latency_stats = {
        "p50_ms": _percentile(latency_snapshot, 50),
        "p95_ms": _percentile(latency_snapshot, 95),
        "p99_ms": _percentile(latency_snapshot, 99),
        "avg_ms": round(sum(latency_snapshot) / len(latency_snapshot), 2) if latency_snapshot else 0,
        "sample_count": len(latency_snapshot),
    }

    total = counters_snapshot.get("total_requests", 0)
    error_rate = (
        round(counters_snapshot.get("failed_requests", 0) / total * 100, 1)
        if total > 0 else 0.0
    )

    return {
        "uptime_seconds": uptime_seconds,
        "requests": {
            **counters_snapshot,
            "error_rate_pct": error_rate,
        },
        "latency": latency_stats,
        "verdicts": verdict_snapshot,
    }