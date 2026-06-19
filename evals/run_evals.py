"""
Evaluation Pipeline for Fake News Detector
==========================================
Runs the pipeline over the golden dataset and scores it with an LLM judge.

Usage:
    python evals/run_evals.py                    # full eval
    python evals/run_evals.py --sample 5         # quick 5-claim smoke test
    python evals/run_evals.py --id 003           # single claim by ID
"""

import json
import time
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent dir so we can import agents
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import run_detector
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# ── LLM Judge ────────────────────────────────────────────────────────────────

def llm_judge(claim: str, predicted_verdict: str, expected_verdict: str,
              evidence: str, explanation: str) -> dict:
    """
    LLM-as-judge: independently evaluates reasoning quality.
    Returns scores for: verdict_correct, reasoning_quality, hallucination_risk
    """
    llm = ChatGroq(model="llama-3.1-8b-instant")

    system = SystemMessage(content=(
        "You are an impartial evaluation judge for an AI fact-checking system. "
        "Assess the quality of a verdict produced by the system. "
        "Respond ONLY with valid JSON, no markdown, with keys:\n"
        '  "verdict_match": true/false (does predicted match expected?),\n'
        '  "reasoning_quality": integer 1-5 (1=poor, 5=excellent),\n'
        '  "hallucination_risk": "Low"/"Medium"/"High" (did the AI make things up?),\n'
        '  "reasoning_notes": string (1-2 sentences on what was good or bad about the reasoning)'
    ))

    human = HumanMessage(content=(
        f"Claim: {claim}\n\n"
        f"Expected verdict: {expected_verdict}\n"
        f"Predicted verdict: {predicted_verdict}\n\n"
        f"Evidence provided: {evidence[:300]}...\n\n"
        f"Explanation: {explanation}"
    ))

    try:
        response = llm.invoke([system, human])
        import re
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        pass

    # Fallback: simple exact match
    return {
        "verdict_match": predicted_verdict == expected_verdict,
        "reasoning_quality": 3,
        "hallucination_risk": "Medium",
        "reasoning_notes": "Judge evaluation failed, fell back to exact match only.",
    }


# ── Eval Runner ───────────────────────────────────────────────────────────────

def run_evals(dataset: list[dict], verbose: bool = True) -> dict:
    results = []
    start_total = time.time()

    print(f"\n{'='*60}")
    print(f"  FAKE NEWS DETECTOR — EVALUATION RUN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(dataset)} claims")
    print(f"{'='*60}\n")

    for i, item in enumerate(dataset, 1):
        claim_id = item["id"]
        claim = item["claim"]
        expected = item["expected_verdict"]

        print(f"[{i}/{len(dataset)}] ID:{claim_id} — {claim[:60]}...")

        start = time.time()
        try:
            result = run_detector(claim)
            latency_ms = round((time.time() - start) * 1000)

            predicted = result.get("final_verdict", "Uncertain")
            exact_match = predicted == expected

            # LLM judge
            judge = llm_judge(
                claim=claim,
                predicted_verdict=predicted,
                expected_verdict=expected,
                evidence=result.get("evidence", ""),
                explanation=result.get("final_explanation", ""),
            )

            composite = round(
                result["evidence_confidence"] * 0.35
                + result["source_score"] * 0.35
                + result["critic_score"] * 0.30
            )

            row = {
                "id": claim_id,
                "claim": claim,
                "expected": expected,
                "predicted": predicted,
                "exact_match": exact_match,
                "composite_score": composite,
                "latency_ms": latency_ms,
                "reasoning_quality": judge.get("reasoning_quality", 3),
                "hallucination_risk": judge.get("hallucination_risk", "Medium"),
                "reasoning_notes": judge.get("reasoning_notes", ""),
                "category": item.get("category", ""),
                "error": None,
            }

            status = "✅" if exact_match else "❌"
            print(f"   {status} Predicted: {predicted:<15} Expected: {expected:<15} | {latency_ms}ms | Quality:{judge.get('reasoning_quality')}/5\n")

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000)
            row = {
                "id": claim_id, "claim": claim, "expected": expected,
                "predicted": "ERROR", "exact_match": False,
                "composite_score": 0, "latency_ms": latency_ms,
                "reasoning_quality": 0, "hallucination_risk": "High",
                "reasoning_notes": "", "category": item.get("category", ""),
                "error": str(e),
            }
            print(f"   💥 ERROR: {e}\n")

        results.append(row)
        time.sleep(0.5)  # rate limit courtesy

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    correct = sum(1 for r in results if r["exact_match"])
    accuracy = round(correct / total * 100, 1)
    avg_latency = round(sum(r["latency_ms"] for r in results) / total)
    avg_quality = round(sum(r["reasoning_quality"] for r in results) / total, 2)
    halluc_high = sum(1 for r in results if r["hallucination_risk"] == "High")

    # Per-category breakdown
    category_stats: dict = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if r["exact_match"]:
            category_stats[cat]["correct"] += 1

    summary = {
        "run_at": datetime.now().isoformat(),
        "total_claims": total,
        "accuracy_pct": accuracy,
        "avg_latency_ms": avg_latency,
        "avg_reasoning_quality": avg_quality,
        "high_hallucination_count": halluc_high,
        "total_duration_s": round(time.time() - start_total),
        "per_category": {
            cat: {
                "accuracy_pct": round(v["correct"] / v["total"] * 100, 1),
                **v
            }
            for cat, v in category_stats.items()
        },
        "results": results,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy:          {accuracy}% ({correct}/{total})")
    print(f"  Avg latency:       {avg_latency}ms")
    print(f"  Avg quality:       {avg_quality}/5")
    print(f"  High halluc risk:  {halluc_high}/{total}")
    print(f"\n  Per-category accuracy:")
    for cat, stats in summary["per_category"].items():
        print(f"    {cat:<30} {stats['accuracy_pct']}%")
    print(f"{'='*60}\n")

    return summary


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(summary: dict):
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"eval_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Report saved → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fake news detector evals")
    parser.add_argument("--sample", type=int, help="Run only N random claims")
    parser.add_argument("--id", type=str, help="Run a single claim by ID")
    parser.add_argument("--no-save", action="store_true", help="Don't save report")
    args = parser.parse_args()

    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    if args.id:
        dataset = [d for d in dataset if d["id"] == args.id]
        if not dataset:
            print(f"No claim found with ID {args.id}")
            sys.exit(1)
    elif args.sample:
        import random
        dataset = random.sample(dataset, min(args.sample, len(dataset)))

    summary = run_evals(dataset)

    if not args.no_save:
        save_report(summary)