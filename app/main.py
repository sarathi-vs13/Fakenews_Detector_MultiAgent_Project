"""
Fake News Detector — FastAPI Backend
Run: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import uuid
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import run_detector, stream_detector
from monitoring.logger import get_logger
from monitoring.metrics import (
    record_request, record_latency, record_verdict, get_metrics_summary,
)
from json_store import save_to_json, load_history
from database import init_db, save_to_db, save_feedback_to_db, load_all_claims, get_db_stats

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("startup", service="fakenews-detector", version="1.0.0")
    yield
    logger.info("shutdown", service="fakenews-detector")


app = FastAPI(
    title="Fake News Detector API",
    description="Multi-agent LangGraph pipeline for claim verification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    request.state.request_id = request_id
    logger.info("request_started", request_id=request_id,
                method=request.method, path=request.url.path)
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info("request_completed", request_id=request_id,
                status_code=response.status_code, latency_ms=latency_ms)
    response.headers["X-Request-ID"] = request_id
    return response


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=10, max_length=1000)

class FeedbackRequest(BaseModel):
    claim: str
    claim_id: int | None = None
    predicted_verdict: str
    correct_verdict: str
    notes: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "fakenews-detector", "version": "1.0.0"}


@app.post("/analyze")
def analyze(body: ClaimRequest, request: Request):
    """Run the full pipeline. Saves result to both JSON file and SQLite."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("analyze_started", request_id=request_id, claim_length=len(body.claim))

    start = time.time()
    try:
        result = run_detector(body.claim)
        latency_ms = round((time.time() - start) * 1000, 2)
        result["latency_ms"] = latency_ms

        verdict = result.get("final_verdict", "Uncertain")
        composite = round(
            result["evidence_confidence"] * 0.35
            + result["source_score"] * 0.35
            + result["critic_score"] * 0.30
        )

        json_entry = save_to_json(body.claim, result)
        db_id      = save_to_db(body.claim, result)

        record_request(success=True)
        record_latency(latency_ms)
        record_verdict(verdict)

        logger.info("analyze_completed", request_id=request_id,
                    verdict=verdict, composite_score=composite,
                    latency_ms=latency_ms, db_id=db_id)

        return {
            "request_id": request_id,
            "db_id": db_id,
            "json_id": json_entry["id"],
            "claim": body.claim,
            "verdict": verdict,
            "explanation": result.get("final_explanation", ""),
            "scores": {
                "evidence_confidence": result["evidence_confidence"],
                "source_score": result["source_score"],
                "critic_score": result["critic_score"],
                "composite": composite,
            },
            "source_reliability": result.get("source_reliability", ""),
            "sources": result.get("sources", []),
            "evidence": result.get("evidence", ""),
            "critic_challenges": result.get("critic_challenges", ""),
            "latency_ms": latency_ms,
        }

    except Exception as e:
        record_request(success=False)
        logger.error("analyze_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/analyze/stream")
def analyze_stream(body: ClaimRequest, request: Request):
    """Stream agent events as NDJSON. Saves final result to both stores."""
    request_id = getattr(request.state, "request_id", "unknown")

    def event_generator():
        start = time.time()
        final_state = None

        try:
            for node, state in stream_detector(body.claim):
                final_state = state
                event = {
                    "node": node,
                    "data": {
                        "agent_logs": state.get("agent_logs", []),
                        "evidence_confidence": state.get("evidence_confidence"),
                        "source_reliability": state.get("source_reliability"),
                        "critic_score": state.get("critic_score"),
                        "final_verdict": state.get("final_verdict"),
                    },
                }
                yield json.dumps(event) + "\n"

            if final_state:
                latency_ms = round((time.time() - start) * 1000, 2)
                final_state["latency_ms"] = latency_ms
                verdict = final_state.get("final_verdict", "Uncertain")

                save_to_json(body.claim, final_state)
                db_id = save_to_db(body.claim, final_state)

                record_request(success=True)
                record_latency(latency_ms)
                record_verdict(verdict)

                yield json.dumps({
                    "node": "done",
                    "data": {
                        "db_id": db_id,
                        "verdict": verdict,
                        "explanation": final_state.get("final_explanation", ""),
                        "scores": {
                            "evidence_confidence": final_state["evidence_confidence"],
                            "source_score": final_state["source_score"],
                            "critic_score": final_state["critic_score"],
                            "composite": round(
                                final_state["evidence_confidence"] * 0.35
                                + final_state["source_score"] * 0.35
                                + final_state["critic_score"] * 0.30
                            ),
                        },
                        "sources": final_state.get("sources", []),
                        "evidence": final_state.get("evidence", ""),
                        "critic_challenges": final_state.get("critic_challenges", ""),
                        "latency_ms": latency_ms,
                    },
                }) + "\n"

        except Exception as e:
            record_request(success=False)
            logger.error("stream_failed", request_id=request_id, error=str(e))
            yield json.dumps({"node": "error", "data": {"error": str(e)}}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    """Save human correction to SQLite feedback table."""
    db_id = save_feedback_to_db(
        claim_id=body.claim_id,
        predicted=body.predicted_verdict,
        correct=body.correct_verdict,
        notes=body.notes,
    )
    is_correct = body.predicted_verdict == body.correct_verdict
    logger.info("feedback_received", claim=body.claim[:80],
                predicted=body.predicted_verdict, correct=body.correct_verdict,
                match=is_correct, db_id=db_id)
    return {"status": "recorded", "feedback_id": db_id, "correct": is_correct}


@app.get("/history/json")
def history_json(limit: int = 50):
    """Return last N claims from the JSON file."""
    data = load_history()
    return {"source": "json_file", "count": len(data), "results": data[-limit:]}


@app.get("/history/db")
def history_db(limit: int = 50):
    """Return last N claims from SQLite."""
    data = load_all_claims(limit=limit)
    return {"source": "sqlite", "count": len(data), "results": data}


@app.get("/history/db/stats")
def db_stats():
    """Aggregate stats from SQLite — verdict distribution, avg scores, feedback accuracy."""
    return get_db_stats()


@app.get("/metrics")
def metrics():
    """In-process metrics — request counts, latency percentiles, verdict distribution."""
    return get_metrics_summary()