"""
Fake News Detector - LangGraph Agent Pipeline
"""

import os
import json
import re
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
#from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


# ── State ────────────────────────────────────────────────────────────────────

class DetectorState(TypedDict):
    claim: str
    evidence: str
    evidence_confidence: int          # 0-100
    sources: list[dict]               # [{name, url, credibility}]
    source_reliability: str           # Low / Medium / High
    source_score: int                 # 0-100
    critic_challenges: str
    critic_score: int                 # 0-100
    final_verdict: str                # Likely True / Uncertain / Likely Fake
    final_explanation: str
    agent_logs: list[str]             # real-time log lines

# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        #max_tokens=1024,
        #api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Extract first JSON object from model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

# ── Agent 1: Evidence Retrieval ───────────────────────────────────────────────

def evidence_retrieval_agent(state: DetectorState) -> DetectorState:
    llm = get_llm()
    system = SystemMessage(content=(
        "You are an Evidence Retrieval Agent. Given a claim, search your knowledge "
        "for relevant facts, statistics, and evidence that either support or contradict it. "
        "Be factual and neutral. "
        "Respond ONLY with valid JSON (no markdown, no extra text) with keys: "
        '"evidence" (string summary), "confidence" (integer 0-100).'
    ))
    human = HumanMessage(content=f"Claim: {state['claim']}")
    response = llm.invoke([system, human])
    data = _parse_json(response.content)

    logs = state.get("agent_logs", [])
    logs.append("🔍 Evidence Retrieval Agent: gathered supporting and contradicting evidence.")

    return {
        **state,
        "evidence": data.get("evidence", response.content),
        "evidence_confidence": int(data.get("confidence", 50)),
        "agent_logs": logs,
    }

# ── Agent 2: Source Verification ─────────────────────────────────────────────

def source_verification_agent(state: DetectorState) -> DetectorState:
    llm = get_llm()
    system = SystemMessage(content=(
        "You are a Source Verification Agent. Given a claim and evidence, identify "
        "the key sources or institutions that would verify or refute this claim. "
        "Assess source credibility based on reputation, bias, and reliability. "
        "Respond ONLY with valid JSON with keys: "
        '"sources" (list of objects with "name" and "credibility": High/Medium/Low), '
        '"reliability" ("High", "Medium", or "Low"), '
        '"score" (integer 0-100).'
    ))
    human = HumanMessage(content=f"Claim: {state['claim']}\n\nEvidence: {state['evidence']}")
    response = llm.invoke([system, human])
    data = _parse_json(response.content)

    logs = state.get("agent_logs", [])
    logs.append("🔎 Source Verification Agent: evaluated source credibility.")

    return {
        **state,
        "sources": data.get("sources", []),
        "source_reliability": data.get("reliability", "Medium"),
        "source_score": int(data.get("score", 50)),
        "agent_logs": logs,
    }

# ── Agent 3: Critic ───────────────────────────────────────────────────────────

def critic_agent(state: DetectorState) -> DetectorState:
    llm = get_llm()
    system = SystemMessage(content=(
        "You are a Critic Agent — a devil's advocate. Your job is to challenge "
        "the evidence and source findings, point out weaknesses, biases, missing context, "
        "logical fallacies, or alternative interpretations. Be rigorous. "
        "Respond ONLY with valid JSON with keys: "
        '"challenges" (string), "adjusted_confidence" (integer 0-100, how much you trust '
        "the evidence after your critique — lower means more suspicious)."
    ))
    human = HumanMessage(content=(
        f"Claim: {state['claim']}\n\n"
        f"Evidence: {state['evidence']}\n\n"
        f"Sources reliability: {state['source_reliability']}"
    ))
    response = llm.invoke([system, human])
    data = _parse_json(response.content)

    logs = state.get("agent_logs", [])
    logs.append("⚔️  Critic Agent: challenged findings and identified weaknesses.")

    return {
        **state,
        "critic_challenges": data.get("challenges", response.content),
        "critic_score": int(data.get("adjusted_confidence", 50)),
        "agent_logs": logs,
    }

# ── Agent 4: Final Verdict ────────────────────────────────────────────────────

def final_agent(state: DetectorState) -> DetectorState:
    llm = get_llm()

    avg_score = (
        state["evidence_confidence"] * 0.35
        + state["source_score"] * 0.35
        + state["critic_score"] * 0.30
    )

    system = SystemMessage(content=(
        "You are the Final Verdict Agent. Synthesise all findings and produce a clear, "
        "concise verdict. "
        "Respond ONLY with valid JSON with keys: "
        '"verdict" (exactly one of: "Likely True", "Uncertain", "Likely Fake"), '
        '"explanation" (2-3 sentences max).'
    ))
    human = HumanMessage(content=(
        f"Claim: {state['claim']}\n\n"
        f"Evidence: {state['evidence']}\n"
        f"Evidence confidence: {state['evidence_confidence']}%\n\n"
        f"Source reliability: {state['source_reliability']} ({state['source_score']}%)\n\n"
        f"Critic challenges: {state['critic_challenges']}\n"
        f"Critic adjusted confidence: {state['critic_score']}%\n\n"
        f"Composite score: {avg_score:.0f}%"
    ))
    response = llm.invoke([system, human])
    data = _parse_json(response.content)

    logs = state.get("agent_logs", [])
    logs.append("✅ Final Agent: verdict produced.")

    return {
        **state,
        "final_verdict": data.get("verdict", "Uncertain"),
        "final_explanation": data.get("explanation", response.content),
        "agent_logs": logs,
    }

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(DetectorState)

    graph.add_node("evidence_retrieval",  evidence_retrieval_agent)
    graph.add_node("source_verification", source_verification_agent)
    graph.add_node("critic",              critic_agent)
    graph.add_node("final",               final_agent)

    graph.set_entry_point("evidence_retrieval")
    graph.add_edge("evidence_retrieval",  "source_verification")
    graph.add_edge("source_verification", "critic")
    graph.add_edge("critic",              "final")
    graph.add_edge("final",               END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_detector(claim: str) -> DetectorState:
    """Run the full pipeline and return the final state."""
    app = build_graph()
    initial: DetectorState = {
        "claim": claim,
        "evidence": "",
        "evidence_confidence": 0,
        "sources": [],
        "source_reliability": "",
        "source_score": 0,
        "critic_challenges": "",
        "critic_score": 0,
        "final_verdict": "",
        "final_explanation": "",
        "agent_logs": [],
    }
    return app.invoke(initial)


def stream_detector(claim: str):
    """Yield (node_name, state) tuples as the graph executes."""
    app = build_graph()
    initial: DetectorState = {
        "claim": claim,
        "evidence": "",
        "evidence_confidence": 0,
        "sources": [],
        "source_reliability": "",
        "source_score": 0,
        "critic_challenges": "",
        "critic_score": 0,
        "final_verdict": "",
        "final_explanation": "",
        "agent_logs": [],
    }
    for event in app.stream(initial):
        for node, state in event.items():
            yield node, state
