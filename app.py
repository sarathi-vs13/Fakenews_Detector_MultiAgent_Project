"""
Fake News Detector — Streamlit UI
Calls FastAPI backend instead of agents.py directly.
Make sure FastAPI is running: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Run: streamlit run app.py
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f0f0f; color: #e8e4dc; }

.fd-header { text-align: center; padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #2a2a2a; margin-bottom: 2rem; }
.fd-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 600; letter-spacing: -0.02em; color: #e8e4dc; margin: 0; }
.fd-header .tagline { font-size: 0.85rem; color: #666; margin-top: 0.4rem; letter-spacing: 0.12em; text-transform: uppercase; }

.stTextArea textarea { background: #1a1a1a !important; border: 1px solid #333 !important; border-radius: 4px !important; color: #e8e4dc !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.95rem !important; resize: vertical !important; }
.stTextArea textarea:focus { border-color: #c0392b !important; box-shadow: 0 0 0 2px rgba(192,57,43,0.15) !important; }

.stButton > button { background: #c0392b !important; color: #fff !important; border: none !important; border-radius: 3px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important; padding: 0.65rem 2rem !important; text-transform: uppercase !important; transition: background 0.15s !important; width: 100% !important; }
.stButton > button:hover { background: #a93226 !important; }

.fd-log { background: #111; border: 1px solid #222; border-radius: 4px; padding: 1rem 1.2rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #888; line-height: 1.7; min-height: 80px; }
.fd-log .log-line { color: #aaa; }

.fd-card { background: #161616; border: 1px solid #2a2a2a; border-radius: 6px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.fd-card .label { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #555; margin-bottom: 0.5rem; }
.fd-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #e8e4dc; line-height: 1; }
.fd-card .sub { font-size: 0.8rem; color: #555; margin-top: 0.4rem; }

.fd-verdict { border-radius: 6px; padding: 1.8rem 2rem; margin: 1.5rem 0; text-align: center; }
.fd-verdict.true  { background: #0d2b1a; border: 1px solid #1a5c35; }
.fd-verdict.fake  { background: #2b0d0d; border: 1px solid #5c1a1a; }
.fd-verdict.uncertain { background: #1a1a0d; border: 1px solid #4a4a1a; }
.fd-verdict .vt { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.5rem; opacity: 0.6; }
.fd-verdict .vv { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; }
.fd-verdict.true  .vv { color: #2ecc71; }
.fd-verdict.fake  .vv { color: #e74c3c; }
.fd-verdict.uncertain .vv { color: #f1c40f; }
.fd-verdict .ve { font-size: 0.9rem; color: #aaa; margin-top: 0.8rem; max-width: 540px; margin-left: auto; margin-right: auto; line-height: 1.5; }

.fd-section { background: #141414; border-left: 3px solid #2a2a2a; border-radius: 0 4px 4px 0; padding: 1rem 1.3rem; margin-bottom: 1rem; font-size: 0.88rem; color: #aaa; line-height: 1.65; }
.fd-section .sh { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: #444; margin-bottom: 0.5rem; }

.fd-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.fd-pill { border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; padding: 0.25rem 0.65rem; border: 1px solid; }
.fd-pill.High   { border-color: #1a5c35; color: #2ecc71; background: #0d1f13; }
.fd-pill.Medium { border-color: #4a4a1a; color: #f1c40f; background: #1a1a0d; }
.fd-pill.Low    { border-color: #5c1a1a; color: #e74c3c; background: #1f0d0d; }

.fd-saved { background: #0d1f13; border: 1px solid #1a5c35; border-radius: 4px; padding: 0.5rem 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #2ecc71; margin-top: 0.5rem; }

hr { border-color: #1e1e1e !important; }
section[data-testid="stSidebar"] { background: #0c0c0c !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="fd-header">
    <h1>🔬 FAKE NEWS DETECTOR</h1>
    <div class="tagline">Multi-agent analysis using LangGraph</div>
</div>
""", unsafe_allow_html=True)

# ── API health check ──────────────────────────────────────────────────────────

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

if not check_api():
    st.error("⚠️ FastAPI backend is not running. Start it with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    st.stop()

# ── Example claims ────────────────────────────────────────────────────────────

EXAMPLES = [
    "NASA confirmed the Moon landing was staged in a Hollywood studio.",
    "Drinking bleach can cure COVID-19 according to WHO.",
    "The Eiffel Tower was built in 1889 for the World's Fair.",
    "Vaccines cause autism according to a 1998 Lancet study.",
]

# ── Input area ────────────────────────────────────────────────────────────────

col_input, col_side = st.columns([3, 1], gap="large")

with col_input:
    claim = st.text_area(
        "Enter a claim to analyse",
        placeholder="e.g. 5G towers spread the coronavirus…",
        height=110,
        label_visibility="collapsed",
    )
    run_btn = st.button("ANALYSE CLAIM →")

with col_side:
    st.markdown("<p style='font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase;color:#444;margin-bottom:.6rem'>Quick examples</p>", unsafe_allow_html=True)
    for ex in EXAMPLES:
        if st.button(ex[:55] + "…" if len(ex) > 55 else ex, key=ex):
            claim = ex
            run_btn = True

# ── Analysis ──────────────────────────────────────────────────────────────────

if run_btn:
    if not claim.strip():
        st.warning("Please enter a claim before analysing.")
        st.stop()

    st.markdown("---")

    log_box = st.empty()
    log_lines: list[str] = []

    def refresh_log():
        lines_html = "".join(f'<div class="log-line">{l}</div>' for l in log_lines)
        log_box.markdown(f'<div class="fd-log">{lines_html}</div>', unsafe_allow_html=True)

    log_lines.append("▶ Sending claim to pipeline…")
    refresh_log()

    try:
        # ── Call FastAPI /analyze ─────────────────────────────────────────────
        log_lines.append("🔍 <b style='color:#e8e4dc'>Evidence Retrieval Agent</b> — Gathering facts…")
        refresh_log()

        response = requests.post(
            f"{API_URL}/analyze",
            json={"claim": claim},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        # Simulate agent log progression
        agent_steps = [
            "🔎 <b style='color:#e8e4dc'>Source Verification Agent</b> — Evaluating credibility…",
            "⚔️  <b style='color:#e8e4dc'>Critic Agent</b> — Challenging findings…",
            "✅ <b style='color:#e8e4dc'>Final Verdict Agent</b> — Synthesising verdict…",
        ]
        for step in agent_steps:
            time.sleep(0.3)
            log_lines.append(step)
            refresh_log()

        log_lines.append("⬛ Analysis complete.")
        refresh_log()

        # ── Show saved confirmation ───────────────────────────────────────────
        db_id   = result.get("db_id")
        json_id = result.get("json_id")
        if db_id:
            log_lines.append(f"💾 Saved — SQLite ID: {db_id} · JSON ID: {json_id}")
            refresh_log()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Is it running on port 8000?")
        st.stop()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    st.markdown("---")

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict  = result.get("verdict", "Uncertain")
    css_cls  = {"Likely True": "true", "Likely Fake": "fake"}.get(verdict, "uncertain")
    explanation = result.get("explanation", "")

    st.markdown(f"""
    <div class="fd-verdict {css_cls}">
        <div class="vt">Final Verdict</div>
        <div class="vv">{verdict}</div>
        <div class="ve">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Saved badge ───────────────────────────────────────────────────────────
    if db_id:
        st.markdown(f"""
        <div class="fd-saved">
            💾 Saved to database — SQLite ID: {db_id} · JSON ID: {json_id}
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    scores = result.get("scores", {})
    ev_conf    = scores.get("evidence_confidence", 0)
    src_score  = scores.get("source_score", 0)
    critic_score = scores.get("critic_score", 0)
    reliability  = result.get("source_reliability", "—")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="fd-card">
            <div class="label">Evidence Confidence</div>
            <div class="value">{ev_conf}%</div>
            <div class="sub">Based on retrieved evidence</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="fd-card">
            <div class="label">Source Reliability</div>
            <div class="value">{reliability}</div>
            <div class="sub">Source credibility score: {src_score}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="fd-card">
            <div class="label">Critic Score</div>
            <div class="value">{critic_score}%</div>
            <div class="sub">Confidence after critique</div>
        </div>""", unsafe_allow_html=True)

    # ── Evidence & Critic detail ──────────────────────────────────────────────
    col_ev, col_cr = st.columns(2, gap="medium")

    with col_ev:
        st.markdown(f"""<div class="fd-section">
            <div class="sh">📄 Evidence Summary</div>
            {result.get('evidence', 'N/A')}
        </div>""", unsafe_allow_html=True)

        sources = result.get("sources", [])
        if sources:
            pills = "".join(
                f'<span class="fd-pill {s.get("credibility","Medium")}">{s["name"]} · {s.get("credibility","?")}</span>'
                for s in sources
            )
            st.markdown(f"""<div class="fd-section">
                <div class="sh">🌐 Identified Sources</div>
                <div class="fd-pills">{pills}</div>
            </div>""", unsafe_allow_html=True)

    with col_cr:
        st.markdown(f"""<div class="fd-section">
            <div class="sh">⚔️ Critic Challenges</div>
            {result.get('critic_challenges', 'N/A')}
        </div>""", unsafe_allow_html=True)

    # ── Composite score bar ───────────────────────────────────────────────────
    composite  = scores.get("composite", 0)
    bar_color  = "#2ecc71" if composite >= 65 else ("#f1c40f" if composite >= 40 else "#e74c3c")
    latency_ms = result.get("latency_ms", 0)

    st.markdown(f"""
    <div style="margin-top:1.2rem">
        <div style="font-size:0.72rem;letter-spacing:.12em;text-transform:uppercase;color:#444;margin-bottom:.5rem">
            Composite Truthfulness Score
        </div>
        <div style="background:#1a1a1a;border-radius:3px;height:8px;overflow:hidden">
            <div style="width:{composite}%;background:{bar_color};height:100%;border-radius:3px"></div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:{bar_color};margin-top:.4rem">
            {composite}% — weighted average (evidence 35% · source 35% · critic 30%)
        </div>
        <div style="font-size:0.75rem;color:#444;margin-top:.3rem">
            ⏱ Pipeline latency: {latency_ms}ms
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style='text-align:center;font-size:0.72rem;color:#333;margin-top:1rem'>
    Powered by LangGraph · Groq llama-3.1-8b · FastAPI · SQLite
</p>
""", unsafe_allow_html=True)