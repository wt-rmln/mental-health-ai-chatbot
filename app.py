# Streamlit chat UI that streams responses and saves history
import os
import re
import json
import math
import threading
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from data_processing import load_store, get_store

# =========================
# Init
# =========================
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("🚨 Please set OPENAI_API_KEY in your environment.")
    st.stop()
client = OpenAI(api_key=API_KEY)

DB_PATH = "db.json"

# Speed knobs
MAX_HISTORY_MSGS = 4           # fewer context messages -> faster
MAX_TOKENS = 200               # smaller output -> faster
TARGET_OUTPUT_TOKENS = 120     # soft limit in prompt

# RAG knobs
RAG_K_PER_DIM = 2              # how many items to fetch per predicted dimension
RAG_FALLBACK_K = 6             # if no dimension predicted, fetch by keyword only
RAG_SNIPPET_MAX_CHARS = 360    # keep each bullet short
RAG_SECTION_MAX_CHARS = 1200   # cap total evidence length

# =========================
# Robust session_state bootstrapping (MUST run before any access)
# =========================
def _init_state():
    """Ensure all session_state keys exist before any read."""
    defaults = {
        "histories": {},          # persisted chat histories by child key
        "current_name": None,     # lowercase internal key
        "display_name": None,     # title-cased name for UI
        "history": [],            # messages sent to the model (for context)
        "messages": [],           # messages rendered in UI
        "data_ctx": None,         # cached child context string
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Load persisted histories only once per session
if os.path.exists(DB_PATH) and not st.session_state["histories"]:
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            st.session_state["histories"] = json.load(f)
    except Exception:
        # Corrupted DB file should not crash the app
        st.session_state["histories"] = {}

# =========================
# Caching: data loading & compact child context
# =========================
@st.cache_resource(show_spinner=False)
def _get_data_store():
    """Load questionnaires once per run and keep a global handle."""
    load_store("./data")
    return get_store()

@st.cache_data(show_spinner=False, ttl=3600)
def _get_child_context(child_name_display: str) -> str:
    """Return a compact per-child time-series/gap context string."""
    ds = _get_data_store()
    return ds.build_chat_context(
        child_name=child_name_display,
        max_items=6,
        only_significant_changes=True,
        include_text_evidence=True,  # include brief free-text evidence in context
    )

@st.cache_data(show_spinner=False, ttl=3600)
def _get_all_dimensions() -> List[str]:
    """
    Build the dimension catalog FROM DATA (safer than hardcoding).
    If empty, return an empty list gracefully.
    """
    ds = _get_data_store()
    dims = set()
    try:
        if ds.ts_df is not None and not ds.ts_df.empty:
            dims |= set(ds.ts_df["dimension"].dropna().astype(str).tolist())
        if ds.pairs_df is not None and not ds.pairs_df.empty:
            dims |= set(ds.pairs_df["dimension"].dropna().astype(str).tolist())
    except Exception:
        pass
    dims = {str(d).strip() for d in dims if str(d).strip()}
    return sorted(dims)

# =========================
# Helpers
# =========================
def _persist_async(path, data):
    """Write JSON to disk in a background thread to avoid blocking the UI."""
    def _do():
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()

def _is_crisis(text: str) -> bool:
    """Lightweight keyword check for imminent risk content."""
    t = (text or "").lower()
    keywords = ("suicide", "kill myself", "end my life", "self-harm", "hurt myself", "overdose")
    return any(k in t for k in keywords)

CRISIS_MSG = (
    "I’m really sorry you’re going through this. If you’re in immediate danger, call your local emergency number now. "
    "In Canada: **9-8-8** (Suicide Crisis Helpline, 24/7). You’re not alone—help is available."
)

def _to_text(x) -> str:
    """Coerce any value to a clean string for display. Treat NaN/None as empty."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _truncate(s, n: int) -> str:
    """Trim long strings for prompts; always operate on text."""
    s = _to_text(s).strip()
    return (s[: n - 1] + "…") if n > 0 and len(s) > n else s

# =========================
# LLM dimension classifier
# =========================
def classify_dimensions_llm(query: str, allowed_dims: List[str], top_k: int = 3) -> Dict[str, Any]:
    """
    Use a compact LLM call to map a natural-language query to up to top_k
    dimensions from allowed_dims. Returns dict with keys:
      - "dimensions": List[str]
      - "confidence": float in [0,1]
    If no good match, returns {"dimensions": [], "confidence": 0.0}.
    """
    if not allowed_dims or not query.strip():
        return {"dimensions": [], "confidence": 0.0}

    sys = (
        "You are a short text classifier. "
        "Given a parent's question about a teen, pick up to N most relevant dimensions "
        "from the provided list. Only use items from the list. "
        "If none fit, return an empty list. Output MUST be pure JSON."
    )
    dims_blob = ", ".join(f'"{d}"' for d in allowed_dims)
    user = (
        "{"
        f"\"dimensions_catalog\":[{dims_blob}],"
        f"\"n\":{top_k},"
        f"\"query\":{json.dumps(query)}"
        "}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=150,
            messages=[
                {"role": "system", "content": sys},
                {
                    "role": "user",
                    "content": (
                        "Return JSON in the form "
                        "{\"dimensions\":[<up to n items from dimensions_catalog>],\"confidence\":<0..1>}."
                        "Be strict: only pick from dimensions_catalog."
                        f"\nINPUT: {user}"
                    ),
                },
            ],
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return {"dimensions": [], "confidence": 0.0}

    # Parse JSON robustly
    def _parse_json(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{.*\}", s, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    data = _parse_json(text)
    dims = data.get("dimensions") if isinstance(data, dict) else None
    if not isinstance(dims, list):
        dims = []
    dims = [d.strip() for d in dims if isinstance(d, str)]
    dims = [d for d in dims if d in allowed_dims][:top_k]
    try:
        conf = float(data.get("confidence"))
    except Exception:
        conf = 0.0
    return {"dimensions": dims, "confidence": max(0.0, min(1.0, conf))}

# =========================
# Build RAG snippets for the prompt
# =========================
# def _format_evidence_item(item: Dict[str, Any]) -> str:
#     """Compact bullet for one paired Q/A item (robust to numeric/None fields)."""
#     dim = _to_text(item.get("dimension") or "Misc")
#     mo  = _to_text(item.get("month") or "unknown")
#     q   = _truncate(item.get("question_text"), 140)
#     teen = _truncate(item.get("teen_answer_text"), 180)
#     par  = _truncate(item.get("parent_answer_text"), 180)

#     lines = [f"- [{dim} | {mo}] {q}"]
#     if teen:
#         lines.append(f"  • Teen: \"{teen}\"")
#     if par:
#         lines.append(f"  • Parent: \"{par}\"")
#     return "\n".join(lines)

# ---- replace the whole function in app.py ----
def _format_evidence_item(item: Dict[str, Any]) -> str:
    """Compact bullet for one paired Q/A item (works for free-text and multiple-choice)."""
    def _has_num(x) -> bool:
        try:
            return x is not None and not (isinstance(x, float) and math.isnan(x))
        except Exception:
            return x is not None

    def _compose_answer(primary_text: Any, score: Any) -> str:
        """
        Build display string:
        - if primary_text (free text or label) exists, show it
        - if score exists too, append as ' (score)'
        - if primary_text missing but score exists, show just the score
        """
        primary = _to_text(primary_text).strip()
        if _has_num(score):
            try:
                score_txt = f"{float(score):g}"
            except Exception:
                score_txt = _to_text(score)
            if primary:
                primary = f"{primary} ({score_txt})"
            else:
                primary = score_txt
        return primary

    dim = _to_text(item.get("dimension") or "Misc")
    mo  = _to_text(item.get("month") or "unknown")
    q   = _truncate(item.get("question_text"), 140)

    teen_disp = _compose_answer(item.get("teen_answer_text"), item.get("teen_score"))
    par_disp  = _compose_answer(item.get("parent_answer_text"), item.get("parent_score"))

    teen_disp = _truncate(teen_disp, 180)
    par_disp  = _truncate(par_disp, 180)

    lines = [f"- [{dim} | {mo}] {q}"]
    if teen_disp:
        lines.append(f"  • Teen: \"{teen_disp}\"")
    if par_disp:
        lines.append(f"  • Parent: \"{par_disp}\"")
    return "\n".join(lines)


def _build_rag_snippets(child_name_display: str, user_query: str) -> str:
    """
    1) Predict relevant dimensions with an LLM.
    2) Pull recent dual-perspective items from those dimensions, OR fallback by keyword only.
    3) Deduplicate & cap length, then return a compact evidence block string.
    """
    ds = _get_data_store()
    allowed_dims = _get_all_dimensions()
    pred = classify_dimensions_llm(user_query, allowed_dims, top_k=3)
    dims = pred.get("dimensions", [])
    conf = float(pred.get("confidence", 0.0))

    items: List[Dict[str, Any]] = []
    if dims and conf >= 0.20:  # slightly relaxed to reduce "no hit" cases
        for d in dims:
            items.extend(ds.retrieve_dual_perspective(child_name_display, dimension=d, top_k=RAG_K_PER_DIM))
    else:
        items = ds.retrieve_dual_perspective(child_name_display, query=user_query, top_k=RAG_FALLBACK_K)

    if not items:
        return ""

    # Deduplicate by (dimension, month, question_text)
    seen = set()
    deduped = []
    for it in items:
        key = (it.get("dimension"), it.get("month"), it.get("question_text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    bullets = []
    total_chars = 0
    for it in deduped:
        b = _format_evidence_item(it)
        if total_chars + len(b) > RAG_SECTION_MAX_CHARS:
            break
        bullets.append(b)
        total_chars += len(b)

    return "Evidence (recent dual-perspective items):\n" + "\n".join(bullets)

# =========================
# Chat generation (with optional rag_block override for debug preview)
# =========================
def _stream_chat(messages, max_tokens):
    """Stream tokens from the LLM so the user sees text immediately."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception:
        # Graceful degradation on transient errors
        yield "\n\n_(Temporary issue generating a reply. Please try again.)_"

def generate_response_stream(user_input: str, rag_block_override: str | None = None):
    # Crisis fast-path (no network call for safety message)
    if _is_crisis(user_input):
        def _gen():
            yield CRISIS_MSG
        return _gen()

    # System: role + mission only
    system_prompt = "You are a caring mental health assistant for a parent."

    child_disp = st.session_state.get("display_name") or "the child"

    # Compact per-child context (time series, gaps, brief text highlights)
    data_context = st.session_state.get("data_ctx") or _get_child_context(child_disp) or ""
    if len(data_context) > 6000:
        data_context = data_context[:6000] + "…"

    # RAG evidence targeted to the current query
    rag_block = rag_block_override if rag_block_override is not None else _build_rag_snippets(child_disp, user_input)
    rag_block = f"\n<evidence>\n{rag_block}\n</evidence>\n" if rag_block else ""

    trimmed_history = st.session_state.get("history", [])[-MAX_HISTORY_MSGS:]

    # User control block with instructions + data context + evidence
    user_control = (
        f"<data>Data context for {child_disp}:\n{data_context}</data>\n"
        f"{rag_block}\n"
        f"Write under ~{TARGET_OUTPUT_TOKENS} tokens total.\n\n"
        "### Summary (3 concise bullets)\n"
        "- Be plain and empathetic.\n"
        "- Use dual-perspective (parent vs teen) and time trends when relevant.\n"
        # "- Do **not** show numbers.\n\n"
        "### Suggestions (≈3 actionable ideas)\n"
        "- Concrete at-home steps; brief and specific.\n"
        "- You are not a substitute for professionals.\n\n"
        "### Interaction\n"
        "- If the parent's question lacks detail, begin with 1–2 brief follow-up questions (bulleted) to clarify, then provide the answer.\n"
        "- If you don't have specific data for the teen, say so explicitly; do not make it up."
    )

    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_control},
        *trimmed_history,
        {"role": "user", "content": user_input},
    ]
    return _stream_chat(payload, MAX_TOKENS)

# =========================
# Sidebar: utilities
# =========================
st.sidebar.header("Utilities")
show_evidence = st.sidebar.checkbox("Show evidence (debug)", value=False)
if st.sidebar.button("Clear caches"):
    try:
        _get_child_context.clear()
        _get_all_dimensions.clear()
        _get_data_store.clear()
    except Exception:
        pass
    st.rerun()

# =========================
# 1) Choose child (first screen)
# =========================
if st.session_state.get("current_name") is None:
    st.title("💬 Mental Health Chatbot Prototype")
    st.write("Hello! I'm here to support your child's mental health.")

    name_input = st.text_input(
        "What's your child's name? (Please enter First name then Last name)",
        key="name_input"
    )

    if st.button("Start Chat") and name_input.strip():
        name_l = name_input.strip().lower()
        name_disp = name_input.strip().title()
        st.session_state["current_name"] = name_l
        st.session_state["display_name"] = name_disp

        # Restore history for this child
        persisted = st.session_state["histories"].get(name_l, [])
        st.session_state["history"] = persisted.copy()
        st.session_state["messages"] = []

        # Warm caches + hold child context in session (avoid recompute)
        _get_data_store()
        st.session_state["data_ctx"] = _get_child_context(name_disp)

        # Greeting
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Hello! How can I help today? I can summarize recent patterns and offer practical suggestions."
        })

    if st.session_state.get("current_name") is None:
        st.stop()  # wait until user sets the child

# =========================
# 2) Page title
# =========================
st.title(f"Chatting about: {st.session_state.get('display_name', 'Child')}")

# =========================
# 3) Chat UI (streaming)
# =========================
# Render history already in UI list
for msg in st.session_state.get("messages", []):
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    # Optional: build & preview evidence in sidebar (avoids double LLM calls)
    rag_preview = None
    if show_evidence:
        rag_preview = _build_rag_snippets(st.session_state.get("display_name") or "the child", prompt)
        st.sidebar.markdown("#### Evidence preview")
        st.sidebar.code(rag_preview or "(no evidence)")

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["history"].append({"role": "user", "content": prompt})

    # Generate + stream reply
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            reply = st.write_stream(generate_response_stream(prompt, rag_block_override=rag_preview))

    # Save reply to state
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["history"].append({"role": "assistant", "content": reply})

    # Persist asynchronously (non-blocking)
    key = st.session_state["current_name"]
    st.session_state["histories"][key] = st.session_state["history"]
    _persist_async(DB_PATH, st.session_state["histories"])
