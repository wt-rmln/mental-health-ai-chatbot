# app.py
import os
import json
import threading
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from data_processing import load_store, get_store

# --- init ---
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("🚨 Please set OPENAI_API_KEY in your environment.")
    st.stop()

client = OpenAI(api_key=API_KEY)
DB_PATH = "db.json"

# Speed knobs
MAX_HISTORY_MSGS = 4          # fewer context messages -> faster
MAX_TOKENS = 200              # smaller output -> faster
TARGET_OUTPUT_TOKENS = 120    # soft limit in prompt

# =========================
# Caching
# =========================
@st.cache_resource(show_spinner=False)
def _get_data_store():
    load_store("./data")
    return get_store()

@st.cache_data(show_spinner=False, ttl=3600)
def _get_child_context(child_name_display: str) -> str:
    ds = _get_data_store()
    # Keep the context compact so each call is fast
    return ds.build_chat_context(
        child_name=child_name_display,
        max_items=6,
        only_significant_changes=True
    )

# =========================
# Session state
# =========================
if "histories" not in st.session_state:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            st.session_state.histories = json.load(f)
    else:
        st.session_state.histories = {}

st.session_state.setdefault("current_name", None)
st.session_state.setdefault("display_name", None)
st.session_state.setdefault("history", [])     # messages sent to model
st.session_state.setdefault("messages", [])    # UI messages
st.session_state.setdefault("data_ctx", None)  # per-child cached context

# =========================
# Helpers
# =========================
def _persist_async(path, data):
    def _do():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    threading.Thread(target=_do, daemon=True).start()

def _is_crisis(text: str) -> bool:
    t = (text or "").lower()
    keywords = ("suicide", "kill myself", "end my life", "self-harm", "hurt myself", "overdose")
    return any(k in t for k in keywords)

CRISIS_MSG = (
    "I’m really sorry you’re going through this. If you’re in immediate danger, call your local emergency number now. "
    "In Canada: **9-8-8** (Suicide Crisis Helpline, 24/7). You’re not alone—help is available."
)

def _stream_chat(messages, max_tokens):
    # Stream tokens so the user sees text immediately
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

def generate_response_stream(user_input: str):
    # Crisis fast-path (zero network call)
    if _is_crisis(user_input):
        def _gen():
            yield CRISIS_MSG
        return _gen()

    system_prompt = (
        "You are a caring mental health assistant for a parent.\n"
        f"Write under ~{TARGET_OUTPUT_TOKENS} tokens total.\n\n"
        "### Summary (3 concise bullets)\n"
        "- Be plain and empathetic.\n"
        "- Use dual-perspective (parent vs teen) and time trends when relevant.\n"
        "- Do **not** show numbers.\n\n"
        "### Suggestions (≈3 actionable ideas)\n"
        "- Concrete at-home steps; brief and specific.\n"
        "- You are not a substitute for professionals."
    )

    child_disp = st.session_state["display_name"]
    data_context = st.session_state.get("data_ctx") or _get_child_context(child_disp) or ""
    # Guard against very long contexts (keep prompt light)
    if len(data_context) > 6000:
        data_context = data_context[:6000] + "…"

    trimmed_history = st.session_state.history[-MAX_HISTORY_MSGS:]
    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Data context for {child_disp}:\n{data_context}"},
        *trimmed_history,
        {"role": "user", "content": user_input},
    ]
    return _stream_chat(payload, MAX_TOKENS)

# =========================
# 1) Choose child (first screen)
# =========================
if st.session_state["current_name"] is None:
    st.title("💬 Mental Health Chatbot Prototype")
    st.write("Hello! I'm here to support your child's mental health.")
    name_input = st.text_input(
        "What's your child's name? (Please enter First name then Last name)",
        key="name_input"
    )
    if st.button("Start Chat") and name_input.strip():
        name_l = name_input.strip().lower()
        name_disp = name_input.strip().title()
        st.session_state.current_name = name_l
        st.session_state.display_name = name_disp

        # Restore history
        persisted = st.session_state.histories.get(name_l, [])
        st.session_state.history = persisted.copy()
        st.session_state.messages = []

        # Warm caches + keep context in session (avoid recompute)
        _get_data_store()
        st.session_state.data_ctx = _get_child_context(name_disp)

        # Greeting
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! How can I help today? I can summarize recent patterns and offer practical suggestions."
        })

    if st.session_state["current_name"] is None:
        st.stop()

# =========================
# 2) Page title
# =========================
st.title(f"Chatting about: {st.session_state['display_name']}")

# =========================
# 3) Chat UI (streaming)
# =========================
# Render history
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    # Show user msg immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "user", "content": prompt})

    # Generate + stream reply
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            reply = st.write_stream(generate_response_stream(prompt))

    # Save reply to state
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.history.append({"role": "assistant", "content": reply})

    # Persist asynchronously (non-blocking)
    key = st.session_state.current_name
    st.session_state.histories[key] = st.session_state.history
    _persist_async(DB_PATH, st.session_state.histories)