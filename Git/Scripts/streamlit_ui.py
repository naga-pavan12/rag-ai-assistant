import streamlit as st
from utils import save_chat_history, load_chat_history
from src.text_to_docx import format_prd
from src.rag_chain import run_query_with_debug  # updated version
import uuid
import datetime
from io import BytesIO

# ─── Material-Inspired CSS ─────────────────────────────────────────
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }

    .stChatMessage {
        background: #f4f6f8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 1rem;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.05);
    }

    .stChatMessage.user {
        background: #e8f0fe;
        border-left: 4px solid #4285F4;
    }

    .stChatMessage.assistant {
        background: #f1f3f4;
        border-left: 4px solid #34A853;
    }

    .stDownloadButton button, .stButton button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .stTextInput>div>div>input {
        padding: 1rem;
        border-radius: 0.9rem;
        font-size: 1rem;
    }

    h1 { font-size: 1.8rem; font-weight: 600; }
    h2 { font-size: 1.3rem; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# ─── Session Initialization ─────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(st.session_state.user_id)
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "last_prd" not in st.session_state:
    st.session_state.last_prd = ""

# ─── Sidebar for Notes & Downloads ─────────────────────────────────
with st.sidebar:
    st.markdown("## 🗒️ Notes Taking")
    st.session_state.notes = st.text_area(
        "Your thoughts...", st.session_state.notes, height=480)

    st.download_button("💾 Save Notes",
                       st.session_state.notes, "notes.txt", "text/plain")

    if st.session_state.last_prd:
        buffer = BytesIO()
        doc = format_prd(st.session_state.last_prd)
        doc.save(buffer)
        buffer.seek(0)
        st.download_button(
            "📥 Download Last PRD",
            data=buffer,
            file_name="generated_prd.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    st.caption("Built with ❤️ by ASBL Progress Porduct Team")

# ─── Chat Title ────────────────────────────────────────────────────
st.title("🧠 Progress AI Assistant 1.0")
st.caption(
    "A smart product assistant trained on the Progress platform data.")

# ─── Chat History ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input Box ─────────────────────────────────────────────────────
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("Product Manager"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        answer, sources, collection_names = run_query_with_debug(user_input)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

    with st.chat_message("Progress Assistant"):
        st.markdown(answer)

        if sources:
            with st.expander("📂 Show Retrieved Context"):
                for i, doc in enumerate(sources):
                    col = doc.metadata.get("source", "unknown")
                    st.markdown(
                        f"**Chunk {i+1}** (from `{col}`):\n```\n{doc.page_content}\n```")

    # PRD detection
    if "Title:" in answer and "Objective:" in answer:
        st.session_state.last_prd = answer

    save_chat_history(st.session_state.user_id, st.session_state.messages)

# ─── Bottom Bar: Clear / Export ───────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_prd = ""
        save_chat_history(st.session_state.user_id, [])
        st.experimental_rerun()

with col2:
    if st.button("📥 Download Chat"):
        log = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("💬 Export Chat (.txt)", log,
                           f"chat_{ts}.txt", "text/plain")
