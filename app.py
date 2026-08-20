# ============================================
# StudyMate AI - College Notes Assistant
# RAG-Based Document Question Answering
# ============================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from chatbot import PDFStudyAssistant

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
)   

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .source-tag {
        background: #fff3cd;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state — init ONCE, persist across reruns ─────────────────
if "assistant" not in st.session_state:
    st.session_state.assistant    = PDFStudyAssistant()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "files_loaded" not in st.session_state:
    st.session_state.files_loaded = []

assistant = st.session_state.assistant


# ── Header ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📚 StudyMate AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">AI-Powered College Notes Assistant using RAG</div>',
    unsafe_allow_html=True
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type                  = ["pdf", "txt"],
        accept_multiple_files = True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.files_loaded:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Read bytes BEFORE any other Streamlit call
                    file_bytes = uploaded_file.read()
                    result = st.session_state.assistant.load_from_bytes(
                        file_bytes, uploaded_file.name
                    )
                if result["success"]:
                    st.session_state.files_loaded.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} — {result['chunks']} chunks")
                else:
                    st.error(f"❌ Failed: {uploaded_file.name}")

    st.divider()

    st.subheader("📝 Quiz Generator")

num_questions = st.slider(
    "Number of questions",
    min_value=3,
    max_value=10,
    value=5
)

difficulty = st.selectbox(
    "Difficulty",
    ["Easy", "Medium", "Hard"]
)

if st.button("📝 Generate Quiz"):

    if st.session_state.assistant.is_ready:

        with st.spinner("Generating quiz from your document..."):

            quiz = st.session_state.assistant.generate_quiz(
                num_questions=num_questions,
                difficulty=difficulty
            )

        st.session_state.quiz = quiz

    else:

        st.warning("Please upload a document first.")


    if "quiz" in st.session_state:

        st.divider()

        st.subheader("📝 Your AI-Generated Quiz")

        st.markdown(st.session_state.quiz)
    
    # ── Loaded files ──────────────────────────────────────────────────
    if st.session_state.files_loaded:
        st.subheader("📄 Loaded Files")
        for fname in st.session_state.files_loaded:
            st.markdown(f"• {fname}")
        st.divider()

    # ── Stats ─────────────────────────────────────────────────────────
    if st.session_state.assistant.is_ready:
        stats = st.session_state.assistant.get_stats()
        vs    = stats["vector_store"]
        docs  = stats["documents"]

        st.subheader("📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files",  docs["files_loaded"])
        with col2:
            st.metric("Chunks", docs["total_chunks"])
        st.metric("Vectors", vs["doc_count"])
        st.divider()

    # ── Actions ───────────────────────────────────────────────────────
    st.subheader("📝 Study Tools")

    if st.button("📝 Summarise Documents", use_container_width=True):
        if st.session_state.assistant.is_ready:
            with st.spinner("Generating summary..."):
                summary = st.session_state.assistant.get_document_summary()
            st.session_state.chat_history.append({
                "role"   : "assistant",
                "content": f"**📝 Document Summary:**\n\n{summary}",
                "sources": [],
            })
            st.rerun()
        else:
            st.warning("Upload a document first.")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.assistant.clear_chat()
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Clear Everything", use_container_width=True):
        st.session_state.assistant.clear_all()
        st.session_state.chat_history = []
        st.session_state.files_loaded = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1])

with col_chat:
    st.subheader("💬 Chat")

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if msg.get("sources"):
                    src_text = " ".join(
                        f'<span class="source-tag">📄 {s}</span>'
                        for s in msg["sources"]
                    )
                    st.markdown(f"**Sources:** {src_text}", unsafe_allow_html=True)

    # Chat input
    if "summary" in st.session_state:

        st.divider()

        st.subheader("📋 Document Summary")

        st.markdown(st.session_state.summary)
    if prompt := st.chat_input(
        "Ask anything about your document...",
        disabled=not st.session_state.assistant.is_ready,
    ):
        st.session_state.chat_history.append({
            "role"   : "user",
            "content": prompt,
            "sources": [],
        })

        with st.spinner("Thinking..."):
            result = st.session_state.assistant.chat(prompt)

        st.session_state.chat_history.append({
            "role"   : "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

        st.rerun()

    if not st.session_state.assistant.is_ready:
        st.info("⬅️ Upload a PDF or TXT file from the sidebar to start chatting.")


with col_info:
    st.subheader("💡 Tips")
    st.markdown("""
**💬 Ask questions**
- What is this topic about?
- Explain this concept in simple language.
- What are the key points?
- Explain this with an example.

**📝 Study Tools**
- Generate an AI quiz from your notes.
- Choose quiz difficulty and number of questions.
- Generate a concise document summary.

**📚 Supported**
- PDF files
- TXT files
- Multiple documents
- Page-level source citations

**⚙️ Powered by**
- Google Gemini
- LangChain
- FAISS
- HuggingFace Embeddings
""")

    st.sidebar.markdown("### 📋 Study Tools")

    if st.sidebar.button("📋 Generate Summary"):

        if st.session_state.assistant.is_ready:

            with st.spinner("Generating summary..."):

                summary = st.session_state.assistant.get_document_summary()

        st.session_state.summary = summary

    else:

        st.warning("Please upload a document first.")