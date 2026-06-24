import streamlit as st
import sys
import os
import time

sys.path.insert(0, ".")
from src.retrieval.rag_chain import ask as rag_ask

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Jordan Legal RAG Assistant",
    page_icon="⚖️",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("⚖️ Jordan Legal RAG Assistant")
st.markdown(
    "Ask questions about **Jordanian Data Protection Law No. 24 of 2023**, "
    "DPIA Guidelines, and Data Disclosure Regulations in English or Arabic."
)
st.divider()

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display chat history ──────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.caption(source)

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Jordanian data protection law..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            try:
                start = time.time()
                result = rag_ask(prompt)
                response_time = round(time.time() - start, 2)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                with st.expander("📄 Sources"):
                    for source in sources:
                        st.caption(source)

                st.caption(f"⏱ Response time: {response_time}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Documents Loaded")
    st.markdown("""
    - Personal Data Protection Law No. 24 (2023)
    - Data Protection Impact Assessment Guidelines
    - Data Disclosure Regulation (2025)
    - Official English Translations
    """)

    st.divider()
    st.header("💡 Sample Questions")
    sample_questions = [
        "What are the rights of data subjects?",
        "When is a DPIA required?",
        "What are the penalties for data breaches?",
        "ما هي حقوق أصحاب البيانات؟",
        "متى يجب إجراء تقييم أثر حماية البيانات؟"
    ]
    for q in sample_questions:
        st.caption(f"• {q}")

    st.divider()
    st.caption("Built by Raneem Abujabal")
    st.caption("Powered by Groq LLaMA3 + ChromaDB")