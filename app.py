import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Jordan Legal RAG Assistant",
    page_icon="⚖️",
    layout="centered"
)

# ── API URL ───────────────────────────────────────────────────
# When running locally the API is on port 8000
# When deployed we'll update this to the Render URL
API_URL = "http://localhost:8000"

# ── Header ────────────────────────────────────────────────────
st.title("⚖️ Jordan Legal RAG Assistant")
st.markdown(
    "Ask questions about **Jordanian Data Protection Law No. 24 of 2023**, "
    "DPIA Guidelines, and Data Disclosure Regulations in English or Arabic."
)
st.divider()

# ── Chat history ──────────────────────────────────────────────
# st.session_state persists data across reruns
# Without this, chat history disappears on every interaction
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

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    response_time = data["response_time_seconds"]

                    st.markdown(answer)

                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.caption(source)

                    st.caption(f"⏱ Response time: {response_time}s")

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    "Make sure the FastAPI server is running on port 8000."
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out. The API is taking too long.")

