"""
app.py
------
Streamlit chat UI for the Website Chatbot & Lead-Qualification Agent.

IMPORTANT: There is no API key input field anywhere in this UI on purpose.
The Gemini API key lives only in a local .env file and is loaded silently
in agent.py / knowledge_base.py via python-dotenv.

Run with:  streamlit run app.py
"""

import streamlit as st
from agent import ChatAgent

st.set_page_config(page_title="Agency Website Chatbot", page_icon="💬")

st.title("💬 Agency Website Chatbot")
st.caption(
    "Ask about our services, pricing, or process — or tell us about your "
    "project and this agent will gather a few quick details for our team."
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
# Building the FAISS index (inside ChatAgent) takes a moment, so we only
# do it once per browser session and reuse it across turns.
if "agent" not in st.session_state:
    with st.spinner("Setting up knowledge base..."):
        st.session_state.agent = ChatAgent()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! How can I help — questions about our "
            "services, or looking to start a project?",
        }
    ]

# ---------------------------------------------------------------------------
# RENDER CHAT HISTORY
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------------------------------------------
# HANDLE NEW MESSAGE
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.session_state.agent.handle_message(user_input)
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
