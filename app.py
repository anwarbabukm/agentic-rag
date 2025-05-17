# agentic_graphrag_app/streamlit_app.py
import streamlit as st
from agent.agent_runner import answer_query
from agent.loader import load_and_ingest  

st.set_page_config(page_title="Agentic GraphRAG Chat", layout="wide")
st.title("🤖 Agentic GraphRAG Chat Interface")

# Custom CSS to adjust sidebar width
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 250px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar loader button
with st.sidebar:
    st.header("⚙️ Data Loader")
    if st.button("📂 Load Knowledge Base"):
        with st.spinner("Loading and indexing documents..."):
            status = load_and_ingest()
        if status == "success":
            st.success("Documents loaded and indexed into Qdrant and Neo4j!")
        else:
            st.error("❌ Document loading failed!")

# Session state to hold chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input area (bottom of screen style)
for sender, message in st.session_state.chat_history:
    with st.chat_message("assistant" if sender == "Agent" else "user"):
        st.markdown(message)

# User input at the bottom
user_input = st.chat_input("Ask your question...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = answer_query(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Agent", response))