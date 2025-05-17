import streamlit as st
import requests
import os

# Set page config
st.set_page_config(page_title="GraphRAG QA", page_icon="🧠")

# Custom dark theme styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0e1117;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #262730;
            color: white;
        }
        .stChatInputContainer {
            background-color: #262730;
            color: white;
        }
        .stChatMessage {
            background-color: #1e222a;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        code {
            background-color: #2d2f3a;
            color: #dcdcaa;
        }
    </style>
""", unsafe_allow_html=True)

# Backend URL (change if needed)
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/v1/chat/completions")

st.title("🧠 Agentic GraphRAG Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    payload = {
        "model": "qwen",
        "messages": st.session_state.chat_history
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        assistant_msg = result["choices"][0]["message"]["content"]
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    except Exception as e:
        assistant_msg = f"❌ Error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])