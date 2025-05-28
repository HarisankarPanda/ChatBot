import streamlit as st
from chat import get_response
from retriever import embed_uploaded_file
import os

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("🧠 Local RAG Chatbot")

# Sidebar for file upload and instructions
with st.sidebar:
    st.header("Upload PDF")
    st.markdown("Upload your PDFs here to embed and ask questions based on their content.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file:
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Embedding file..."):
            embed_uploaded_file(file_path)
        st.success(f"'{uploaded_file.name}' has been embedded successfully!")
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Chat history setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with styling
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input with loading spinner
if query := st.chat_input("Ask something..."):
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Generating response..."):
        response = get_response(query)
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
