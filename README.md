# ChatBot
# 🧠 Local RAG Chatbot (Free & Offline)

**Retrieval-Augmented Generation (RAG)** chatbot that answers user queries based on uploaded PDF documents using local Large Language Models (LLMs). Built entirely with **free, open-source tools**—no OpenAI API or internet connection required.

## ✨ Features

- 🔍 Retrieve answers from your own PDFs
- 🧠 Uses local LLMs via [Ollama](https://ollama.com) (e.g., Mistral, LLaMA 3)
- 📄 PDF document parsing with PyMuPDF
- 🧬 Embedding with Sentence Transformers (`all-MiniLM-L6-v2`)
- 🧠 Vector search with ChromaDB
- 💬 Simple chat interface via Streamlit
- 💯 Fully offline & free to use

---

## 🧰 Tech Stack

| Component        | Tool/Library               |
|------------------|-----------------------------|
| LLM              | [Ollama](https://ollama.com) (Mistral) |
| Embeddings       | [sentence-transformers](https://www.sbert.net) |
| Vector DB        | [ChromaDB](https://www.trychroma.com) |
| Document Parsing | [PyMuPDF](https://pymupdf.readthedocs.io) |
| Framework        | [LangChain](https://www.langchain.com) |
| UI               | [Streamlit](https://streamlit.io) |

---
