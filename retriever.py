from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_PATH = "./chroma_db"

def embed_uploaded_file(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)
    vectordb.add_documents(chunks)
    vectordb.persist()
