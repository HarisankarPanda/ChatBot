from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
#from langchain_community.chains import RetrievalQA
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedder)
llm = Ollama(model="mistral")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

def get_response(query):
    return qa_chain.run(query)
