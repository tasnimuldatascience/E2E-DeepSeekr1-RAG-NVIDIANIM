import streamlit as st
import os
import time
import pickle
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

## Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="deepseek-ai/deepseek-r1")

# Define paths for storing FAISS and metadata
FAISS_INDEX_PATH = "./faiss_index"
METADATA_PATH = "./faiss_metadata.pkl"

def save_faiss_index(vector_store):
    """Save FAISS index and metadata."""
    vector_store.save_local(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(st.session_state.final_documents, f)

def load_faiss_index():
    """Load FAISS index and metadata with safe deserialization."""
    with open(METADATA_PATH, "rb") as f:
        documents = pickle.load(f)

    embeddings = NVIDIAEmbeddings()

    # FIX: Allow FAISS index to load with explicit permission
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    return vector_store, documents

def vector_embedding():
    """Create or load FAISS vector store."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        st.session_state.vectors, st.session_state.final_documents = load_faiss_index()
        st.write("Loaded existing FAISS index.")
    else:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunking
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        save_faiss_index(st.session_state.vectors)
        st.write("Vector Store DB created and saved.")

st.title("RAG using NVIDIA NIM and DeepSeek R1")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    <context>
    
    Questions: {input}
    """
)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready.")

if prompt1:
    if "vectors" not in st.session_state:
        st.write("Please embed documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
