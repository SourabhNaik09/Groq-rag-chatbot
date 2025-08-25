import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# -------------------------
# Your Groq API key here
# -------------------------
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

# -------------------------
# Initialize vector store
# -------------------------
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")  # URL to load
    st.session_state.docs = st.session_state.loader.load()

    if not st.session_state.docs:
        st.warning("No documents loaded from the URL. Check the URL or loader.")
    else:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )
        st.session_state.vector = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# -------------------------

# Streamlit UI
# -------------------------
st.title("Chat Groq Demo")

user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start_time = time.process_time()

    # -------------------------
    # Initialize LLM
    # -------------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"  # Supported model
    )

    answer = None
    docs = []

    # -------------------------
    # Use RetrievalQA if documents exist
    # -------------------------
    if "vector" in st.session_state:
        retriever = st.session_state.vector.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        try:
            result = qa_chain({"query": user_prompt})
            answer = result.get("result", "")
            docs = result.get("source_documents", [])
        except Exception as e:
            st.error(f"Error in RetrievalQA chain: {e}")

    # -------------------------
    # Fallback to direct LLM if no answer
    # -------------------------
    if not answer:
        try:
            answer = llm.invoke({"input": user_prompt})
        except Exception as e:
            st.error(f"Error in direct LLM call: {e}")

    # -------------------------
    # Show response
    # -------------------------
    st.write("Response time:", round(time.process_time() - start_time, 2), "seconds")
    st.write("Answer:", answer)

    # -------------------------
    # Show source documents (if any)
    # -------------------------
    if docs:
        with st.expander("Documents similarity search"):
            for i, doc in enumerate(docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("-----------------------")
