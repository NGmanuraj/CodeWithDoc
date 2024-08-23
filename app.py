import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data_loaders import load_pdf, load_text
from text_processing import split_documents
from database import connect_db, insert_data
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if HUGGINGFACE_API_KEY is None:
    st.error("Hugging Face API key is not set in the environment variables.")
    st.stop()

# Initialize SentenceTransformer model
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def get_embeddings(texts):
    return np.array(model.encode(texts), dtype=np.float32)

def create_faiss_index(chunks):
    if not chunks:
        return None
    embeddings = get_embeddings(chunks)
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_information(query, vectorstore):
    query_embedding = get_embeddings([query])
    if query_embedding.size == 0:
        return [], []
    distances, indices = vectorstore.search(query_embedding, k=5)
    return distances, indices

def query_llm(prompt):
    # Replace this with actual LLM API integration if required
    return f"Generated response for prompt: {prompt}"

def format_extracted_information(info):
    processed_info = []
    for item in info:
        if isinstance(item, str):
            processed_info.append(item)
        elif isinstance(item, dict):
            if 'text' in item:
                processed_info.append(item['text'])
            else:
                st.write(f"Unexpected dictionary format: {item}")
                processed_info.append(str(item))
        else:
            st.write(f"Unexpected item encountered: {item}")
            processed_info.append(str(item))

    text = ' '.join(processed_info)
    text = text.replace(' . ', '. ').replace(' , ', ', ').replace(' ; ', '; ').replace(' : ', ': ')
    return text

def load_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.RequestException as e:
        st.error(f"Error loading URL: {e}")
        return None

def main():
    st.title("🧠 Information Extraction & Retrieval App")

    # Initialize session state if not already
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.chunks = None
        st.session_state.documents = None
        st.session_state.url = ""

    # Display chat history
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            message_type = 'user-message' if message['role'] == 'user' else 'bot-message'
            st.markdown(f"<div class='message {message_type}'>{message['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # User input and "Extract Specific Information" section
    with st.container():
        st.header("🔍 Extract Specific Information")

    user_input = st.text_input("Type your message:", "")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})

        if st.session_state.vectorstore and st.session_state.chunks:
            with st.spinner("Searching..."):
                distances, indices = search_information(user_input, st.session_state.vectorstore)
                result_chunks = [st.session_state.chunks[i] for i in indices[0]] if indices.size > 0 else []
                formatted_result = format_extracted_information(result_chunks)
                
                llm_result = query_llm(formatted_result)
                
                st.session_state.chat_history.append({'role': 'bot', 'content': llm_result})
                st.write(f"Debug: Bot Response - {llm_result}")  # Debug statement
                
                conn, cursor = connect_db()
                insert_data(cursor, user_input, llm_result)
                st.success("Search results saved to database.")
        else:
            st.error("Please upload and process data first.")

    # Data upload and processing
    st.sidebar.header("📂 Upload or Enter Data")
    data_source = st.sidebar.selectbox("Select Data Source", ("PDF", "URL", "Text File"))

    documents = None
    chunks = None

    if data_source == "PDF":
        uploaded_file = st.sidebar.file_uploader("📄 Upload PDF File", type=["pdf"])
        if uploaded_file is not None:
            documents = load_pdf(uploaded_file)
            st.sidebar.success("PDF Loaded Successfully!")
    elif data_source == "URL":
        st.session_state.url = st.sidebar.text_input("🔗 Enter URL", value=st.session_state.url)
    elif data_source == "Text File":
        uploaded_file = st.sidebar.file_uploader("📃 Upload Text File", type=["txt"])
        if uploaded_file is not None:
            documents = load_text(uploaded_file)
            st.sidebar.success("Text File Loaded Successfully!")

    if st.sidebar.button("🔄 Process Data"):
        if data_source == "URL" and st.session_state.url:
            documents = load_url(st.session_state.url)
            if documents:
                documents = [documents]  # Ensure documents is a list
            else:
                st.error("No data found at the URL.")
                st.session_state.vectorstore = None
                st.session_state.chunks = None
                return
        
        if documents:
            chunks = split_documents(documents)
            vectorstore = create_faiss_index(chunks)
            st.success("Data processed and stored in FAISS")
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.documents = documents
        else:
            st.error("No documents to process.")

if __name__ == "__main__":
    main()
