import time
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter  
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import requests
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

# Set the page configuration at the top
st.set_page_config(page_title="Intelligent Document Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .title { color: #1a73e8; font-weight: bold; font-size: 3rem; text-align: center; }
    .header { color: #3f51b5; font-weight: bold; font-size: 2.5rem; text-align: center; }
    .message { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #ffffff; }
    .user-message { background-color: #e1f5fe; color: #0d47a1; }
    .assistant-message { background-color: #e8f5e9; color: #1b5e20; }
    .stButton button {
        background-color: #3f51b5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background-color: #283593;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text extraction was successful
                text += page_text + "\n"
    return text

# Function to extract text from uploaded DOCX files
def get_docx_text(docx_files):
    text = ""
    for docx_file in docx_files:
        doc = Document(docx_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Function to extract text from uploaded TXT files
def get_txt_text(txt_files):
    text = ""
    for txt_file in txt_files:
        text += txt_file.read().decode("utf-8") + "\n"
    return text

# Function to extract text from web URLs
def get_url_text(urls):
    text = ""
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for paragraph in paragraphs:
                text += paragraph.get_text() + "\n"
        except requests.exceptions.RequestException as e:
            st.warning(f"Error fetching {url}: {e}")
    return text

# Function to split extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store for the text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain using the OpenAI API
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context"; don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    context = vector_store.similarity_search(user_question, k=3)
    context_text = "\n".join([doc.page_content for doc in context])

    chat_model = get_conversational_chain()  # Pass openai_api_key here
    response = chat_model({"context": context_text, "question": user_question})
    assistant_response = response['text']

    # Update chat history
    st.session_state.chat_history.append({"user": user_question, "assistant": assistant_response})
    return assistant_response  # Return the assistant's response

import streamlit as st

def main():
    st.markdown("<h1 class='title'>Intelligent Document Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='header'>Engage in Conversational Queries with Your Documents</h2>", unsafe_allow_html=True)

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
   
    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False

    # Sidebar for API Key and file uploads based on file type selection
    with st.sidebar:
        st.markdown("<h3 class='header'>Upload Files</h3>", unsafe_allow_html=True)
        file_type = st.selectbox("Select the type of content to upload and process", ('PDF', 'DOCX', 'TXT', 'URL'))

        uploaded_files = None  # Variable to hold uploaded files based on type

        if file_type == 'PDF':
            uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
            if st.button("Submit & Process PDFs"):
                if not uploaded_files:
                    st.warning("Please upload PDF files before processing.")
                else:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(uploaded_files)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDF files have been processed. You can ask questions now!")
                        st.session_state.data_processed = True

        elif file_type == 'DOCX':
            uploaded_files = st.file_uploader("Upload your DOCX Files", accept_multiple_files=True, type=['docx'])
            if st.button("Submit & Process DOCX"):
                if not uploaded_files:
                    st.warning("Please upload DOCX files before processing.")
                else:
                    with st.spinner("Processing DOCX files..."):
                        raw_text = get_docx_text(uploaded_files)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("DOCX files have been processed. You can ask questions now!")
                        st.session_state.data_processed = True

        elif file_type == 'TXT':
            uploaded_files = st.file_uploader("Upload your TXT Files", accept_multiple_files=True, type=['txt'])
            if st.button("Submit & Process TXT"):
                if not uploaded_files:
                    st.warning("Please upload TXT files before processing.")
                else:
                    with st.spinner("Processing TXT files..."):
                        raw_text = get_txt_text(uploaded_files)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("TXT files have been processed. You can ask questions now!")
                        st.session_state.data_processed = True

        elif file_type == 'URL':
            url_input = st.text_area("Enter URLs (one per line):", height=200)
            if st.button("Submit & Process URLs"):
                if not url_input.strip():
                    st.warning("Please enter at least one URL.")
                else:
                    urls = url_input.splitlines()  # Using splitlines for cleaner handling
                    with st.spinner("Processing URLs..."):
                        raw_text = get_url_text(urls)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("URLs have been processed. You can ask questions now!")
                        st.session_state.data_processed = True

    # User question input section
    if st.session_state.data_processed:
        st.markdown("<h3 class='header'>Ask a Question</h3>", unsafe_allow_html=True)
        user_question = st.text_area("Type your question here:", height=100, placeholder="What would you like to ask?")

        # Button to submit question
        if st.button("Submit"):
            if not user_question.strip():
                st.warning("Please enter a question before submitting.")
            else:
                with st.spinner("Getting the answer..."):
                    answer = user_input(user_question)
                    st.success("Here is your answer:")
                    st.markdown(f"<div class='message assistant-message'>{answer}</div>", unsafe_allow_html=True)
                    st.session_state.chat_history.append({'user': user_question, 'assistant': answer})  # Store the chat history

        # Display chat history and clear history button
        if st.session_state.chat_history:
            st.markdown("<h3 class='header'>Chat History</h3>", unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                st.markdown(f"<div class='message user-message'>User: {chat['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='message assistant-message'>Assistant: {chat['assistant']}</div>", unsafe_allow_html=True)
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
