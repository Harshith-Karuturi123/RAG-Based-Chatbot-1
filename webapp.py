import os
from dotenv import load_dotenv          # <-- added
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # This for chunking
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf

# ------------------- LOAD ENV -------------------
load_dotenv()                            # <-- load .env file
key = os.getenv("GOOGLE_API_KEY")        # <-- get API key
genai.configure(api_key=key)             # <-- configure GenAI

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips = '''Follow the steps to use this application:
* Upload your pdf document in sidebar.
* Write your query and start chatting with the bot.'''
st.subheader(tips)

# Load PDF in Side Bar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF Only)]')
file_uploaded = st.sidebar.file_uploader('Upload File')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)
    
    # Configure LLM
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Configure Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Step 2 : Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 3: Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Step 4: Configure retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Function to generate response
    def generate_response(query):
        # Step 6: Retrieval (R) using similarity_search
        retrived_docs = vector_store.similarity_search(query, k=3)  # returns list of Document objects
        context = ' '.join([doc.page_content for doc in retrived_docs])

        # Step 7: Augmented prompt
        prompt = f'''You are a helpful assistant using RAG
Here is the context = {context}
The query asked by user is as follows = {query}'''

        # Step 9: Generation (G)
        content = llm_model.generate_content(prompt)
        return content.text

# ---------------- CHAT STATE ----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    if msg['role'] == 'user':
        st.write(f":green[User:] :blue[{msg['text']}]")
    else:
        st.write(f":orange[Chatbot:] {msg['text']}")

# ---------------- CHAT INPUT ----------------
with st.form('Chat Form', clear_on_submit=True):
    user_input = st.text_input(
        'Enter your query here:',
        disabled=not file_uploaded
    )
    send = st.form_submit_button(
        'Send',
        disabled=not file_uploaded
    )

# ---------------- PROCESS QUERY ----------------
if file_uploaded and user_input and send:
    st.session_state.history.append({"role": "user", "text": user_input})

    model_output = generate_response(user_input)

    st.session_state.history.append({"role": "chatbot", "text": model_output})

    st.rerun()