# 📄 RAG-Based PDF Chatbot using Gemini & FAISS


## APP LINK
https://rag-based-chatbot-1-bfvizy4xtktetw3e8zat3c.streamlit.app/


### 🚀 Project Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload a PDF document and interact with it through natural language queries.
The chatbot retrieves relevant content from the document using FAISS vector search and generates accurate answers using Google Gemini LLM.

### 🎯 Key Features

📄 Upload and process PDF documents
✂️ Intelligent text chunking for efficient retrieval
🔍 Semantic search using FAISS vector database
🧠 Context-aware responses using Gemini LLM
💬 Interactive chat interface built with Strea
🔐 Secure API key management using .env

### Tech Stack

Python 3.12
Streamlit – Web UI
LangChain (Community) – Text splitting & vector handling
FAISS – Vector similarity search
HuggingFace Embeddings – all-MiniLM-L6-v2
Google Gemini API – Large Language Model
PyPDF – PDF text extraction
python-dotenv – Environment variable management

### 🧩 How It Works (Architecture)

1.User uploads a PDF file
2.Text is extracted and split into chunks
3.Chunks are converted into embeddings
4.FAISS stores embeddings for fast similarity search
5.User query retrieves relevant document chunks
6.Retrieved context is sent to Gemini LLM
7.LLM generates a context-aware response

### 📌 Usage Instructions

Upload a PDF document from the sidebar
Enter your query in the chat box
Receive accurate answers based on document content

### 🔒 Security Best Practices
API keys are stored securely using .env
.env file is excluded using .gitignore
No sensitive data is committed to the repository

## 📈 Future Enhancements

Multi-PDF support
Chat history persistence
Answer citations with page numbers
UI improvements
Cloud deployment (Streamlit Cloud / HuggingFace Spaces)