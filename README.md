# 📚 StudyMate AI

### AI-Powered College Notes Assistant using RAG

StudyMate AI is a Retrieval-Augmented Generation (RAG) based study assistant that allows students to upload their college notes and ask questions about them.

The system retrieves relevant information from uploaded documents using semantic search and uses Google Gemini to generate grounded answers.

## ✨ Features

- 📄 Upload PDF and TXT study materials
- 💬 Ask questions about uploaded notes
- 🔎 Semantic search using FAISS
- 🧠 HuggingFace embeddings using `all-MiniLM-L6-v2`
- 📚 Page-level source citations
- 📝 AI-generated quizzes
- 🎯 Adjustable quiz difficulty
- 🔢 Adjustable number of quiz questions
- 📋 AI-generated document summaries
- 💭 Conversational follow-up questions
- 📊 Document and vector statistics
- 🌐 Streamlit web interface

## 🏗️ RAG Architecture

```text
                PDF / TXT
                    │
                    ▼
            Document Processing
                    │
                    ▼
              Text Chunking
                    │
                    ▼
        HuggingFace Embeddings
          all-MiniLM-L6-v2
                    │
                    ▼
             FAISS Vector Store
                    │
                    ▼
             Similarity Search
                    │
              Top-K Chunks
                    │
                    ▼
              RAG Engine
                    │
          Context + User Query
                    │
                    ▼
             Google Gemini
                    │
                    ▼
            Generated Response

## 📂 Project Structure

StudyMate-AI/
│
├── src/
│   ├── chatbot.py
│   ├── config.py
│   ├── document_processor.py
│   ├── rag_engine.py
│   └── vector_store.py
│
├── app.py
├── requirements.txt
├── test_chatbot.py
├── .env.example
├── .gitignore
└── README.md

## 🔧 Tech Stack

Technology	    Purpose
Python	        Core programming language
Streamlit	    Web interface
LangChain	    RAG pipeline and LLM integration
FAISS	        Vector similarity search
HuggingFace	    Text embeddings
Google Gemini	Answer and content generation
PyPDF	        PDF document processing

## 🔄 How RAG Works

1. Document Upload

The user uploads a PDF or TXT document.

2. Text Extraction

The application extracts the document content and divides it into smaller overlapping chunks.

3. Embeddings

Each chunk is converted into a numerical vector using the HuggingFace all-MiniLM-L6-v2 embedding model.

4. Vector Storage

The embeddings are stored in a FAISS vector index.

5. Retrieval

When the user asks a question, the question is converted into an embedding and FAISS retrieves the most relevant document chunks.

6. Generation

The retrieved chunks are provided as context to Google Gemini, which generates the final answer.

This helps the assistant answer using information from the uploaded documents instead of relying only on general model knowledge.

## 📝 Custom Features

Page-Level Source Citations

Answers include the source document and page number of the retrieved content.

Example:Sources:
📄 notes.pdf — Page 12
📄 notes.pdf — Page 15

AI Quiz Generator

Users can generate multiple-choice questions from their study material.

Options include:

Number of questions
Easy
Medium
Hard

Each question includes:

Four options
Correct answer
Short explanation
AI Document Summary

The application can generate a concise summary of important concepts from the uploaded study material.

## 🎯 Project Goal

The project is designed as a lightweight AI study assistant for college students. It demonstrates the practical implementation of a RAG pipeline using document processing, embeddings, vector search, and an LLM.

##📌 Future Improvements
Better quiz interface with interactive answer selection
Support for more document formats
Persistent vector database
User authentication
Improved retrieval and reranking
Deployment on a cloud platform

## 🙌 Attribution


This project was customized and extended from the original
https://github.com/Prateek234775/AI-PDF-Study-Assistant?utm_source=chatgpt.com.

