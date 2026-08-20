# ============================================
# Chatbot - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Day 56 - 66 Days GenAI Challenge
# ============================================

import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from config import TOP_K_RETRIEVAL, EMBEDDING_MODEL, LLM_MODEL


class PDFStudyAssistant:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vs        = VectorStore()
        self.rag       = RAGEngine()
        self.is_ready  = False

        print("\n🎓 PDF Study Assistant initialized!")
        print(f"   Embedding model : {EMBEDDING_MODEL}")
        print(f"   LLM model       : {LLM_MODEL}")
        print(f"   Chunk size      : 500 tokens")
        print(f"   Top-K retrieval : {TOP_K_RETRIEVAL}\n")

    # ── Load a file from disk path ────────────────────────────────────
    def load_file(self, file_path: str) -> Dict:
        print(f"\n📂 Loading: {file_path}")
        chunks = self.processor.process_file(file_path)

        if not chunks:
            return {"success": False, "message": f"Failed to load {file_path}", "chunks": 0}

        success = self.vs.add_documents(chunks)
        if success:
            self.is_ready = True

        return {
            "success": success,
            "message": f"Loaded {len(chunks)} chunks from {file_path}",
            "chunks" : len(chunks),
        }

    # ── Load from Streamlit UploadedFile bytes ────────────────────────
    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Dict:
        docs = self.processor.load_from_bytes(file_bytes, filename)

        if not docs:
            return {"success": False, "message": f"Failed to process {filename}", "chunks": 0}

        chunks  = self.processor.split_documents(docs)
        success = self.vs.add_documents(chunks)

        if success:
            self.is_ready = True

        return {
            "success": success,
            "message": f"Processed {len(chunks)} chunks from {filename}",
            "chunks" : len(chunks),
        }

    # ── Chat ──────────────────────────────────────────────────────────
    def chat(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> Dict:
        if not self.is_ready:
            return {
                "answer" : "⚠️ Please upload a document first.",
                "sources": [],
                "chunks" : 0,
            }

        standalone = self.rag.condense_question(question)
        docs       = self.vs.search(standalone, top_k=top_k)
        return self.rag.answer(question, docs)

    # ── Summarise all loaded documents ────────────────────────────────
    def get_document_summary(self) -> str:
        if not self.is_ready:
            return "No documents loaded."

        docs = self.vs.search("main topics overview summary", top_k=8)
        return self.rag.summarise(docs)

    def clear_chat(self):
        self.rag.clear_history()

    def clear_all(self):
        self.vs.clear()
        self.rag.clear_history()
        self.processor.loaded_files = []
        self.is_ready = False

    def generate_quiz(
    self,
    num_questions: int = 5,
    difficulty: str = "Medium"
    ) -> str:

        if not self.is_ready:
            return "⚠️ Please upload a document first."

    # Retrieve important content from the uploaded documents
        docs = self.vs.search(
        "important concepts topics key definitions main ideas",
        top_k=10
        )

        if not docs:
            return "No relevant document content found."

    # Send retrieved chunks to the RAG engine
        return self.rag.generate_quiz(
        docs,
        num_questions=num_questions,
        difficulty=difficulty
    )

    def get_stats(self) -> Dict:
        return {
            "is_ready"    : self.is_ready,
            "documents"   : self.processor.get_stats(),
            "vector_store": self.vs.get_stats(),
            "rag_engine"  : self.rag.get_stats(),
        }
