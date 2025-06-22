# rag-ai-assistant
# 🤖 AI PRD Agent – Generate Product Requirement Docs Using Your Own Data

🚀 This open-source AI agent automatically generates PRDs (Product Requirement Documents) using your internal product data and past documentation — fully offline, using local models.

## 🔍 What It Does

 ✅ Takes your product metadata (`.jsonl`) and old PRDs (`.txt`)
 ✅ Stores them in a vector database (ChromaDB)
 ✅ Uses RAG (Retrieval-Augmented Generation)
 ✅ Runs a local LLM (LLaMA 3 via Ollama)
 ✅ Generates full-length PRDs via a Streamlit chatbot

---

## 📦 Tech Stack

| Layer       | Tool               |
|-------------|------------------- |
| Embedding   | `nomic-embed-text` |
| Vector DB   | ChromaDB           |
| LLM         | LLaMA 3 via Ollama |
| Frontend    | Streamlit          |
| Language    | Python             |

---

## 📁 Folder Structure

```bash
├── ui.py             # Streamlit interface
├── agent_core.py     # RAG pipeline
├── data/             # Sample product data
├── vectorstore_db/   # Vector DB location
├── prompts/          # Customizable prompt formats
├── requirements.txt  # All Python deps
