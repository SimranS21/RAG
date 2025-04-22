# 📄 Multimodal Document Processing RAG with LangChain

An intelligent **Streamlit-based application** designed for uploading and querying multimodal documents with ease. The system extracts, embeds, stores, and retrieves information from a variety of file formats, powered by **LangChain**, **Milvus**, **Transformers**, **Whisper**, and more.

---

## 🚀 What This Project Does

This app provides an end-to-end workflow for processing diverse document types and performing smart searches using **RAG (Retrieval-Augmented Generation)**. Whether it’s audio files, PDFs, or YAML configs — this tool helps you extract insights using natural language queries.

---

## ✨ Key Features

### 🗂️ File Upload & Content Extraction

Supports multiple input formats:

- **Textual**: `.txt`, `.pdf`, `.csv`, `.json`, `.yaml`, `.docx`  
- **Media**: `.mp3`, `.wav`, `.mp4`

Extraction handled by:

- 🔊 **Audio**: `Whisper` + `pydub`  
- 🎥 **Video**: Custom parsing pipeline  
- 📄 **Text/Docs**: LangChain document loaders  

---

### 🧠 Smart Storage with Milvus

- Vectorizes content using **HuggingFace Embeddings**  
- Stores vectors in **Milvus** for high-speed similarity search  

---

### 🔍 AI-Powered Querying

- Ask natural language questions  
- Retrieves relevant data via LangChain’s **RAG mechanism**  
- Returns intelligent, contextual responses  

---

## 🖥️ Getting Started

### ▶️ Launch the App

```bash

streamlit run app.py

🧭 Modes of Operation::

📤 Upload Mode

- Upload any supported file  
- Extract content and store embeddings in **Milvus**  
- View extracted data immediately  

❓ Query Mode

- Enter a natural language question  
- Fetch related chunks from the **vector database**  
- Generate AI-driven answers using **LangChain**

---

📁 Project Structure

📁 project-root/
├── app.py                # Main app logic: UI, upload, query
├── utils/
│   ├── audio_utils.py    # Audio chunking & transcription
│   ├── video_utils.py    # Video file processing
│   └── document_utils.py # CSV, JSON, YAML, PDF, etc.
└── README.md             # Project documentation


📌 Example Workflow

🔼 Upload & Store

1. Switch to **Upload Files**  
2. Drop in a file (e.g., `meeting.mp3`)  
3. View extracted content and embedding status  

❓ Ask a Question

1. Go to **Query**  
2. Type in a question like _"What was discussed about project deadlines?"_  
3. Receive a detailed, factual answer using **RAG**

---

🌱 Future Improvements
 
- 📁 Support more file formats and embedding strategies  
- ⚙️ Scale for large corpora and concurrent queries  

---

🧠 Intent Detection

- Determines whether a user query is **chitchat** or requires **vector database search**  
- Helps optimize performance by skipping retrieval for casual or irrelevant questions  
- Routes intent-based queries to LangChain’s **RAG pipeline**, and handles small talk separately  
