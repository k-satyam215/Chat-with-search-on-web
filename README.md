# Chat with Search on Web 🔎🤖

**Chat with Search on Web** is an **AI research assistant** built using **Streamlit, LangChain, and Groq LLMs** that can answer questions by intelligently combining **live web search**, **Wikipedia**, and **Arxiv research papers**.

Unlike a basic chatbot, this system uses a **tool-augmented reasoning agent** that decides when and how to query external sources to generate **grounded, research-grade answers**.

---

## 🎯 Objective

The goal of this project is to:
- Build a **search-aware AI assistant**
- Overcome the static knowledge limitation of LLMs
- Demonstrate **agentic reasoning with external tools**
- Provide concise, factual, and up-to-date answers for research and technical queries

---

## 🚀 Key Features

### 🤖 Agentic AI with Tool Usage
- Zero-shot ReAct agent (reason + act loop)
- Dynamically decides which tool to use
- Handles multi-step reasoning automatically

---

### 🔍 Web-Integrated Search
- **DuckDuckGo Search** for real-time web results
- **Wikipedia API** for factual grounding
- **Arxiv API** for academic & research papers

---

### 🧠 LLM Powered by Groq
- Uses **Groq-hosted LLaMA 3.1 (8B Instant)**
- Fast inference with streaming responses
- Controlled and concise answer generation

---

### 💬 Interactive Chat Interface
- Streamlit-based chat UI
- Persistent conversation history
- Live reasoning visualization using callbacks

---

### 🎨 Polished UI/UX
- Custom gradient background
- Glassmorphism chat bubbles
- Sidebar-based API key input
- Wide-layout research-friendly interface

---

## 🧠 How It Works

1. User enters a question in natural language
2. The agent analyzes the query
3. Based on intent, it may:
   - Search the web
   - Query Wikipedia
   - Fetch Arxiv papers
4. Retrieved information is combined with LLM reasoning
5. The agent produces a **grounded final answer**

This follows a **tool-augmented reasoning loop**, not simple text generation.

---

## 📂 Project Structure

Chat-with-search-on-web/ │ ├── app.py                  # Streamlit application ├── requirements.txt        # Dependencies ├── .env                    # Environment variables (optional) └── README.md

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Groq (LLaMA 3.1)**
- **DuckDuckGo Search**
- **Wikipedia API**
- **Arxiv API**
- **dotenv**

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/k-satyam215/Chat-with-search-on-web.git
cd Chat-with-search-on-web
