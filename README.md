# 🎓 KIET EDU Assist

<div align="center">
  <h3>An Intelligent, Multi-Agent AI Assistant for University Campus Environments</h3>
  <p>Providing seamless, context-aware, real-time responses using advanced RAG and multi-agent orchestration.</p>
</div>

---

## 📖 Overview

**KIET EDU Assist** is an advanced AI chatbot system designed to support students and faculty at KIET University. Operating at **Iteration 6/7**, this system replaces legacy conversational flows with a high-performance, real-time streaming pipeline powered by multiple autonomous agents.

Built on top of **FastAPI**, **LangChain**, and **FAISS**, this solution elegantly combines local Document RAG, intelligent Web Scraping, and dynamic LLM generation to deliver highly accurate answers. It also features a fully comprehensive Admin Dashboard for detailed usage analytics, FAQ management, and Vector Database administration.

---

## 🌟 Key Features

*   **⚡ Real-Time SSE Streaming:** Low latency response generation using Server-Sent Events (SSE) mimicking ChatGPT-like token streaming.
*   **🤖 Multi-Agent Architecture:** A complex orchestration system consisting of specialized agents (FAQ Agent, VectorDB Agent, Webscraper Agent, LLM Agent, Summarizing Agent, and Consistency Check Agent).
*   **📚 Hybrid Knowledge Retrieval (RAG):**
    *   **Local FAISS Vector DB:** Indexes static resources (PDFs, TXTs, Docs).
    *   **Live Web Scraping:** Uses `crawl4ai` for real-time extraction from KIET web domains.
*   **🛡️ Smart Fallbacks & Consistency Checks:** Bypasses costly LLM invocations for known high-frequency queries (e.g., Timetables, Hostels) and utilizes a Consistency Check Agent to avoid hallucinated responses.
*   **📊 Comprehensive Admin Dashboard:** Secure UI for query analytics, live logs, dynamic FAQ CRUD operations, target URL management, and manual Vector DB re-indexing.
*   **🐳 Fully Dockerized:** Includes Docker and Docker Compose for out-of-the-box multi-container orchestration.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Web Framework** | FastAPI, Uvicorn, SSE (Server-Sent Events) |
| **AI / Orchestration** | LangChain, HuggingFace Transformers, PyTorch |
| **Vector Database** | FAISS (Local CPU/GPU Embeddings) |
| **Persistent Storage** | MongoDB (Analytics, Query Logs, FAQs) |
| **Web Scraping** | Crawl4AI, BeautifulSoup4, LXML |
| **Document Parsing** | PyMuPDF (fitz), PyPDF2 |

---

## 📂 Project Structure

```text
├── admin/                  # Admin logging and analytics handlers
├── agents/                 # Specialized autonomous agents
│   ├── orchestrator_agent.py    # Main coordinator
│   ├── faq_agent.py             # Rapid fuzzy-match response agent
│   ├── vectordb_agent.py        # Local Document RAG retrieval
│   ├── webscraper_agent.py      # Live website data extraction
│   ├── llm_agent.py             # Response generation (Local/HF LLMs)
│   ├── consistency_check_agent.py # Hallucination prevention
│   └── summarizing_agent.py     # Context summarization
├── config/                 # Project configurations and settings
│   ├── settings.py
│   └── urls.txt                 # Target domains for web scraping
├── data/                   # Persistent volume (Documents, FAISS indexes)
├── static/                 # Frontend HTML/CSS/JS (Chat & Admin UI)
├── utils/                  # Helper functions and utilities
├── main.py                 # FastAPI Application entry point
├── pipeline.py             # Orchestrator instantiation & initialization
├── docker-compose.yml      # Multi-container setup (App + MongoDB)
├── Dockerfile              # Docker image definition
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites

*   **Python 3.9+**
*   **MongoDB** (running locally on port `27017` or configured via `.env`)
*   *(Optional)* **CUDA Toolkit** if running models on GPU.

### 1. Clone & Setup Virtual Environment

```bash
git clone <repository_url>
cd <repository_directory>

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

*For CPU-only (default):*
```bash
pip install -r requirements.txt
```

*For GPU Support (CUDA 11.8):*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory. You can use the provided `.env.example` as a template:

```ini
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=20
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin1234
MONGO_URI=mongodb://localhost:27017
```

### 4. Run the Application

Start the FastAPI application via Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be accessible at:
*   **Chat UI:** [http://localhost:8000/chat](http://localhost:8000/chat)
*   **Admin Dashboard:** [http://localhost:8000/admin](http://localhost:8000/admin)
*   **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

The simplest way to run KIET EDU Assist in production is via Docker Compose, which spins up both the FastAPI application and the MongoDB database securely.

### 1. Setup Docker Environment
Ensure you have a `.env` file present at the root (you can copy from `.env.docker.example` if it exists).

### 2. Build and Start Services
```bash
docker-compose up --build -d
```

### 3. Verification
Check the container logs to ensure successful startup:
```bash
docker-compose logs -f app
```

*Note: The `data/` and `config/` folders are volume-mounted to persist your vector databases and URL configurations across restarts.*

---

## 👥 Admin Access

To access the administrative interface:
1. Navigate to `http://localhost:8000/admin`.
2. Login using the credentials defined in your `.env` file (`ADMIN_USERNAME` and `ADMIN_PASSWORD`).
3. Use the dashboard to monitor system health, view un-answered questions, edit the FAQ database, and manually trigger Vector DB re-indexing.

---

## 🤝 Contributing

When contributing to this project, please adhere to the modular architecture:
*   Add new capabilities by creating a new agent in the `agents/` directory.
*   Update the `OrchestratorAgent` to integrate new agents into the decision pipeline.
*   Ensure the Admin UI (`static/admin.html`) is updated if exposing new administrative endpoints.

---
<div align="center">
  <i>Developed with ❤️ for KIET University</i>
</div>
