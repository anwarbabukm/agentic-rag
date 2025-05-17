
# Agentic GraphRAG

Agentic GraphRAG is a powerful RAG (Retrieval-Augmented Generation) application that combines vector similarity search with knowledge graph reasoning. It features a Streamlit UI for interactive querying, powered by LangChain agents running on top of Qwen via Ollama, with vector search through Qdrant and relational reasoning via Neo4j.

---

## 🧠 Architecture

```
                +-------------------------+
                |     Streamlit UI        |
                +-----------+-------------+
                            |
                            v
                +-------------------------+
                |    LangChain Agent      |
                |  (Qwen via Ollama)      |
                +-----------+-------------+
                            |
          +----------------+-------------------+
          |                                    |
+---------------------+         +------------------------+
|   Qdrant Vector DB  |         |     Neo4j Graph DB     |
|   (semantic docs)   |         | (entity relationships) |
+---------------------+         +------------------------+
```

---

## 📁 Folder Structure

```
agentic_graphrag/
├── streamlit_app.py
├── agent/
│   ├── vector_tool.py
│   ├── graph_tool.py
│   ├── loader.py           # PDF loader
│   ├── storage.py          # Build Qdrant + Neo4j
│   └── agent_runner.py
├── data/
│   └── pdfs/               # English PDFs
├── qdrant/                 # Vector setup
├── neo4j/                  # Graph setup
└── .env
```

---

## ⚙️ Configuration

The application supports both OpenAI and open-source Ollama models. Easily switch using the `provider` setting in `config.yaml`:

```yaml
provider: openai

llm:
  openai:
    model_name: gpt-4o
    temperature: 0.0
    max_tokens: 2048
    api_key: ${OPENAI_API_KEY}

  ollama:
    model_name: qwen2.5:7b-instruct
    temperature: 0.0
    format: json

embedding:
  openai:
    model_name: text-embedding-3-small
    dimensions: 1536
    api_key: ${OPENAI_API_KEY}

  ollama:
    model_name: nomic-embed-text
```

---

## 🚀 How to Run the Application

### 1. Clone the Repository
```bash
git clone https://github.com/anwarbabukm/agentic-rag.git
cd agentic-rag
```

### 2. Start Qdrant and Neo4j via Docker
Make sure the Docker Desktop is running in the background to run the services
```bash
# Qdrant
docker compose up -d build
```

### 3. Install Ollama and Pull Models
```bash
brew install ollama     # or refer to https://ollama.com/download
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Load Your Knowledge Base
Place PDFs inside `data/pdfs/`. The app will process and index them once you click on the 'Load Knowledge Base' in the UI.

### 6. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## ✨ Features

- ✅ RAG with combined Vector + Graph search
- ✅ LangChain Agentic orchestration with Qwen LLM
- ✅ Ollama for open-source local model inference
- ✅ Qdrant for fast semantic similarity
- ✅ Neo4j for knowledge graph queries
- ✅ Streamlit UI for easy document upload and querying

---

## 🧠 Context

This project demonstrates an **Agentic RAG** system using LangChain, with:
- Vector similarity powered by **Qdrant**
- Relational reasoning using **Neo4j**
- Multimodal ingestion via **PDF Loader**
- **LangChain agents** orchestrating tool use
- **Streamlit** interface to interactively load and query the system

---

## 📄 License

MIT License. Feel free to fork, extend, and contribute.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [Qdrant](https://qdrant.tech/)
- [Neo4j](https://neo4j.com/)
- [Streamlit](https://streamlit.io/)