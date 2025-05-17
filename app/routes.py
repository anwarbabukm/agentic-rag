import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_loader import load_and_chunk_pdf
from app.embedding_utils import generate_qdrant_store
from app.agents import initialize_agentic_rag
from langchain_neo4j import Neo4jGraph

from app.embedding_utils import extract_and_store_triplets

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------------------
# PDF Loading & Chunking
# -------------------------------
logger.info("Starting PDF loading and chunking process...")
all_chunks = []
pdf_dir = "data/pdfs"

for file_name in os.listdir(pdf_dir):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, file_name)
        logger.info(f"Processing PDF: {file_path}")
        try:
            chunks = load_and_chunk_pdf(file_path)
            all_chunks.extend(chunks)
            logger.info(f"Added {len(chunks)} chunk(s) from: {file_name}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

logger.info(f"Total chunks generated from all PDFs: {len(all_chunks)}")

# -------------------------------
# Embedding + Qdrant Vector Store
# -------------------------------
logger.info("Generating embeddings and storing into Qdrant...")
vector_store = generate_qdrant_store(all_chunks)
logger.info("Qdrant vector store initialized")

# -------------------------------
# Neo4j Graph Setup
# -------------------------------
logger.info("Connecting to Neo4j...")
try:
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="test1234"
    )
    logger.info("Neo4j connection established")

    extract_and_store_triplets(all_chunks, graph)
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    raise

# -------------------------------
# Initialize Agentic GraphRAG
# -------------------------------
logger.info("Initializing Agentic GraphRAG pipeline...")
query_agentic_rag = initialize_agentic_rag(vector_store, graph)
logger.info("Agentic RAG system ready")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    logger.info(f"Incoming question: {request.question}")
    try:
        answer = query_agentic_rag(request.question)
        logger.info("Answer generated successfully")
        return {"answer": answer.content}
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        return {"answer": "An error occurred while processing your question."}
    
