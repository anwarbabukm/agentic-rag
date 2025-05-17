import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agent.graph_tool import extract_and_store_triplets
from agent.vector_tool import generate_qdrant_store
from langchain_neo4j import Neo4jGraph

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    logger.info(f"Loading PDF from path: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {pdf_path}")
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        raise

    logger.info(f"Splitting documents into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunk(s) from {pdf_path}")

    return chunks

# -------------------------------
# PDF Loading & Chunking
# -------------------------------

def load_and_ingest():

    logger.info("Starting PDF loading and chunking process...")
    all_chunks = []
    pdf_dir = "data/pdfs"

    for file_name in os.listdir(pdf_dir):
        print(file_name)
        
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
    generate_qdrant_store(all_chunks)
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

    
    return "success"
