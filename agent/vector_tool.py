import logging
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

from load_config import load_config
from agent.init_llm import initialize_llm
from agent.init_embedding import initialize_embedding

# Load configuration
provider, llm_config, embedding_config = load_config()

# Initialize models
llm = initialize_llm(provider, llm_config)
embedding_model = initialize_embedding(provider, embedding_config)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_qdrant_store(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", collection="graphrag_knowledge_base"):
    logger.info(f"Initializing embedding model: {model_name}")

    logger.info("Connecting to Qdrant client at localhost:6333")
    qdrant_client = QdrantClient(host="localhost", port=6333)

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    logger.info(f"Preparing to embed {len(texts)} document(s)")

    try:
        vectors = embedding_model.embed_documents(texts)
        logger.info("Document embeddings generated successfully")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

    logger.info(f"Recreating Qdrant collection: {collection} with vector size {len(vectors[0])}")
    qdrant_client.recreate_collection(
        collection_name=collection,
        vectors_config={"size": len(vectors[0]), "distance": "Cosine"}
    )

    logger.info(f"Initializing Qdrant vector store for collection: {collection}")
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=collection,
        embeddings=embedding_model
    )

    logger.info(f"Adding {len(texts)} vectors to Qdrant collection: {collection}")
    qdrant.add_texts(texts=texts, metadatas=metadatas)
    logger.info("Documents successfully stored in Qdrant")

def vector_search_tool(query: str) -> str:

    qdrant_client = QdrantClient(host="localhost", port=6333)

    qdrant = Qdrant(
    client=qdrant_client,
    collection_name="graphrag_knowledge_base",
    embeddings=embedding_model
    )

    docs = qdrant.similarity_search(query, k=3)
    
    return "\n".join([doc.page_content for doc in docs])