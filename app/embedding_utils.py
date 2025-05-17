import logging
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
import ast

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def insert_triplets_to_neo4j(triplets: list[tuple[str, str, str]], graph: Neo4jGraph):
    for head, rel, tail in triplets:
        try:
            # Escape quotes to prevent Cypher errors
            head = head.replace("'", "\\'")
            tail = tail.replace("'", "\\'")
            rel = rel.replace("'", "\\'")

            cypher = f"""
            MERGE (h:Entity {{name: '{head}'}})
            MERGE (t:Entity {{name: '{tail}'}})
            MERGE (h)-[:`{rel.upper()}`]->(t)
            """
            logger.info(f"Inserting triplet into Neo4j: ({head})-[:{rel.upper()}]->({tail})")
            graph.query(cypher)

        except Exception as e:
            logger.error(f"Error inserting triplet ({head}, {rel}, {tail}): {e}")


def extract_and_store_triplets(documents, graph):
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

    extraction_prompt = PromptTemplate.from_template(
        """
        Extract a list of (entity1, relationship, entity2) triples from the following text:

        {text}

        Strictly provide with the given format Format: [("Entity1", "relation", "Entity2"), ...]
        """
    )
    llm_chain = extraction_prompt | llm

    for doc in documents:
        try:
            logger.info("Extracting triples from chunk")
            triplets_output = llm_chain.invoke({"text": doc.page_content})
            logger.info(f"LLM raw output: {triplets_output}")

            # triplets = triplets_output.content

            triplets_str = triplets_output.content  # Still a string
            triplets = ast.literal_eval(triplets_str)  # ✅ Parse it into Python list

            
            # ✅ Filter only valid triplets
            valid_triplets = [
                t for t in triplets
                if isinstance(t, (list, tuple)) and len(t) == 3
            ]

            if not valid_triplets:
                logger.warning("No valid triplets extracted from chunk")
                continue

            insert_triplets_to_neo4j(valid_triplets, graph)

        except Exception as e:
            logger.error(f"Failed to extract or store triplets: {e}")

def generate_qdrant_store(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", collection="graphrag_knowledge_base"):
    logger.info(f"Initializing embedding model: {model_name}")
    #embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)



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

    return qdrant