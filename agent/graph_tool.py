from neo4j import GraphDatabase
import logging
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import ast

from load_config import load_config
from agent.init_llm import initialize_llm

# Load environment and config
load_dotenv()

provider, llm_config, embedding_config = load_config()
llm = initialize_llm(provider, llm_config)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Neo4j driver
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test1234"))

# Prompt to extract main entity
entity_prompt = PromptTemplate.from_template(
    "Extract the main entity (place, person, object, or concept) from this user query: '{query}'\nStrictly return a single entity name."
)


def insert_triplets_to_neo4j(triplets: list[tuple[str, str, str]], graph: Neo4jGraph):
    for head, rel, tail in triplets:
        try:
            head = head.replace("'", "\\'")
            tail = tail.replace("'", "\\'")
            rel = rel.replace("'", "\\'")
            
            # Simple heuristics: e.g., assign type based on relationship
            head_type = "Concept"
            tail_type = "Concept"

            cypher = f"""
            MERGE (h:Entity {{name: '{head}'}})
            ON CREATE SET h.type = '{head_type}'
            MERGE (t:Entity {{name: '{tail}'}})
            ON CREATE SET t.type = '{tail_type}'
            MERGE (h)-[:`{rel.upper()}`]->(t)
            """
            logger.info(f"Inserting triplet: ({head})-[:{rel.upper()}]->({tail})")
            graph.query(cypher)
        except Exception as e:
            logger.error(f"Error inserting triplet ({head}, {rel}, {tail}): {e}")


def extract_and_store_triplets(documents, graph):
    logger.info("Starting extraction of triplets from document chunks")
    
    extraction_prompt = PromptTemplate.from_template(
        """
        Extract a list of (entity1, relationship, entity2) triples from the following text:

        {text}

        Strictly provide with the format: [("Entity1", "relation", "Entity2"), ...]
        """
    )
    llm_chain = extraction_prompt | llm

    for doc in documents:
        try:
            logger.info("Invoking LLM to extract triples...")
            triplets_output = llm_chain.invoke({"text": doc.page_content})
            logger.info(f"LLM raw output: {triplets_output}")
            
            triplets_str = triplets_output.content
            logger.info(f"Parsing triplet string to Python list: {triplets_str}")
            triplets = ast.literal_eval(triplets_str)

            valid_triplets = [
                t for t in triplets
                if isinstance(t, (list, tuple)) and len(t) == 3
            ]

            logger.info(f"Extracted {len(valid_triplets)} valid triplet(s)")
            if not valid_triplets:
                logger.warning("No valid triplets found in this chunk.")
                continue

            insert_triplets_to_neo4j(valid_triplets, graph)

        except Exception as e:
            logger.error(f"Failed to extract or store triplets: {e}")

def graph_search_tool(prompt: str) -> str:
    try:
        logger.info(f"Received user query: {prompt}")
        logger.info("Initializing LLMChain to extract main entity...")

        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

        logger.info("Invoking LLM to extract main entity...")
        response = entity_chain.invoke({"query": prompt})
        entity = response["text"].strip()
        logger.info(f"Extracted entity: {entity}")

        logger.info("Querying Neo4j for relationships...")
        
        with driver.session() as session:
            query = """
            MATCH (n)-[r]->(m)
            WHERE toLower(n.name) = toLower($entity)
            RETURN type(r) as relation, m.name as target
            LIMIT 5
            """
            results = session.run(query, {"entity": entity})
            output = [f"{entity} --[{r['relation']}]--> {r['target']}" for r in results]

        if output:
            logger.info(f"Found {len(output)} relationship(s) for entity: {entity}")
        else:
            logger.warning(f"No relationships found for entity: {entity}")

        return "\n".join(output) if output else f"No relations found for '{entity}'."

    except Exception as e:
        logger.error(f"Error in graph_search_tool: {e}")
        return f"Error: {str(e)}"