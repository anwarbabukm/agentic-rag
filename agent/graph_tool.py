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

# Load configuration
provider, llm_config, embedding_config = load_config()

# Initialize models
llm = initialize_llm(provider, llm_config)

# Entity extraction prompt template
entity_prompt = PromptTemplate.from_template(
    "Extract the main entity (place, person, object, or concept) from this user query: '{query}'\Strictly return a sinlge entity name."
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test1234"))

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


def graph_search_tool(prompt: str) -> str:
    try:
        # Chain to extract entity
        # entity_chain = llm | entity_prompt

        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

        print("Invoking LLM to find the entity from the prompt.")

        # Step 1: Extract entity using LangChain + OpenAI
        response = entity_chain.invoke({"query": prompt})
        print(response)
        entity = response["text"].strip()

        # Step 2: Safely query Neo4j using parameterized Cypher
        with driver.session() as session:
            query = """
            MATCH (n)-[r]->(m)
            WHERE toLower(n.name) = toLower($entity)
            RETURN type(r) as relation, m.name as target
            LIMIT 5
            """
            results = session.run(query, {"entity": entity})
            output = [f"{entity} --[{r['relation']}]--> {r['target']}" for r in results]

            return "\n".join(output) if output else f"No relations found for '{entity}'."
    except Exception as e:
        return f"Error: {str(e)}"
