import logging
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import TypedDict
from dotenv import load_dotenv
import os
import re


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphRAGState(TypedDict):
    query: str
    retrieved: str
    graph: str
    final_answer: str

def initialize_agentic_rag(vector_store, graph):
    logger.info("Initializing LLM with Ollama model: tinyllama")
    #llm = ChatOllama(model="tinyllama")

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

    logger.info("Creating retriever from vector store")
    retriever = vector_store.as_retriever()

    logger.info("Setting up RetrievalQA chain")
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    def run_retriever(state: GraphRAGState) -> GraphRAGState:
        query = state["query"]
        logger.info(f"Running retrieval for query: {query}")
        result = retrieval_chain.invoke({"query": query})
        state["retrieved"] = result["result"]
        return state

    def extract_keywords(text):
        stopwords = {"how", "is", "are", "the", "a", "an", "why", "what", "??", "?"}
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in stopwords]

    def query_neo4j(state: GraphRAGState) -> GraphRAGState:
        query = state["query"]
        keywords = extract_keywords(query)
        if not keywords:
            state["graph"] = "No keywords found in query."
            return state

        pattern = " OR ".join([f"e.name CONTAINS '{kw}'" for kw in keywords])
        cypher = f"MATCH (e) WHERE {pattern} RETURN e LIMIT 5"

        logger.info(f"Executing Neo4j query: {cypher}")
        try:
            results = graph.query(cypher)
            logger.info(f"Graph query returned {len(results)} result(s)")
            state["graph"] = str(results)
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            state["graph"] = "Graph query failed."
        return state


    logger.info("Creating merge prompt for combining retrieved and graph results")
    merge_prompt = PromptTemplate(
        input_variables=["retrieved", "graph"],
        template="""
        You are a helpful assistant. Merge the following two information sources:

        1. Retrieved Answer:
        {retrieved}

        2. Graph Insight:
        {graph}

        Provide a coherent, informative, and concise answer that integrates both sources.
        """
    )

    merge_chain = merge_prompt | llm

    def merge_results(state: GraphRAGState) -> GraphRAGState:
        logger.info("Merging retrieved and graph insights")
        final = merge_chain.invoke({"retrieved": state["retrieved"], "graph": state["graph"]})
        state["final_answer"] = final
        return state

    logger.info("Building LangGraph state graph")
    workflow = StateGraph(GraphRAGState)
    workflow.add_node("retrieve", RunnableLambda(run_retriever))
    workflow.add_node("graph_query", RunnableLambda(query_neo4j))
    workflow.add_node("merge", RunnableLambda(merge_results))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "graph_query")
    workflow.add_edge("graph_query", "merge")
    workflow.add_edge("merge", END)

    graph_executor = workflow.compile()

    def query_agentic_rag(query: str) -> str:
        logger.info(f"Received query: {query}")
        state = {"query": query}
        result = graph_executor.invoke(state)
        return result.get("final_answer", "No answer generated.")

    return query_agentic_rag