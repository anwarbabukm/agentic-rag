from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from agent.vector_tool import vector_search_tool
from agent.graph_tool import graph_search_tool
from dotenv import load_dotenv
import os

from load_config import load_config
from agent.init_llm import initialize_llm

# Load configuration
provider, llm_config, embedding_config = load_config()

# Initialize models
llm = initialize_llm(provider, llm_config)

print("llm:", llm)

tools = [
    Tool(
        func=vector_search_tool,
        name="VectorSearch",
        description="Use for semantic questions that can be answered from documents"
    ),
    Tool(
        func=graph_search_tool,
        name="GraphSearch",
        description="Use for relationship queries between concepts"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def answer_query(query):
    return agent.run(query)