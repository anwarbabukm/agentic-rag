from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
import os

def initialize_llm(provider, config):
    if provider == "openai":
        return ChatOpenAI(
            model=config["model_name"],
            temperature=config["temperature"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "ollama":
        return ChatOllama(
            model=config["model_name"],
            temperature=config["temperature"])
    else:
        raise ValueError(f"Unsupported provider: {provider}")