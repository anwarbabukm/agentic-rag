from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings, OpenAIEmbeddings
import os

def initialize_embedding(provider, config):
    if provider == "openai":
        return OpenAIEmbeddings(
            model=config["model_name"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    elif provider == "ollama":
        return OllamaEmbeddings(
            model=config["model_name"]
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")