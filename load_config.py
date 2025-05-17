import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    provider = config["provider"]
    llm_config = config["llm"][provider]
    embed_config = config["embedding"][provider]
    return provider, llm_config, embed_config