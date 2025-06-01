"""
Configuration module to load environment variables.
"""
import os
from typing import Union, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = 1536

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Text Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# RAG Configuration
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.7"))

def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Get an environment variable or raise an exception if it's not set.
    
    Args:
        var_name: The name of the environment variable.
        default: Optional default value if the variable is not set.
        
    Returns:
        The environment variable value.
        
    Raises:
        ValueError: If the environment variable is not set and no default is provided.
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value 