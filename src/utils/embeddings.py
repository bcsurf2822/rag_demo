"""
Module for generating embeddings using OpenAI.
"""
import logging
import os
from typing import List, Union

import openai
from openai import OpenAI

from src.utils.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Class for generating embeddings using OpenAI's API.
    """
    
    def __init__(self):
        """Initialize the embedding generator with API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided as OPENAI_API_KEY environment variable.")
        
        # Set up the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Default embedding dimension for text-embedding-3-small
        self.embedding_dim = OPENAI_EMBEDDING_DIMENSIONS
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text using OpenAI's API.
        
        Args:
            text: The text to generate embeddings for.
            
        Returns:
            A list of floats representing the embedding vector.
            
        Raises:
            ValueError: If the text is empty or the embeddings generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Normalize text by removing excessive whitespace
        text = " ".join(text.split())
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}")
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: A list of texts to generate embeddings for.
            
        Returns:
            A list of embedding vectors, each being a list of floats.
            
        Raises:
            ValueError: If the text list is empty or the embeddings generation fails.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and normalize whitespace
        filtered_texts = [" ".join(text.split()) for text in texts if text and text.strip()]
        
        if not filtered_texts:
            raise ValueError("All texts were empty after filtering")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=filtered_texts
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise ValueError(f"Failed to generate batch embeddings: {str(e)}")

# For backward compatibility, create a single global instance
_embedding_generator = None

def _get_embedding_generator():
    """
    Get or create the global embedding generator instance.
    
    Returns:
        An instance of EmbeddingGenerator.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    global _embedding_generator
    if _embedding_generator is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        try:
            _embedding_generator = EmbeddingGenerator()
        except Exception as e:
            logger.error(f"Error initializing EmbeddingGenerator: {str(e)}")
            raise
    return _embedding_generator

# Maintain backward compatibility with existing function calls
def generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings for text using OpenAI's API.
    
    This is a wrapper around EmbeddingGenerator.generate_embedding for backward compatibility.
    
    Args:
        text: The text to generate embeddings for.
        
    Returns:
        A list of floats representing the embedding vector.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set or the embeddings generation fails.
    """
    return _get_embedding_generator().generate_embedding(text)

def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch.
    
    This is a wrapper around EmbeddingGenerator.generate_batch_embeddings for backward compatibility.
    
    Args:
        texts: A list of texts to generate embeddings for.
        
    Returns:
        A list of embedding vectors, each being a list of floats.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set or the embeddings generation fails.
    """
    return _get_embedding_generator().generate_batch_embeddings(texts) 