#!/usr/bin/env python3
"""
Script to directly test OpenAI's embedding and chat models.
"""
import logging
import os
import sys
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.embeddings import generate_embedding
from openai import OpenAI
from src.utils.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL

def test_embedding_model(texts: List[str]):
    """
    Test the OpenAI embedding model on multiple texts.
    
    Args:
        texts: List of text samples to embed.
    """
    logger.info(f"Testing embedding model: {OPENAI_EMBEDDING_MODEL}")
    
    for i, text in enumerate(texts):
        try:
            logger.info(f"Sample {i+1}: '{text[:50]}...'")
            embedding = generate_embedding(text)
            
            # Verify embedding format and content
            if not embedding or not isinstance(embedding, list):
                logger.error(f"Invalid embedding format: {type(embedding)}")
                continue
                
            # Log more details
            logger.info(f"Embedding length: {len(embedding)}")
            logger.info(f"First 5 values: {embedding[:5]}")
            logger.info(f"Last 5 values: {embedding[-5:]}")
            
            # Check for non-zero values
            if all(abs(v) < 0.0001 for v in embedding[:10]):
                logger.warning("Warning: First 10 embedding values are all near zero")
            
            # Check for diversity
            if all(abs(v - embedding[0]) < 0.0001 for v in embedding[1:5]):
                logger.warning("Warning: First 5 embedding values are all similar")
            else:
                logger.info("Embedding looks diverse and valid")
                
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error generating embedding for sample {i+1}: {e}")

def test_chat_model(messages: List[dict]):
    """
    Test the OpenAI chat model with a conversation.
    
    Args:
        messages: List of message dictionaries.
    """
    logger.info(f"Testing chat model: {OPENAI_MODEL}")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        
        logger.info(f"Chat response: {response.choices[0].message.content}")
        logger.info(f"Usage: {response.usage.total_tokens} tokens")
        
    except Exception as e:
        logger.error(f"Error using chat model: {e}")

def main():
    """Main function to test both models."""
    logger.info("Starting direct model testing")
    
    # Sample texts for embedding
    texts = [
        "This is a test document about vector embeddings and semantic search.",
        "KLoBot Inc - Developer - Paid Internship Agreement terms and conditions.",
        "Document chunking is a critical component of our RAG system.",
        "Intern agrees to adhere by all the policies, procedures, rules and regulations set forth by KLoBot.",
        "The quick brown fox jumps over the lazy dog."  # Classic diverse text
    ]
    
    # Test embedding model
    test_embedding_model(texts)
    
    # Sample conversation for chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key components of a RAG (Retrieval Augmented Generation) system?"}
    ]
    
    # Test chat model
    test_chat_model(messages)
    
    logger.info("Testing completed.")

if __name__ == "__main__":
    main() 