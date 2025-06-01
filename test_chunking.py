#!/usr/bin/env python3
"""
Test script for validating the improved document chunking implementation.
"""
import logging
import os
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL
from src.utils.text_chunking import chunk_text_with_metadata, get_token_count
from src.utils.embeddings import generate_embedding, generate_batch_embeddings
from src.utils.database import supabase_manager
from src.ingestion.document_processor import DocumentProcessor

def generate_test_document() -> Dict[str, Any]:
    """
    Generate a test document for chunking validation.
    
    Returns:
        Dictionary containing the document data.
    """
    # Create test document with repeated paragraphs to reach sufficient length
    base_paragraph = """
    This is a test paragraph for document chunking verification. 
    It contains sentences of various lengths to test the sentence boundary detection.
    Can it properly handle questions? Yes, it should be able to!
    And what about exclamation marks! Those are important sentence boundaries too.
    
    Each paragraph should be treated as a semantic unit if possible, 
    and the chunking algorithm should try to preserve paragraph boundaries.
    """
    
    paragraphs = []
    for i in range(100):  # Create 100 paragraphs for a large document
        paragraphs.append(f"Paragraph {i+1}: {base_paragraph}")
    
    content = "\n\n".join(paragraphs)
    
    return {
        "content": content,
        "title": "Test Document for Chunking",
        "filename": "test_chunking.txt"
    }

def test_chunking() -> None:
    """
    Test the chunking implementation with a generated document.
    """
    logger.info("Generating test document")
    test_doc = generate_test_document()
    
    logger.info(f"Document length: {len(test_doc['content'])} characters")
    logger.info(f"Approximate token count: {get_token_count(test_doc['content'])}")
    
    # Test chunking with metadata
    metadata = {
        "title": test_doc["title"],
        "filename": test_doc["filename"],
        "test": True
    }
    
    logger.info(f"Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    chunks = chunk_text_with_metadata(test_doc["content"], metadata)
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Show token counts for first few chunks
    for i, chunk in enumerate(chunks[:5]):
        token_count = get_token_count(chunk["content"])
        logger.info(f"Chunk {i}: {token_count} tokens, {len(chunk['content'])} characters")
    
    # Test batch embedding generation
    logger.info("Generating embeddings for chunks")
    chunk_texts = [chunk["content"] for chunk in chunks[:5]]  # Just use first 5 for the test
    
    try:
        embeddings = generate_batch_embeddings(chunk_texts)
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimensions: {len(embeddings[0])}")
        
        # Test storing a chunk in the database
        try:
            # Store a test document
            logger.info("Creating test document in database")
            document = supabase_manager.store_document(test_doc["title"], test_doc["filename"])
            document_id = document["id"]
            logger.info(f"Created document with ID: {document_id}")
            
            # Store one chunk with its embedding
            logger.info("Storing a test chunk with embedding")
            chunk_content = chunks[0]["content"]
            chunk_metadata = chunks[0]["metadata"]
            chunk_metadata["document_id"] = document_id  # Set the document ID
            
            stored_chunk = supabase_manager.store_chunk(
                document_id=document_id,
                content=chunk_content,
                embedding=embeddings[0],
                metadata=chunk_metadata
            )
            
            logger.info(f"Successfully stored chunk with ID: {stored_chunk['id']}")
            logger.info("Chunking and embedding test successful!")
        
        except Exception as e:
            logger.error(f"Error storing chunk in database: {e}")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def test_document_processing() -> None:
    """
    Test the full document processing pipeline.
    """
    logger.info("Testing document processing with chunking and embeddings")
    test_doc = generate_test_document()
    
    try:
        result = DocumentProcessor.process_file_content(test_doc)
        logger.info(f"Document processed with ID: {result['document_id']}")
        logger.info(f"Created {result['chunk_count']} chunks")
        logger.info("Document processing test successful!")
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")

if __name__ == "__main__":
    logger.info("Starting chunking validation tests")
    
    try:
        # Run chunking test
        test_chunking()
        
        # Run document processing test
        test_document_processing()
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    
    logger.info("All tests completed") 