#!/usr/bin/env python3
"""
Process a test document from the test_embeddings folder.
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.file_loader import load_file
from src.ingestion.document_processor import DocumentProcessor

def main():
    """Process the sample document and print results."""
    try:
        # Load the sample document
        file_path = "test_embeddings/sample.txt"
        logger.info(f"Loading sample document from {file_path}")
        
        file_data = load_file(file_path)
        logger.info(f"Loaded document: {file_data['title']}")
        
        # Process the document
        logger.info("Processing document...")
        result = DocumentProcessor.process_file_content(file_data)
        
        logger.info(f"Document processed successfully!")
        logger.info(f"Document ID: {result['document_id']}")
        logger.info(f"Chunks created: {result['chunk_count']}")
        
        # Check the database for the chunks
        from src.utils.database import supabase_manager
        
        logger.info(f"Checking for chunks in database...")
        chunks_response = supabase_manager.client.table("chunks").select("*").eq("document_id", result['document_id']).execute()
        chunks = chunks_response.data
        
        logger.info(f"Found {len(chunks)} chunks in database")
        
        if chunks:
            # Show info about the first chunk
            logger.info(f"First chunk content snippet: {chunks[0]['content'][:100]}...")
            
    except Exception as e:
        logger.error(f"Error processing sample document: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 