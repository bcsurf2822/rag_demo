#!/usr/bin/env python3
"""
Script to clear the chunks table and re-ingest documents with proper embeddings.
"""
import logging
import os
import sys
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.database import supabase_manager
from src.ingestion.document_processor import DocumentProcessor
from src.utils.file_loader import load_file

def clear_chunks_table():
    """Delete all records from the chunks table."""
    try:
        # Execute SQL to clear the chunks table
        response = supabase_manager.client.table("chunks").delete().neq("id", 0).execute()
        logger.info(f"Successfully cleared chunks table. Deleted {len(response.data)} records.")
        return True
    except Exception as e:
        logger.error(f"Error clearing chunks table: {e}")
        return False

def get_all_documents():
    """Get all documents from the documents table."""
    try:
        response = supabase_manager.client.table("documents").select("*").execute()
        logger.info(f"Retrieved {len(response.data)} documents from database.")
        return response.data
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def validate_embeddings():
    """Validate that embeddings are properly stored."""
    try:
        # Check a few chunks to ensure embeddings are properly stored
        response = supabase_manager.client.table("chunks").select("embedding").limit(3).execute()
        if not response.data:
            logger.warning("No chunks found to validate.")
            return False
        
        for i, chunk in enumerate(response.data):
            embedding = chunk.get("embedding", [])
            # Check if embedding is a list with valid length
            if not isinstance(embedding, list) or len(embedding) < 10:
                logger.error(f"Validation failed: Chunk {i} has invalid embedding format: {embedding[:5]}...")
                return False
            
            # Check if embedding values are diverse (not all the same)
            first_values = embedding[:5]
            if all(abs(v - first_values[0]) < 0.0001 for v in first_values[1:]):
                logger.warning(f"Suspicious embedding pattern - all similar values: {first_values}")
                return False
            
            logger.info(f"Chunk {i} embedding sample (first 5): {first_values}")
            
        logger.info("Embedding validation passed - embeddings look valid.")
        return True
    except Exception as e:
        logger.error(f"Error validating embeddings: {e}")
        return False

def reprocess_document(doc: Dict[str, Any]):
    """
    Reprocess a document using its ID and metadata.
    
    Args:
        doc: The document record from the database.
    
    Returns:
        Success status (boolean)
    """
    try:
        doc_id = doc["id"]
        title = doc["title"]
        filename = doc["filename"]
        
        logger.info(f"Reprocessing document {doc_id}: {title}")
        
        # Since we don't have the original file content in the database,
        # we need to either reload the file or use a different approach
        
        # Option 1: If files are stored in a known location
        try:
            # Try to load the file from a known location (if applicable)
            file_path = f"test_embeddings/{filename}"
            file_data = load_file(file_path)
            result = DocumentProcessor.process_file_content(file_data)
            logger.info(f"Reprocessed document {doc_id} with {result['chunk_count']} chunks")
            return True
        except Exception as file_error:
            logger.warning(f"Could not load file from disk: {file_error}")
            
            # Option 2: Create a placeholder chunk if original content is unavailable
            logger.warning(f"Creating a placeholder chunk for document {doc_id}: {title}")
            placeholder_content = f"Document: {title}\n\nThis is a placeholder chunk. Original content not available."
            
            # Generate embedding for the placeholder content
            from src.utils.embeddings import generate_embedding
            embedding = generate_embedding(placeholder_content)
            
            # Store the placeholder chunk
            metadata = {
                "document_id": doc_id,
                "title": title,
                "filename": filename,
                "is_placeholder": True
            }
            
            supabase_manager.store_chunk(
                document_id=doc_id,
                content=placeholder_content,
                embedding=embedding,
                metadata=metadata
            )
            
            logger.info(f"Created placeholder chunk for document {doc_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error reprocessing document {doc.get('id', 'unknown')}: {e}")
        return False

def main():
    """Main function to clear and reingest documents."""
    logger.info("Starting the clear and reingest process")
    
    # Step 1: Clear the chunks table
    if not clear_chunks_table():
        logger.error("Failed to clear chunks table. Exiting.")
        sys.exit(1)
    
    # Step 2: Get all documents
    documents = get_all_documents()
    if not documents:
        logger.error("No documents found in the database. Exiting.")
        sys.exit(1)
    
    # Step 3: Reprocess each document
    success_count = 0
    for doc in documents:
        if reprocess_document(doc):
            success_count += 1
    
    logger.info(f"Reprocessed {success_count} out of {len(documents)} documents")
    
    # Step 4: Validate the new embeddings
    if validate_embeddings():
        logger.info("Embeddings validation passed. Re-ingestion completed successfully.")
    else:
        logger.error("Embeddings validation failed. Please check the embedding generation process.")

if __name__ == "__main__":
    main() 