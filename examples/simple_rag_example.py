#!/usr/bin/env python3
"""
Simple example of using the RAG system.
"""
import asyncio
import logging
import os
import sys

# Add the parent directory to the path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.rag_agent import RAGAgent
from src.ingestion.document_processor import DocumentProcessor
from src.utils.file_loader import extract_file_content
from src.utils.database import supabase_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """
    Main function to demonstrate the RAG system.
    """
    # Check if Supabase connection is working
    try:
        client = supabase_manager.client
        logger.info("Successfully connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return

    # Example: Ingest a document
    sample_doc_path = "../sample_docs/sample.txt"
    if os.path.exists(sample_doc_path):
        try:
            # Extract content
            content = extract_file_content(sample_doc_path)
            
            # Process the document
            doc_info = DocumentProcessor.process_document(
                content=content,
                title="Sample Document",
                filename=os.path.basename(sample_doc_path)
            )
            
            logger.info(f"Ingested document with ID: {doc_info['document_id']}")
            logger.info(f"Created {doc_info['chunk_count']} chunks")
            
            # Create the RAG agent
            agent = RAGAgent()
            
            # Example query
            query = "What is the main topic of the document?"
            logger.info(f"Querying: {query}")
            
            # Get the answer
            response = await agent.answer_question(query)
            
            logger.info(f"Answer: {response.answer}")
            logger.info(f"Sources: {response.sources}")
            logger.info(f"Confidence: {response.confidence}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
    else:
        logger.warning(f"Sample document not found at {sample_doc_path}")
        logger.info("You can create a sample document or specify a different path")

if __name__ == "__main__":
    asyncio.run(main()) 