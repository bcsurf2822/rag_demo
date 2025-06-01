#!/usr/bin/env python3
"""
Main module for the RAG AI Agent.
"""
import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

from src.agent.rag_agent import RAGAgent
from src.ingestion.document_processor import DocumentProcessor
from src.utils.file_loader import extract_file_content, load_text_file, load_pdf_file
from src.utils.database import supabase_manager
from src.database.init_database import initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def ingest_document(file_path: str, title: Optional[str] = None) -> dict:
    """
    Ingest a document into the system.
    
    Args:
        file_path: Path to the document file.
        title: Title for the document (if not provided, will use filename).
        
    Returns:
        Dictionary containing information about the ingested document.
    """
    try:
        # Extract content
        content = extract_file_content(file_path)
        
        # Use filename as title if not provided
        if not title:
            title = os.path.basename(file_path)
        
        # Process the document
        doc_info = DocumentProcessor.process_document(
            content=content,
            title=title,
            filename=os.path.basename(file_path)
        )
        
        logger.info(f"Ingested document '{title}' with ID: {doc_info['document_id']}")
        logger.info(f"Created {doc_info['chunk_count']} chunks")
        
        return doc_info
        
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise

async def query_agent(query: str) -> dict:
    """
    Query the RAG agent with a question.
    
    Args:
        query: The question to ask.
        
    Returns:
        The agent's response.
    """
    try:
        # Create the RAG agent
        agent = RAGAgent()
        
        # Get the answer
        response = await agent.answer_question(query)
        
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }
        
    except Exception as e:
        logger.error(f"Error querying agent: {e}")
        raise

async def main():
    """
    Main function for the RAG AI Agent.
    """
    parser = argparse.ArgumentParser(description="RAG AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", help="Path to the document file")
    ingest_parser.add_argument("--title", help="Title for the document")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the agent")
    query_parser.add_argument("question", help="The question to ask")
    
    args = parser.parse_args()
    
    # Check if Supabase connection is working
    try:
        client = supabase_manager.client
        logger.info("Successfully connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return
    
    if args.command == "init":
        # Initialize the database
        success = initialize_database()
        if success:
            logger.info("Database initialized successfully")
        else:
            logger.error("Failed to initialize database")
            
    elif args.command == "ingest":
        # Ingest a document
        try:
            doc_info = await ingest_document(args.file, args.title)
            print(f"Document ingested successfully. ID: {doc_info['document_id']}")
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            
    elif args.command == "query":
        # Query the agent
        try:
            response = await query_agent(args.question)
            print("\nAnswer:")
            print(response["answer"])
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")
            print(f"\nConfidence: {response['confidence']}")
            
        except Exception as e:
            logger.error(f"Failed to query agent: {e}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 