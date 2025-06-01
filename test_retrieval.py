#!/usr/bin/env python3
"""
Test script for retrieving information from an embedded document.
"""
import logging
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI model configuration to ensure compatibility
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"  # Use gpt-4o-mini for chat
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"  # Use text-embedding-3-small for embeddings

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
from src.retrieval.vector_search import VectorSearch
from src.agent.rag_agent import RAGAgent

async def test_retrieval():
    """Test document retrieval and question answering."""
    try:
        # 1. First, ingest a sample document
        file_path = "sample_docs/sample.txt"
        if not os.path.exists(file_path):
            logger.error(f"Sample document not found at {file_path}")
            logger.info("Please create a sample document or specify a different path")
            return
        
        logger.info(f"Loading and processing sample document from {file_path}")
        file_data = load_file(file_path)
        logger.info(f"Loaded document: {file_data['title']}")
        
        # Process the document
        logger.info("Processing document...")
        result = DocumentProcessor.process_file_content(file_data)
        
        logger.info(f"Document processed successfully!")
        logger.info(f"Document ID: {result['document_id']}")
        logger.info(f"Chunks created: {result['chunk_count']}")
        
        # 2. Now test retrieval with a simple query
        test_query = "What are the benefits of RAG systems?"
        logger.info(f"Testing retrieval with query: '{test_query}'")
        
        # Get relevant chunks
        search_results = VectorSearch.search(test_query, match_count=3, match_threshold=0.3)
        logger.info(f"Retrieved {len(search_results)} chunks from vector search")
        
        if search_results:
            # Show the first result
            logger.info(f"Best match (similarity: {search_results[0]['similarity']:.4f}):")
            logger.info(f"Content: {search_results[0]['content'][:150]}...")
        
        # 3. Test the RAG agent with the same query
        logger.info("Testing RAG agent...")
        rag_agent = RAGAgent()
        response = await rag_agent.answer_question(
            question=test_query,
            match_count=3,
            match_threshold=0.3
        )
        
        logger.info(f"RAG Agent Response:")
        logger.info(f"Answer: {response.answer}")
        logger.info(f"Confidence: {response.confidence}")
        logger.info(f"Sources: {response.sources}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during retrieval test: {e}")
        return None

if __name__ == "__main__":
    # Run the async test
    response = asyncio.run(test_retrieval())
    
    if response:
        print("\n----- Test Completed Successfully -----")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Sources Used: {', '.join(response.sources)}")
    else:
        print("\n----- Test Failed -----") 