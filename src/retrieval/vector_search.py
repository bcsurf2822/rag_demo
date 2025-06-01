"""
Vector search module for retrieving relevant document chunks.
"""
import logging
from typing import List, Dict, Any, Optional

from src.utils.database import supabase_manager
from src.utils.embeddings import generate_embedding
from src.utils.config import MATCH_COUNT, MATCH_THRESHOLD

logger = logging.getLogger(__name__)

class VectorSearch:
    """
    Class for performing vector searches against the document chunks.
    """
    
    @staticmethod
    def search(query: str, match_count: int = MATCH_COUNT, 
              match_threshold: float = MATCH_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Search for document chunks relevant to the query.
        
        Args:
            query: The search query.
            match_count: The maximum number of matches to return.
            match_threshold: The similarity threshold (0-1).
            
        Returns:
            A list of matching document chunks with similarity scores.
        """
        logger.info(f"Searching for: '{query}'")
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        logger.info(f"Query embedding (first 5): {query_embedding[:5]}")
        
        # Search for similar chunks
        results = supabase_manager.query_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=match_threshold,
            match_count=match_count
        )
        
        logger.info(f"Found {len(results)} matching chunks")
        
        if results:
            logger.info(f"First result similarity: {results[0].get('similarity')}, content: {results[0].get('content')[:100]}")
        
        return results
    
    @staticmethod
    def format_results_for_context(results: List[Dict[str, Any]]) -> str:
        """
        Format search results as a context string for the AI model.
        
        Args:
            results: The search results from the vector search.
            
        Returns:
            A formatted context string.
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            document_id = result.get("document_id")
            similarity = result.get("similarity", 0)
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            title = metadata.get("title", "Unknown")
            
            # Format each chunk with its source and content
            context_part = f"[{i+1}] From '{title}' (similarity: {similarity:.2f}):\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    @staticmethod
    def retrieve_context(query: str, match_count: int = MATCH_COUNT, 
                        match_threshold: float = MATCH_THRESHOLD) -> str:
        """
        Retrieve and format context for a query.
        
        Args:
            query: The search query.
            match_count: The maximum number of matches to return.
            match_threshold: The similarity threshold (0-1).
            
        Returns:
            A formatted context string.
        """
        results = VectorSearch.search(query, match_count, match_threshold)
        return VectorSearch.format_results_for_context(results) 