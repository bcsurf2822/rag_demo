"""
Multi-query expansion module for improving retrieval coverage.
"""
import logging
from typing import List, Dict, Any, Set
import asyncio
from openai import AsyncOpenAI

from src.utils.config import OPENAI_API_KEY, OPENAI_MODEL
from src.retrieval.vector_search import VectorSearch

logger = logging.getLogger(__name__)

class QueryExpansion:
    """
    Class for expanding queries to improve retrieval coverage.
    """
    
    def __init__(self):
        """Initialize the query expansion with OpenAI client."""
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    async def generate_related_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """
        Generate related queries to expand search coverage.
        
        Args:
            original_query: The original search query.
            num_queries: Number of related queries to generate.
            
        Returns:
            List of generated related queries.
        """
        logger.info(f"Generating {num_queries} related queries for: '{original_query}'")
        
        prompt = f"""
        Given the following question: "{original_query}"
        
        Generate {num_queries} related questions that would help find relevant information to answer the original question. 
        
        The related questions should:
        1. Use different wording and terminology
        2. Focus on different aspects of the topic
        3. Include broader and more specific variants
        
        Return only the questions, one per line, without numbering or additional text.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates related search queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            related_queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
            
            # Ensure we don't return duplicates or the original query
            unique_queries = []
            for query in related_queries:
                if query.lower() != original_query.lower() and query not in unique_queries:
                    unique_queries.append(query)
            
            logger.info(f"Generated {len(unique_queries)} unique related queries")
            return unique_queries[:num_queries]
            
        except Exception as e:
            logger.error(f"Error generating related queries: {e}")
            return []
    
    async def multi_query_search(self, original_query: str, 
                                match_count: int = 15,
                                match_threshold: float = 0.3,
                                num_expansion_queries: int = 2,
                                final_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform multi-query search by expanding the original query.
        
        Args:
            original_query: The original search query.
            match_count: Number of results per query.
            match_threshold: Similarity threshold.
            num_expansion_queries: Number of additional queries to generate.
            final_top_k: Final number of unique results to return.
            
        Returns:
            Combined and deduplicated search results.
        """
        logger.info(f"Performing multi-query search for: '{original_query}'")
        
        # Generate related queries
        related_queries = await self.generate_related_queries(original_query, num_expansion_queries)
        
        # Combine original query with related queries
        all_queries = [original_query] + related_queries
        
        # Search with each query
        all_results = []
        seen_chunks: Set[str] = set()  # To avoid duplicates
        
        for i, query in enumerate(all_queries):
            logger.info(f"Searching with query {i+1}/{len(all_queries)}: '{query}'")
            
            results = VectorSearch.search(
                query=query,
                match_count=match_count,
                match_threshold=match_threshold
            )
            
            # Add query context to results and deduplicate
            for result in results:
                chunk_id = result.get('id')
                if chunk_id not in seen_chunks:
                    result['source_query'] = query
                    result['query_rank'] = i  # Original query gets rank 0
                    all_results.append(result)
                    seen_chunks.add(chunk_id)
        
        # Sort by similarity score (descending) and take top_k
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        final_results = all_results[:final_top_k]
        
        logger.info(f"Multi-query search found {len(final_results)} unique chunks from {len(all_queries)} queries")
        
        return final_results
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: The complex query to decompose.
            
        Returns:
            List of simpler sub-queries.
        """
        # Simple heuristic decomposition based on conjunctions
        conjunctions = ['and', 'also', 'additionally', 'furthermore', 'moreover']
        
        sub_queries = [query]  # Start with original
        
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.lower().split(conj)
                if len(parts) > 1:
                    sub_queries.extend([part.strip() for part in parts if part.strip()])
                    break
        
        return list(set(sub_queries))  # Remove duplicates 