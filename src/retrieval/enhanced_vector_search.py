"""
Enhanced vector search module with re-ranking and multi-query expansion.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from src.retrieval.vector_search import VectorSearch
from src.retrieval.reranker import SemanticReranker
from src.retrieval.query_expansion import QueryExpansion
from src.utils.observability import SearchMetrics, PerformanceTimer

logger = logging.getLogger(__name__)

class EnhancedVectorSearch:
    """
    Enhanced vector search with semantic re-ranking and multi-query expansion.
    """
    
    def __init__(self):
        """Initialize the enhanced vector search."""
        self.query_expansion = QueryExpansion()
        self.reranker = SemanticReranker()
    
    async def search_with_reranking(self, query: str, 
                                   match_count: int = 15,
                                   match_threshold: float = 0.3,
                                   rerank_top_k: int = 10,
                                   use_cross_encoder: bool = False) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Search with semantic re-ranking.
        
        Args:
            query: The search query.
            match_count: Initial number of results to retrieve.
            match_threshold: Similarity threshold.
            rerank_top_k: Number of results after re-ranking.
            use_cross_encoder: Whether to use cross-encoder re-ranking.
            
        Returns:
            Tuple of (reranked results, search metrics).
        """
        with PerformanceTimer("search_with_reranking") as timer:
            # Initial search
            initial_results = VectorSearch.search(
                query=query,
                match_count=match_count,
                match_threshold=match_threshold
            )
            
            if not initial_results:
                return [], SearchMetrics(
                    query=query,
                    search_time_ms=timer.duration_ms,
                    num_results=0,
                    used_reranking=True,
                    used_cache=False
                )
            
            # Re-rank results
            if use_cross_encoder:
                reranked_results = self.reranker.cross_encoder_rerank(
                    query, initial_results, rerank_top_k
                )
            else:
                reranked_results = self.reranker.rerank_results(
                    query, initial_results, rerank_top_k
                )
            
            # Calculate metrics
            similarities = [r.get('rerank_score', 0) for r in reranked_results]
            metrics = SearchMetrics(
                query=query,
                search_time_ms=timer.duration_ms,
                num_results=len(reranked_results),
                avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                max_similarity=max(similarities) if similarities else 0,
                min_similarity=min(similarities) if similarities else 0,
                used_reranking=True,
                used_cache=False
            )
            
            return reranked_results, metrics
    
    async def multi_query_search_with_reranking(self, query: str,
                                               match_count: int = 15,
                                               match_threshold: float = 0.3,
                                               num_expansion_queries: int = 2,
                                               final_top_k: int = 10,
                                               use_reranking: bool = True) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Perform multi-query expansion with optional re-ranking.
        
        Args:
            query: The original search query.
            match_count: Number of results per query.
            match_threshold: Similarity threshold.
            num_expansion_queries: Number of additional queries to generate.
            final_top_k: Final number of results to return.
            use_reranking: Whether to apply re-ranking.
            
        Returns:
            Tuple of (search results, search metrics).
        """
        with PerformanceTimer("multi_query_search") as timer:
            # Perform multi-query search
            multi_query_results = await self.query_expansion.multi_query_search(
                original_query=query,
                match_count=match_count,
                match_threshold=match_threshold,
                num_expansion_queries=num_expansion_queries,
                final_top_k=final_top_k * 2 if use_reranking else final_top_k  # Get more for re-ranking
            )
            
            # Optional re-ranking
            if use_reranking and multi_query_results:
                final_results = self.reranker.rerank_results(
                    query, multi_query_results, final_top_k
                )
            else:
                final_results = multi_query_results[:final_top_k]
            
            # Get expansion queries used
            expansion_queries = getattr(self.query_expansion, '_last_expansion_queries', [])
            
            # Calculate metrics
            similarities = [r.get('rerank_score' if use_reranking else 'similarity', 0) 
                          for r in final_results]
            
            metrics = SearchMetrics(
                query=query,
                search_time_ms=timer.duration_ms,
                num_results=len(final_results),
                avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                max_similarity=max(similarities) if similarities else 0,
                min_similarity=min(similarities) if similarities else 0,
                used_reranking=use_reranking,
                used_multi_query=True,
                used_cache=False,
                expansion_queries=expansion_queries
            )
            
            return final_results, metrics
    
    async def adaptive_search(self, query: str,
                             match_count: int = 10,
                             match_threshold: float = 0.3,
                             min_results_threshold: int = 3) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Adaptive search that escalates techniques based on initial results quality.
        
        Args:
            query: The search query.
            match_count: Number of results to retrieve.
            match_threshold: Similarity threshold.
            min_results_threshold: Minimum number of good results before escalating.
            
        Returns:
            Tuple of (search results, search metrics).
        """
        logger.info(f"Starting adaptive search for: '{query}'")
        
        # Step 1: Basic search
        with PerformanceTimer("adaptive_search") as timer:
            basic_results = VectorSearch.search(query, match_count, match_threshold)
            
            # Check if we have enough high-quality results
            high_quality_results = [r for r in basic_results if r.get('similarity', 0) > 0.7]
            
            if len(high_quality_results) >= min_results_threshold:
                logger.info("Basic search yielded sufficient high-quality results")
                similarities = [r.get('similarity', 0) for r in basic_results]
                metrics = SearchMetrics(
                    query=query,
                    search_time_ms=timer.duration_ms,
                    num_results=len(basic_results),
                    avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                    max_similarity=max(similarities) if similarities else 0,
                    min_similarity=min(similarities) if similarities else 0,
                    used_cache=False
                )
                return basic_results, metrics
            
            # Step 2: Try re-ranking if basic search was insufficient
            logger.info("Basic search insufficient, trying re-ranking")
            reranked_results = self.reranker.rerank_results(query, basic_results)
            
            high_quality_reranked = [r for r in reranked_results if r.get('rerank_score', 0) > 0.7]
            
            if len(high_quality_reranked) >= min_results_threshold:
                logger.info("Re-ranking improved results sufficiently")
                similarities = [r.get('rerank_score', 0) for r in reranked_results]
                metrics = SearchMetrics(
                    query=query,
                    search_time_ms=timer.duration_ms,
                    num_results=len(reranked_results),
                    avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                    max_similarity=max(similarities) if similarities else 0,
                    min_similarity=min(similarities) if similarities else 0,
                    used_reranking=True,
                    used_cache=False
                )
                return reranked_results, metrics
            
            # Step 3: Use multi-query expansion as final escalation
            logger.info("Re-ranking insufficient, using multi-query expansion")
            return await self.multi_query_search_with_reranking(
                query=query,
                match_count=match_count,
                match_threshold=match_threshold * 0.8,  # Lower threshold for expansion
                num_expansion_queries=3,
                final_top_k=match_count,
                use_reranking=True
            )
    
    def format_enhanced_results_for_context(self, results: List[Dict[str, Any]], 
                                           include_scores: bool = True,
                                           include_query_source: bool = True) -> str:
        """
        Format enhanced search results with additional metadata.
        
        Args:
            results: The search results.
            include_scores: Whether to include similarity scores.
            include_query_source: Whether to include source query info.
            
        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            document_id = result.get("document_id")
            similarity = result.get("rerank_score", result.get("similarity", 0))
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            source_query = result.get("source_query", "")
            
            title = metadata.get("title", "Unknown")
            
            # Build context part
            context_part = f"[{i+1}] From '{title}'"
            
            if include_scores:
                context_part += f" (relevance: {similarity:.2f})"
            
            if include_query_source and source_query and source_query != results[0].get("source_query", ""):
                context_part += f" [found via: '{source_query}']"
            
            context_part += f":\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts) 