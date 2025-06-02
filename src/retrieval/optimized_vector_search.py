import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from functools import lru_cache
import hashlib

from src.retrieval.vector_search import VectorSearch
from src.retrieval.reranker import SemanticReranker
from src.utils.embeddings import generate_embedding
from src.utils.observability import SearchMetrics, PerformanceTimer
from src.utils.config import OPENAI_API_KEY, OPENAI_MODEL
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for optimized search."""
    match_count: int = 8  
    match_threshold: float = 0.3
    rerank_top_k: int = 5  
    expansion_queries: int = 1  
    enable_caching: bool = True
    parallel_timeout: float = 10.0

class OptimizedVectorSearch:
    """
    Optimized vector search with parallel processing and intelligent caching.
    """
    
    def __init__(self):
        """Initialize the optimized search."""
        self.reranker = SemanticReranker()
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._query_cache = {}
        self._embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @lru_cache(maxsize=100)
    def _cache_embedding(self, text: str) -> List[float]:
        """Cache embeddings to avoid regeneration."""
        return generate_embedding(text)
    
    def _get_query_hash(self, query: str, config: SearchConfig) -> str:
        """Generate a hash key for query caching."""
        key_data = f"{query}_{config.match_count}_{config.match_threshold}_{config.rerank_top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def fast_search_with_reranking(self, query: str, 
                                        config: SearchConfig = None) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Fast search with optimized reranking and caching.
        
        Args:
            query: The search query.
            config: Search configuration.
            
        Returns:
            Tuple of (reranked results, search metrics).
        """
        config = config or SearchConfig()
        
        with PerformanceTimer("fast_search_with_reranking") as timer:
            # Check cache 
            cache_key = self._get_query_hash(query, config)
            if config.enable_caching and cache_key in self._query_cache:
                cached_results, cached_time = self._query_cache[cache_key]
                if time.time() - cached_time < 300:  
                    logger.info(f"Using cached results for query: '{query}'")
                    metrics = SearchMetrics(
                        query=query,
                        search_time_ms=timer.duration_ms,
                        num_results=len(cached_results),
                        used_cache=True
                    )
                    return cached_results, metrics
            
            # Define async wrapper for synchronous search
            async def async_vector_search():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    lambda: VectorSearch.search(
                        query,
                        config.match_count * 2,  
                        config.match_threshold
                    )
                )
            
            # Parallel initial search and query embedding generation
            search_task = asyncio.create_task(async_vector_search())
            
            # Pre-generate query embedding for reranking
            query_embedding = self._cache_embedding(query)
            
            # Wait for search results
            initial_results = await search_task
            
            if not initial_results:
                return [], SearchMetrics(
                    query=query,
                    search_time_ms=timer.duration_ms,
                    num_results=0,
                    used_cache=False
                )
            
            # Fast reranking with pre-computed embedding
            reranked_results = self._fast_rerank(query_embedding, initial_results, config.rerank_top_k)
            
            # Cache results
            if config.enable_caching:
                self._query_cache[cache_key] = (reranked_results, time.time())
               
                if len(self._query_cache) > 50:
                    oldest_key = min(self._query_cache.keys(), 
                                   key=lambda k: self._query_cache[k][1])
                    del self._query_cache[oldest_key]
            
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
    
    def _fast_rerank(self, query_embedding: List[float], 
                    results: List[Dict[str, Any]], 
                    top_k: int) -> List[Dict[str, Any]]:
        """
        Fast reranking using pre-computed query embedding.
        
        Args:
            query_embedding: Pre-computed query embedding.
            results: Search results to rerank.
            top_k: Number of top results to return.
            
        Returns:
            Reranked results.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if not results:
            return results
        
        # Batch compute similarities
        embeddings = []
        valid_results = []
        
        for result in results:
            embedding = result.get('embedding')
            if embedding:
                embeddings.append(embedding)
                valid_results.append(result)
        
        if not embeddings:
            return results[:top_k]
        
        # Vectorized similarity computation
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Update results with rerank scores
        for i, result in enumerate(valid_results):
            result['rerank_score'] = float(similarities[i])
        
        # Sort and return top_k
        valid_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return valid_results[:top_k]
    
    async def parallel_multi_query_search(self, query: str,
                                         config: SearchConfig = None) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Parallel multi-query search with minimal expansion.
        
        Args:
            query: Original search query.
            config: Search configuration.
            
        Returns:
            Tuple of (search results, search metrics).
        """
        config = config or SearchConfig()
        
        with PerformanceTimer("parallel_multi_query") as timer:
            # Generate minimal expansion queries quickly
            expansion_task = asyncio.create_task(
                self._fast_query_expansion(query, config.expansion_queries)
            )
            
            # Define async wrapper for synchronous search
            async def async_vector_search(search_query, match_count, threshold):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    lambda: VectorSearch.search(search_query, match_count, threshold)
                )
            
            # Start original query search immediately
            original_search_task = asyncio.create_task(
                async_vector_search(query, config.match_count, config.match_threshold)
            )
            
         
            expansion_queries = await expansion_task
            
            # Launch parallel searches for expansion queries
            expansion_tasks = [
                asyncio.create_task(
                    async_vector_search(exp_query, config.match_count // 2, config.match_threshold)
                )
                for exp_query in expansion_queries
            ]
            
     
            try:
                all_tasks = [original_search_task] + expansion_tasks
                results_list = await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True),
                    timeout=config.parallel_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Multi-query search timed out, using available results")
                results_list = [original_search_task.result() if original_search_task.done() else []]
            
            # Combine results efficiently
            all_results = []
            seen_ids = set()
            
            for i, results in enumerate(results_list):
                if isinstance(results, Exception):
                    logger.warning(f"Search {i} failed: {results}")
                    continue
                
                for result in results:
                    chunk_id = result.get('id')
                    if chunk_id not in seen_ids:
                        result['query_rank'] = i
                        all_results.append(result)
                        seen_ids.add(chunk_id)
            
            # Fast reranking
            query_embedding = self._cache_embedding(query)
            final_results = self._fast_rerank(query_embedding, all_results, config.rerank_top_k)
            
            # Calculate metrics
            similarities = [r.get('rerank_score', r.get('similarity', 0)) for r in final_results]
            metrics = SearchMetrics(
                query=query,
                search_time_ms=timer.duration_ms,
                num_results=len(final_results),
                avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                max_similarity=max(similarities) if similarities else 0,
                min_similarity=min(similarities) if similarities else 0,
                used_reranking=True,
                used_multi_query=True,
                used_cache=False,  
                expansion_queries=expansion_queries
            )
            
            return final_results, metrics
    
    async def _fast_query_expansion(self, query: str, num_queries: int = 1) -> List[str]:
        """
        Fast query expansion with optimized prompting.
        
        Args:
            query: Original query.
            num_queries: Number of expansion queries.
            
        Returns:
            List of expansion queries.
        """
        if num_queries == 0:
            return []
        
        # Simplified prompt for faster generation
        prompt = f"Rewrite this question using different words: {query}"
        
        try:
            response = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,  
                temperature=0.3  
            )
            
            expansion = response.choices[0].message.content.strip()
            return [expansion] if expansion and expansion.lower() != query.lower() else []
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []
    
    async def smart_adaptive_search(self, query: str,
                                   config: SearchConfig = None) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Smart adaptive search that chooses the best strategy based on query characteristics.
        Implements RAG best practices for optimal retrieval.
        
        Args:
            query: Search query.
            config: Search configuration.
            
        Returns:
            Tuple of (search results, search metrics).
        """
        config = config or SearchConfig()
        
        # Enhanced query analysis 
        query_words = query.split()
        query_length = len(query_words)
        
        # Detect complex queries that benefit from multi-query expansion
        complex_indicators = [
            'and', 'or', 'also', 'both', 'either', 'neither', 'compare', 'contrast',
            'difference', 'similar', 'versus', 'vs', 'between', 'what are', 'how do'
        ]
        has_multiple_concepts = any(indicator in query.lower() for indicator in complex_indicators)
        
        # Detect technical/specific queries that benefit from precise matching
        technical_indicators = ['error', 'code', 'function', 'method', 'api', 'configure', 'install']
        is_technical = any(indicator in query.lower() for indicator in technical_indicators)
        
        # Question type analysis
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        is_question = any(query.lower().startswith(qw) for qw in question_words)
        
        logger.info(f"Query analysis: length={query_length}, complex={has_multiple_concepts}, "
                   f"technical={is_technical}, question={is_question}")
        
        # Smart strategy selection
        if query_length > 12 or (has_multiple_concepts and query_length > 6):
   
            logger.info("Using multi-query strategy for complex query")
            return await self.parallel_multi_query_search(query, config)
        
        elif is_technical and query_length > 4:

            logger.info("Using enhanced reranking for technical query")
            # Increase match count for better recall, then rerank for precision
            enhanced_config = SearchConfig(
                match_count=config.match_count * 2,
                match_threshold=config.match_threshold * 0.8,  
                rerank_top_k=config.rerank_top_k,
                enable_caching=config.enable_caching,
                parallel_timeout=config.parallel_timeout
            )
            return await self.fast_search_with_reranking(query, enhanced_config)
        
        else:
            # Simple/direct query - use fast reranking
            logger.info("Using fast reranking for simple query")
            return await self.fast_search_with_reranking(query, config)
    
    def clear_cache(self):
        """Clear all caches."""
        self._query_cache.clear()
        self._embedding_cache.clear()
        self._cache_embedding.cache_clear()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 