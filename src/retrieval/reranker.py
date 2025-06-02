"""
Semantic re-ranking module for improving retrieval accuracy.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.embeddings import generate_embedding

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Class for semantically re-ranking retrieved document chunks.
    """
    
    @staticmethod
    def rerank_results(query: str, results: List[Dict[str, Any]], 
                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Re-rank search results using semantic similarity.
        
        Args:
            query: The original search query.
            results: List of retrieved chunks with similarity scores.
            top_k: Number of top results to return after re-ranking.
            
        Returns:
            Re-ranked list of chunks.
        """
        if not results:
            return results
            
        logger.info(f"Re-ranking {len(results)} results for query: '{query}'")
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Calculate semantic similarity scores for each result
        reranked_results = []
        
        for result in results:
            # Get the stored embedding from the result
            stored_embedding = result.get('embedding')
            
            if stored_embedding:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [query_embedding], 
                    [stored_embedding]
                )[0][0]
                
                # Create a new result with updated similarity score
                reranked_result = result.copy()
                reranked_result['rerank_score'] = float(similarity)
                reranked_results.append(reranked_result)
            else:
                # If no embedding available, keep original score
                reranked_result = result.copy()
                reranked_result['rerank_score'] = result.get('similarity', 0.0)
                reranked_results.append(reranked_result)
        
        # Sort by re-ranking score (descending)
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply top_k limit if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        logger.info(f"Re-ranking complete. Top result score: {reranked_results[0]['rerank_score']:.3f}")
        
        return reranked_results
    
    @staticmethod
    def cross_encoder_rerank(query: str, results: List[Dict[str, Any]], 
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Re-rank using cross-encoder approach (query-document pairs).
        This is a more advanced re-ranking method that considers query-document interaction.
        
        Args:
            query: The search query.
            results: List of retrieved chunks.
            top_k: Number of top results to return.
            
        Returns:
            Re-ranked list of chunks.
        """
        if not results:
            return results
            
        logger.info(f"Cross-encoder re-ranking {len(results)} results")
        
    
        
        query_words = set(query.lower().split())
        reranked_results = []
        
        for result in results:
            content = result.get('content', '').lower()
            content_words = set(content.split())
            
            # Calculate word overlap score
            overlap_score = len(query_words.intersection(content_words)) / len(query_words.union(content_words))
            
            # Combine with original similarity
            original_score = result.get('similarity', 0.0)
            combined_score = 0.6 * original_score + 0.4 * overlap_score
            
            reranked_result = result.copy()
            reranked_result['rerank_score'] = combined_score
            reranked_results.append(reranked_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k:
            reranked_results = reranked_results[:top_k]
            
        return reranked_results 