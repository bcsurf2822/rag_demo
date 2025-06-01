"""
Fast RAG agent with optimized performance and response times.
"""
import logging
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.utils.config import OPENAI_MODEL
from src.retrieval.optimized_vector_search import OptimizedVectorSearch, SearchConfig
from src.utils.observability import RAGMetrics, TokenUsage, metrics_collector
from src.utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class FastAgentContext(BaseModel):
    """
    Fast context for optimized RAG agent.
    """
    # Optimized search parameters
    match_count: int = 5  # Reduced default
    match_threshold: float = 0.3
    rerank_top_k: int = 3  # Minimal for speed
    
    # Performance settings
    use_caching: bool = True
    parallel_timeout: float = 8.0  # Shorter timeout
    
    # Session tracking
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class FastAgentResponse(BaseModel):
    """
    Fast response model optimized for performance.
    """
    answer: str = Field(description="The answer with citations")
    sources: List[str] = Field(description="Source document titles")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    
    # Performance metadata
    num_sources_used: int = Field(description="Number of sources used")
    search_method_used: str = Field(description="Search method applied")
    processing_time_ms: float = Field(description="Total processing time")
    token_usage: Dict[str, int] = Field(description="Token usage")
    
    # Optional debug info
    debug_info: Optional[str] = Field(default=None)

class FastRAGAgent:
    """
    Fast RAG agent optimized for response time while maintaining accuracy.
    """
    
    def __init__(self, model_name: str = OPENAI_MODEL):
        """Initialize the fast RAG agent."""
        self.optimized_search = OptimizedVectorSearch()
        
        # FIX: Use simple string output instead of complex structured output
        # This prevents the "Exceeded maximum retries for result validation" error
        self.agent = Agent(
            model_name,
            deps_type=FastAgentContext,
            output_type=str,  # Changed from FastAgentResponse to str
            system_prompt=self._get_optimized_system_prompt(),
            instrument=True,
            # Add retry settings for better error handling
            max_retries=2  # Allow more retries for robustness
        )
        
        # Register fast search tool
        self.agent.tool(self._fast_knowledge_search)
    
    def _get_optimized_system_prompt(self) -> str:
        """Get optimized system prompt for fast responses."""
        return """
        You are a fast, efficient AI assistant with access to a knowledge base.
        
        CRITICAL: Return ONLY the answer text. Do not attempt to format as JSON or structured data.
        
        Instructions for SPEED and ACCURACY:
        1. Provide direct, concise answers based on retrieved information
        2. Focus on the most relevant information first
        3. Use bullet points and clear structure for readability
        4. Include citations in this format: [Source Title]
        5. If information is insufficient, say so quickly rather than elaborating
        
        Response Guidelines:
        - Start with the main answer
        - Support with key evidence from sources
        - Keep citations simple: [Source Title]
        - Be concise but comprehensive
        - Use markdown formatting for clarity
        
        Example good response:
        "Based on the retrieved information, [main answer here].
        
        Key points:
        • [Point 1 from Source A]
        • [Point 2 from Source B]
        
        [Source A], [Source B]"
        """
    
    async def _fast_knowledge_search(self, ctx: RunContext[FastAgentContext], 
                                   query: str) -> str:
        """
        Fast knowledge search optimized for performance.
        
        Args:
            ctx: The run context with configuration.
            query: The search query.
            
        Returns:
            Formatted search results.
        """
        logger.info(f"Fast search for: '{query}'")
        
        config_data = ctx.deps
        search_config = SearchConfig(
            match_count=config_data.match_count,
            match_threshold=config_data.match_threshold,
            rerank_top_k=config_data.rerank_top_k,
            enable_caching=config_data.use_caching,
            parallel_timeout=config_data.parallel_timeout,
            expansion_queries=1 if len(query.split()) > 8 else 0  # Adaptive expansion
        )
        
        try:
            # Use smart adaptive search for optimal performance
            results, search_metrics = await self.optimized_search.smart_adaptive_search(
                query=query,
                config=search_config
            )
            
            # Store search metadata in context for later use
            ctx.deps._search_results = results
            ctx.deps._search_metrics = search_metrics
            ctx.deps._search_method = "fast_adaptive"
            
            if results:
                # Optimized context formatting for better LLM processing
                context_parts = []
                for i, result in enumerate(results):
                    content = result.get('content', '')
                    title = result.get('title', f'Source {i+1}')
                    score = result.get('rerank_score', result.get('similarity', 0))
                    
                    # Improved chunking: limit to key sentences for better focus
                    sentences = content.split('. ')
                    key_content = '. '.join(sentences[:3])  # First 3 sentences
                    if len(sentences) > 3:
                        key_content += '...'
                    
                    context_parts.append(
                        f"**{title}** (Relevance: {score:.2f})\n{key_content}"
                    )
                
                context = "\n\n".join(context_parts)
                logger.info(f"Fast search found {len(results)} relevant chunks")
                return context
            else:
                logger.warning(f"No results found for: '{query}'")
                return "No relevant information found in the knowledge base."
                
        except Exception as e:
            logger.error(f"Fast search failed: {e}")
            return f"Search error occurred: {str(e)}"
    
    async def answer_question(self, question: str,
                             match_count: int = 5,
                             match_threshold: float = 0.3,
                             rerank_top_k: int = 3,
                             use_caching: bool = True,
                             include_debug_info: bool = False,
                             session_id: Optional[str] = None) -> FastAgentResponse:
        """
        Answer a question quickly with optimized retrieval.
        
        Args:
            question: The user's question.
            match_count: Number of chunks to retrieve initially.
            match_threshold: Similarity threshold for retrieval.
            rerank_top_k: Number of chunks after reranking.
            use_caching: Whether to use result caching.
            include_debug_info: Whether to include debug information.
            session_id: Session identifier.
            
        Returns:
            FastAgentResponse with answer and metadata.
        """
        from src.utils.observability import PerformanceTimer
        
        with PerformanceTimer("fast_answer_question") as timer:
            # Create optimized context
            context = FastAgentContext(
                match_count=match_count,
                match_threshold=match_threshold,
                rerank_top_k=rerank_top_k,
                use_caching=use_caching,
                session_id=session_id or str(uuid.uuid4())
            )
            
            try:
                # Run the agent with improved error handling
                result = await asyncio.wait_for(
                    self.agent.run(question, deps=context),
                    timeout=30.0  # 30-second hard timeout
                )
                
                # Extract the simple string answer
                answer_text = result.data  # Now this is just a string
                
                # Extract search metadata from context
                search_results = getattr(context, '_search_results', [])
                search_metrics = getattr(context, '_search_metrics', None)
                search_method = getattr(context, '_search_method', 'unknown')
                
                # Calculate confidence based on 2024 best practices
                confidence = self._calculate_enhanced_confidence(search_results, answer_text, search_metrics)
                
                # Build structured response from our metadata
                response = FastAgentResponse(
                    answer=answer_text,
                    sources=[r.get('title', 'Unknown') for r in search_results],
                    confidence=confidence,
                    num_sources_used=len(search_results),
                    search_method_used=search_method,
                    processing_time_ms=timer.duration_ms,
                    token_usage={
                        'total_tokens': getattr(result.usage(), 'total_tokens', 0),
                        'request_tokens': getattr(result.usage(), 'request_tokens', 0),
                        'response_tokens': getattr(result.usage(), 'response_tokens', 0)
                    },
                    debug_info=self._format_debug_info(search_metrics, search_results) if include_debug_info else None
                )
                
                # Record metrics with enhanced tracking
                rag_metrics = RAGMetrics(
                    session_id=context.session_id,
                    query=question,
                    response=response.answer,
                    num_sources=response.num_sources_used,
                    total_time_ms=timer.duration_ms,
                    confidence_score=confidence,
                    search_metrics=search_metrics,
                    token_usage=TokenUsage(
                        total_tokens=response.token_usage['total_tokens'],
                        request_tokens=response.token_usage['request_tokens'],
                        response_tokens=response.token_usage['response_tokens']
                    )
                )
                
                metrics_collector.record_rag_interaction(rag_metrics)
                
                logger.info(f"Fast question answered in {timer.duration_ms:.1f}ms with {len(search_results)} sources")
                return response
                
            except asyncio.TimeoutError:
                logger.error(f"Question answering timed out after 30 seconds: {question}")
                return self._create_error_response("timeout", timer.duration_ms, 
                    "I apologize, but the request timed out. Please try a simpler question or try again later.")
                
            except Exception as e:
                logger.error(f"Error in fast question answering: {e}")
                return self._create_error_response("error", timer.duration_ms,
                    f"I encountered an error while processing your question: {str(e)}")
    
    def _create_error_response(self, error_type: str, duration_ms: float, message: str) -> FastAgentResponse:
        """Create a standardized error response."""
        return FastAgentResponse(
            answer=message,
            sources=[],
            confidence=0.0,
            num_sources_used=0,
            search_method_used=error_type,
            processing_time_ms=duration_ms,
            token_usage={'total_tokens': 0, 'request_tokens': 0, 'response_tokens': 0}
        )
    
    def _calculate_enhanced_confidence(self, search_results: List[Dict[str, Any]], 
                                     answer: str, search_metrics: Any) -> float:
        """
        Enhanced confidence calculation based on 2024 RAG best practices.
        
        Args:
            search_results: Retrieved search results.
            answer: Generated answer.
            search_metrics: Search performance metrics.
            
        Returns:
            Confidence score between 0 and 1.
        """
        if not search_results:
            return 0.0
        
        # 1. Base confidence on semantic similarity scores
        scores = [r.get('rerank_score', r.get('similarity', 0)) for r in search_results]
        avg_score = sum(scores) / len(scores)
        
        # 2. Quality threshold bonus (2024 best practice: focus on high-quality chunks)
        high_quality_sources = sum(1 for score in scores if score > 0.7)
        quality_bonus = min(high_quality_sources * 0.15, 0.4)
        
        # 3. Source diversity bonus (multiple sources = higher confidence)
        unique_sources = len(set(r.get('title', '') for r in search_results))
        diversity_bonus = min(unique_sources * 0.1, 0.2)
        
        # 4. Answer completeness factor (optimal length indicates good coverage)
        answer_words = len(answer.split())
        if 20 <= answer_words <= 200:  # Sweet spot for RAG answers
            completeness_factor = 1.0
        elif answer_words < 20:
            completeness_factor = 0.7  # Too short
        else:
            completeness_factor = 0.9  # Too long
        
        # 5. Search performance factor (fast retrieval indicates good indexing)
        speed_factor = 1.0
        if search_metrics and hasattr(search_metrics, 'search_time_ms'):
            if search_metrics.search_time_ms < 1000:  # Under 1 second
                speed_factor = 1.1
            elif search_metrics.search_time_ms > 5000:  # Over 5 seconds
                speed_factor = 0.9
        
        # Combine all factors with proper weighting
        confidence = (avg_score * 0.4 +  # Primary factor
                     quality_bonus +      # High-quality sources
                     diversity_bonus) * completeness_factor * speed_factor
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _format_debug_info(self, search_metrics, search_results) -> str:
        """Format debug information with enhanced details."""
        if not search_metrics:
            return "No search metrics available"
        
        info_parts = [
            f"Search time: {search_metrics.search_time_ms:.1f}ms",
            f"Results: {search_metrics.num_results}",
            f"Avg similarity: {search_metrics.avg_similarity:.3f}"
        ]
        
        if hasattr(search_metrics, 'used_cache') and search_metrics.used_cache:
            info_parts.append("✅ Cache hit")
        
        if hasattr(search_metrics, 'used_reranking') and search_metrics.used_reranking:
            info_parts.append("🔄 Reranked")
        
        if hasattr(search_metrics, 'expansion_queries') and search_metrics.expansion_queries:
            info_parts.append(f"🔍 {len(search_metrics.expansion_queries)} expansions")
        
        # Add top result quality
        if search_results:
            top_score = max(r.get('rerank_score', r.get('similarity', 0)) for r in search_results)
            info_parts.append(f"Top score: {top_score:.3f}")
        
        return " | ".join(info_parts)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from metrics collector."""
        return metrics_collector.get_performance_summary()
    
    def clear_cache(self):
        """Clear search caches for fresh performance."""
        if hasattr(self.optimized_search, '_query_cache'):
            self.optimized_search._query_cache.clear()
            logger.info("Search cache cleared") 