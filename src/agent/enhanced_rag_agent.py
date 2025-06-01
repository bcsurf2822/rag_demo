"""
Enhanced RAG agent module with improved retrieval and observability.
"""
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.utils.config import OPENAI_MODEL
from src.retrieval.enhanced_vector_search import EnhancedVectorSearch
from src.utils.observability import (
    RAGMetrics, TokenUsage, SearchMetrics, 
    PerformanceTimer, metrics_collector
)
from src.utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class EnhancedAgentContext(BaseModel):
    """
    Enhanced context for the RAG agent with improved search options.
    """
    # Search parameters
    match_count: int = 10
    match_threshold: float = 0.7
    use_reranking: bool = True
    use_multi_query: bool = False
    adaptive_search: bool = True
    
    # Response formatting
    include_debug_info: bool = False
    format_with_citations: bool = True
    
    # Session tracking
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class EnhancedAgentResponse(BaseModel):
    """
    Enhanced response model with additional metadata.
    """
    answer: str = Field(description="The formatted answer with citations")
    raw_answer: str = Field(description="The raw answer from the LLM")
    sources: List[str] = Field(description="Source document titles used")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    
    # Additional metadata
    num_sources_used: int = Field(description="Number of source chunks used")
    search_method_used: str = Field(description="Search method applied")
    processing_time_ms: float = Field(description="Total processing time")
    token_usage: Dict[str, int] = Field(description="Token usage statistics")
    
    # Debug information (optional)
    debug_info: Optional[str] = Field(default=None, description="Debug information about search")

class EnhancedRAGAgent:
    """
    Enhanced Retrieval-Augmented Generation agent with improved capabilities.
    """
    
    def __init__(self, model_name: str = OPENAI_MODEL):
        """
        Initialize the enhanced RAG agent.
        
        Args:
            model_name: The OpenAI model to use.
        """
        self.enhanced_search = EnhancedVectorSearch()
        
        # Configure Pydantic AI agent with enhanced capabilities
        self.agent = Agent(
            model_name,
            deps_type=EnhancedAgentContext,
            output_type=EnhancedAgentResponse,
            system_prompt=self._get_enhanced_system_prompt(),
            instrument=True  # Enable observability
        )
        
        # Register enhanced search tool
        self.agent.tool(self._enhanced_knowledge_search)
    
    def _get_enhanced_system_prompt(self) -> str:
        """
        Get the enhanced system prompt for the agent.
        
        Returns:
            The enhanced system prompt text.
        """
        return """
        You are an advanced AI assistant with access to a sophisticated knowledge base.
        
        When answering questions:
        1. Use the enhanced knowledge search tool to find the most relevant information
        2. Base your answers primarily on retrieved content, supplemented carefully with your knowledge
        3. If the knowledge base lacks sufficient information, acknowledge this clearly
        4. Provide well-structured, comprehensive answers with proper citations
        5. Use markdown formatting for clarity (headers, lists, emphasis)
        6. Calculate a confidence score based on:
           - Quality and relevance of retrieved sources
           - Coverage of the query topic
           - Consistency of information across sources
        
        Response Guidelines:
        - Use clear, professional language
        - Structure complex answers with headers and bullet points
        - Cite sources when referencing specific information
        - Be honest about limitations and uncertainties
        - Provide actionable information when possible
        
        Always return:
        - A well-formatted answer with citations
        - The raw answer text (for processing)
        - List of source document titles
        - Confidence score (0.0-1.0)
        - Metadata about the search process
        """
    
    async def _enhanced_knowledge_search(self, ctx: RunContext[EnhancedAgentContext], 
                                       query: str) -> str:
        """
        Enhanced knowledge base search with multiple strategies.
        
        Args:
            ctx: The run context with agent configuration.
            query: The search query.
            
        Returns:
            Formatted search results for the agent.
        """
        logger.info(f"Enhanced search for: '{query}'")
        
        # Get search configuration from context
        config = ctx.deps
        
        try:
            # Choose search strategy based on configuration
            if config.adaptive_search:
                results, search_metrics = await self.enhanced_search.adaptive_search(
                    query=query,
                    match_count=config.match_count,
                    match_threshold=config.match_threshold
                )
                search_method = "adaptive"
                
            elif config.use_multi_query:
                results, search_metrics = await self.enhanced_search.multi_query_search_with_reranking(
                    query=query,
                    match_count=config.match_count * 2,
                    match_threshold=config.match_threshold,
                    use_reranking=config.use_reranking,
                    final_top_k=config.match_count
                )
                search_method = "multi-query"
                
            elif config.use_reranking:
                results, search_metrics = await self.enhanced_search.search_with_reranking(
                    query=query,
                    match_count=config.match_count * 2,
                    match_threshold=config.match_threshold,
                    rerank_top_k=config.match_count
                )
                search_method = "reranked"
                
            else:
                # Fallback to basic search with metrics tracking
                with PerformanceTimer("basic_search") as timer:
                    from src.retrieval.vector_search import VectorSearch
                    results = VectorSearch.search(
                        query=query,
                        match_count=config.match_count,
                        match_threshold=config.match_threshold
                    )
                    
                    similarities = [r.get('similarity', 0) for r in results]
                    search_metrics = SearchMetrics(
                        query=query,
                        search_time_ms=timer.duration_ms,
                        num_results=len(results),
                        avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                        max_similarity=max(similarities) if similarities else 0,
                        min_similarity=min(similarities) if similarities else 0,
                        used_cache=False
                    )
                    search_method = "basic"
            
            # Store search metadata for later use
            ctx.deps._search_results = results
            ctx.deps._search_metrics = search_metrics
            ctx.deps._search_method = search_method
            
            # Format results for the agent
            if results:
                context = self.enhanced_search.format_enhanced_results_for_context(
                    results, 
                    include_scores=True,
                    include_query_source=config.use_multi_query
                )
                logger.info(f"Found {len(results)} relevant chunks using {search_method} search")
                return context
            else:
                logger.warning(f"No relevant information found for query: '{query}'")
                return "No relevant information found in the knowledge base."
                
        except Exception as e:
            logger.error(f"Error in enhanced knowledge search: {e}")
            return f"Error occurred during search: {str(e)}"
    
    async def answer_question(self, question: str, 
                             match_count: int = 10,
                             match_threshold: float = 0.7,
                             use_reranking: bool = True,
                             use_multi_query: bool = False,
                             adaptive_search: bool = True,
                             include_debug_info: bool = False,
                             session_id: Optional[str] = None) -> EnhancedAgentResponse:
        """
        Answer a question using the enhanced RAG pipeline.
        
        Args:
            question: The user's question.
            match_count: Maximum number of chunks to retrieve.
            match_threshold: Similarity threshold for retrieval.
            use_reranking: Whether to apply semantic re-ranking.
            use_multi_query: Whether to use multi-query expansion.
            adaptive_search: Whether to use adaptive search strategy.
            include_debug_info: Whether to include debug information.
            session_id: Optional session ID for tracking.
            
        Returns:
            Enhanced agent response with metadata.
        """
        # Start timing
        with PerformanceTimer("total_rag_pipeline") as total_timer:
            
            # Create session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            metrics_collector.start_session(session_id)
            
            logger.info(f"Processing question: {question}")
            
            # Set up enhanced context
            context = EnhancedAgentContext(
                match_count=match_count,
                match_threshold=match_threshold,
                use_reranking=use_reranking,
                use_multi_query=use_multi_query,
                adaptive_search=adaptive_search,
                include_debug_info=include_debug_info,
                session_id=session_id
            )
            
            # Run the agent with timing
            with PerformanceTimer("agent_generation") as generation_timer:
                result = await self.agent.run(question, deps=context)
                agent_response = result.output
            
            # Extract search metadata
            search_results = getattr(context, '_search_results', [])
            search_metrics = getattr(context, '_search_metrics', None)
            search_method = getattr(context, '_search_method', 'unknown')
            
            # Calculate confidence if not provided
            if agent_response.confidence == 0:
                agent_response.confidence = self._calculate_confidence(search_results, agent_response.raw_answer)
            
            # Format the response with citations
            if context.format_with_citations:
                formatted_answer = ResponseFormatter.format_response_with_citations(
                    response=agent_response.raw_answer,
                    sources=search_results,
                    confidence_score=agent_response.confidence
                )
                agent_response.answer = formatted_answer
            
            # Add debug information if requested
            if include_debug_info and search_results:
                debug_info = ResponseFormatter.format_search_debug_info(
                    sources=search_results,
                    search_metrics=search_metrics.__dict__ if search_metrics else None
                )
                agent_response.debug_info = debug_info
            
            # Update response metadata
            agent_response.num_sources_used = len(search_results)
            agent_response.search_method_used = search_method
            agent_response.processing_time_ms = total_timer.duration_ms
            
            # Extract token usage from agent result
            usage = result.usage()
            agent_response.token_usage = {
                "request_tokens": usage.request_tokens,
                "response_tokens": usage.response_tokens,
                "total_tokens": usage.total_tokens
            }
            
            # Create and log comprehensive metrics
            rag_metrics = RAGMetrics(
                session_id=session_id,
                query=question,
                response=agent_response.answer,
                confidence_score=agent_response.confidence,
                total_time_ms=total_timer.duration_ms,
                retrieval_time_ms=search_metrics.search_time_ms if search_metrics else 0,
                generation_time_ms=generation_timer.duration_ms,
                token_usage=TokenUsage(
                    request_tokens=usage.request_tokens,
                    response_tokens=usage.response_tokens,
                    total_tokens=usage.total_tokens,
                    model_name=OPENAI_MODEL
                ),
                search_metrics=search_metrics,
                num_sources=len(search_results),
                context_chunks=search_results
            )
            
            metrics_collector.log_rag_metrics(rag_metrics)
            
            return agent_response
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]], 
                            answer: str) -> float:
        """
        Calculate confidence score based on search results and answer quality.
        
        Args:
            search_results: List of retrieved chunks.
            answer: The generated answer.
            
        Returns:
            Confidence score between 0 and 1.
        """
        if not search_results:
            return 0.1
        
        # Base confidence on average similarity scores
        similarities = [r.get('rerank_score', r.get('similarity', 0)) for r in search_results]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Adjust based on number of sources
        source_factor = min(len(search_results) / 5, 1.0)  # Optimal around 5 sources
        
        # Adjust based on answer length (too short or too long reduces confidence)
        answer_length = len(answer.split())
        if answer_length < 10:
            length_factor = 0.7
        elif answer_length > 500:
            length_factor = 0.9
        else:
            length_factor = 1.0
        
        # Combine factors
        confidence = avg_similarity * source_factor * length_factor
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for recent queries.
        
        Returns:
            Performance summary dictionary.
        """
        return metrics_collector.get_performance_summary(last_n=10) 