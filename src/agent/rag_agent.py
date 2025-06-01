"""
RAG agent module using Pydantic AI.
"""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.utils.config import OPENAI_MODEL
from src.retrieval.vector_search import VectorSearch

logger = logging.getLogger(__name__)

class AgentContext(BaseModel):
    """
    Context used by the Pydantic AI agent for RAG.
    """
    kb_match_count: int = 5
    kb_match_threshold: float = 0.3

class AgentResponse(BaseModel):
    """
    Response model for the RAG agent.
    """
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="Sources used to generate the answer")
    confidence: float = Field(description="Confidence score (0-1) in the answer", ge=0, le=1)

class RAGAgent:
    """
    Retrieval-Augmented Generation agent using Pydantic AI.
    """
    
    def __init__(self, model_name: str = OPENAI_MODEL):
        """
        Initialize the RAG agent.
        
        Args:
            model_name: The OpenAI model to use.
        """
        self.agent = Agent(
            model_name,
            deps_type=AgentContext,
            output_type=AgentResponse,
            system_prompt=self._get_system_prompt()
        )
        
        # Register the knowledge base search tool
        self.agent.tool(self._search_knowledge_base)
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            The system prompt text.
        """
        return """
        You are a helpful assistant with access to a knowledge base of documents.
        
        When answering questions:
        1. Use the knowledge base search tool to find relevant information
        2. Base your answers primarily on the retrieved content, not on your general knowledge
        3. If the knowledge base doesn't contain relevant information, acknowledge this
        4. Always cite your sources by referencing the document title
        5. Be concise but complete in your answers
        
        Format your responses carefully and ensure they are well-structured.
        """
    
    async def _search_knowledge_base(self, ctx: RunContext[AgentContext], query: str) -> str:
        """
        Search the knowledge base for information relevant to the query.
        
        Args:
            ctx: The run context with agent dependencies.
            query: The search query.
            
        Returns:
            The retrieved context as a string.
        """
        logger.info(f"Searching knowledge base for: {query}")
        
        # Get search parameters from context
        match_count = ctx.deps.kb_match_count
        match_threshold = ctx.deps.kb_match_threshold
        
        # Perform the search
        results = VectorSearch.search(
            query=query,
            match_count=match_count,
            match_threshold=match_threshold
        )
        
        # Format the results for the agent
        context = VectorSearch.format_results_for_context(results)
        
        logger.info(f"Retrieved {len(results)} relevant chunks from knowledge base")
        
        return context
    
    async def answer_question(self, question: str, match_count: int = 5, 
                             match_threshold: float = 0.7) -> AgentResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question.
            match_count: The maximum number of knowledge base matches to consider.
            match_threshold: The similarity threshold for knowledge base matches.
            
        Returns:
            An AgentResponse object with the answer and metadata.
        """
        logger.info(f"Answering question: {question}")
        
        # Set up the agent context
        context = AgentContext(
            kb_match_count=match_count,
            kb_match_threshold=match_threshold
        )
        
        # Run the agent
        result = await self.agent.run(question, deps=context)
        
        return result.output 