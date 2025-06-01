"""
Unit tests for the RAG agent.
"""
import unittest
import pytest
from unittest.mock import patch, MagicMock

from src.agent.rag_agent import RAGAgent, AgentResponse

@pytest.mark.asyncio
class TestRAGAgent(unittest.TestCase):
    """
    Test case for the RAG agent.
    """
    
    @patch('src.agent.rag_agent.Agent')
    @patch('src.retrieval.vector_search.VectorSearch.search')
    async def test_answer_question(self, mock_search, mock_agent):
        """
        Test answering a question with the RAG agent.
        """
        # Mock the search results
        mock_search.return_value = [
            {
                'content': 'This is a test document about RAG.',
                'metadata': {
                    'title': 'Test Document',
                    'document_id': 1
                }
            }
        ]
        
        # Mock the agent response
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        mock_response = MagicMock()
        mock_response.answer = "This is a test answer."
        mock_response.sources = ["Test Document"]
        mock_response.confidence = 0.95
        
        mock_agent_instance.run.return_value = mock_response
        
        # Create the agent and call the method
        agent = RAGAgent()
        result = await agent.answer_question("What is RAG?")
        
        # Verify the search was called
        mock_search.assert_called_once()
        
        # Verify the agent was initialized
        mock_agent.assert_called_once()
        
        # Verify the result
        assert isinstance(result, AgentResponse)
        assert result.answer == "This is a test answer."
        assert result.sources == ["Test Document"]
        assert result.confidence == 0.95
    
    @patch('src.retrieval.vector_search.VectorSearch.search')
    async def test_answer_question_no_results(self, mock_search):
        """
        Test answering a question with no results.
        """
        # Mock an empty search result
        mock_search.return_value = []
        
        # Create the agent and call the method
        agent = RAGAgent()
        result = await agent.answer_question("What is something not in the database?")
        
        # Verify the search was called
        mock_search.assert_called_once()
        
        # Verify the result
        assert isinstance(result, AgentResponse)
        assert "I don't have enough information" in result.answer.lower()
        assert result.sources == []
        assert result.confidence < 0.5
    
    @patch('src.retrieval.vector_search.VectorSearch.search')
    async def test_answer_question_error(self, mock_search):
        """
        Test handling an error during question answering.
        """
        # Mock a search error
        mock_search.side_effect = Exception("Test error")
        
        # Create the agent and call the method
        agent = RAGAgent()
        
        # The method should raise an exception
        with pytest.raises(Exception):
            await agent.answer_question("What causes an error?") 