"""
Unit tests for the embeddings module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pytest

from src.utils.embeddings import (
    generate_embedding, 
    generate_batch_embeddings, 
    EmbeddingGenerator, 
    _get_embedding_generator
)
from src.retrieval.vector_search import VectorSearch

class TestEmbeddings(unittest.TestCase):
    """
    Test case for the embeddings module.
    """
    
    @patch('src.utils.embeddings._get_embedding_generator')
    def test_generate_embedding(self, mock_get_generator):
        """
        Test generating a single embedding.
        """
        # Mock the generator and its response
        mock_generator = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_generator.generate_embedding.return_value = mock_embedding
        mock_get_generator.return_value = mock_generator
        
        # Call the function
        embedding = generate_embedding("Test text")
        
        # Assertions
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_generator.generate_embedding.assert_called_once_with("Test text")
    
    @patch('src.utils.embeddings._get_embedding_generator')
    def test_generate_batch_embeddings(self, mock_get_generator):
        """
        Test generating batch embeddings.
        """
        # Mock the generator and its response
        mock_generator = MagicMock()
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_generator.generate_batch_embeddings.return_value = mock_embeddings
        mock_get_generator.return_value = mock_generator
        
        # Call the function
        embeddings = generate_batch_embeddings(["Text 1", "Text 2"])
        
        # Assertions
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        mock_generator.generate_batch_embeddings.assert_called_once_with(["Text 1", "Text 2"])
    
    @patch('src.utils.embeddings.os')
    @patch('src.utils.embeddings.OpenAI')
    def test_embedding_generator_init(self, mock_openai, mock_os):
        """
        Test initialization of EmbeddingGenerator.
        """
        # Mock environment variables
        mock_os.getenv.side_effect = lambda key, default=None: {
            "OPENAI_API_KEY": "test_api_key",
            "EMBEDDING_MODEL": "test-embedding-model"
        }.get(key, default)
        
        # Create instance
        generator = EmbeddingGenerator()
        
        # Assertions
        self.assertEqual(generator.api_key, "test_api_key")
        self.assertEqual(generator.model, "test-embedding-model")
        mock_openai.assert_called_once_with(api_key="test_api_key")
    
    @patch('src.utils.embeddings.os')
    @patch('src.utils.embeddings.OpenAI')
    def test_embedding_generator_default_model(self, mock_openai, mock_os):
        """
        Test EmbeddingGenerator uses default model when env var not set.
        """
        # Mock environment variables with no EMBEDDING_MODEL
        mock_os.getenv.side_effect = lambda key, default=None: {
            "OPENAI_API_KEY": "test_api_key",
        }.get(key, default)
        
        # Create instance
        generator = EmbeddingGenerator()
        
        # Assertions
        self.assertEqual(generator.model, "text-embedding-3-small")
    
    @patch('openai.OpenAI')
    def test_embedding_generator_api_key_required(self, mock_openai):
        """
        Test EmbeddingGenerator raises error when API key not provided.
        """
        # Patch os.getenv to return None for OPENAI_API_KEY
        with patch('os.getenv', return_value=None):
            with self.assertRaises(ValueError):
                EmbeddingGenerator()
    
    @patch('src.utils.embeddings.EmbeddingGenerator')
    @patch('src.utils.embeddings.OPENAI_API_KEY', 'test_key')
    def test_get_embedding_generator(self, mock_generator_class):
        """
        Test _get_embedding_generator creates instance once.
        """
        # Reset the global instance
        from src.utils.embeddings import _embedding_generator
        import src.utils.embeddings
        src.utils.embeddings._embedding_generator = None
        
        # First call should create a new instance
        _get_embedding_generator()
        mock_generator_class.assert_called_once()
        
        # Reset the mock to check if it's called again
        mock_generator_class.reset_mock()
        
        # Second call should reuse existing instance
        _get_embedding_generator()
        mock_generator_class.assert_not_called()

def test_vector_search_retrieves_relevant_chunk():
    """
    Test that a relevant query retrieves at least one chunk from the sample document.
    """
    query = "What is document chunking?"
    results = VectorSearch.search(query, match_count=3, match_threshold=0.3)
    assert len(results) > 0, "No chunks were retrieved for a relevant query."
    # Check that at least one result is about chunking
    assert any("chunk" in r["content"].lower() for r in results), "No relevant chunk found in results."


def test_vector_search_irrelevant_query_returns_nothing():
    """
    Test that an irrelevant query returns no results.
    """
    query = "Quantum entanglement in black holes"
    results = VectorSearch.search(query, match_count=3, match_threshold=0.3)
    assert len(results) == 0 or all(r["similarity"] < 0.3 for r in results), "Irrelevant query should not return results."


def test_vector_search_empty_query_raises():
    """
    Test that an empty query raises a ValueError.
    """
    with pytest.raises(ValueError):
        VectorSearch.search("")

if __name__ == "__main__":
    unittest.main() 