import pytest
from src.retrieval.vector_search import VectorSearch

def test_vector_search_retrieves_relevant_chunk():
    """
    Test that a relevant query retrieves at least one chunk from the sample document.
    """
    query = "What is document chunking?"
    results = VectorSearch.search(query, match_count=3, match_threshold=0.3)
    assert len(results) > 0, "No chunks were retrieved for a relevant query."
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