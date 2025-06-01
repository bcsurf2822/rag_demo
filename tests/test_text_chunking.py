"""
Unit tests for the text chunking module.
"""
import unittest
from typing import List

from src.utils.text_chunking import chunk_text, chunk_text_with_metadata

class TestTextChunking(unittest.TestCase):
    """
    Test case for the text chunking module.
    """
    
    def test_empty_text(self):
        """
        Test chunking with empty text should return an empty list.
        """
        chunks = chunk_text("")
        self.assertEqual(len(chunks), 0)
        
        chunks = chunk_text(None)
        self.assertEqual(len(chunks), 0)
    
    def test_short_text(self):
        """
        Test chunking with text shorter than chunk size.
        """
        text = "This is a short text."
        chunk_size = 100
        
        chunks = chunk_text(text, chunk_size)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_paragraph_breaks(self):
        """
        Test chunking breaks at paragraphs when possible.
        """
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
        chunk_size = 30
        
        chunks = chunk_text(text, chunk_size)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is paragraph one.")
        self.assertEqual(chunks[1], "This is paragraph two.")
        self.assertEqual(chunks[2], "This is paragraph three.")
    
    def test_newline_breaks(self):
        """
        Test chunking breaks at newlines when paragraphs aren't available.
        """
        text = "Line one.\nLine two.\nLine three."
        chunk_size = 15
        
        chunks = chunk_text(text, chunk_size)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Line one.")
        self.assertEqual(chunks[1], "Line two.")
        self.assertEqual(chunks[2], "Line three.")
    
    def test_sentence_breaks(self):
        """
        Test chunking breaks at sentences when no paragraph or newline breaks are available.
        """
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunk_size = 25
        
        chunks = chunk_text(text, chunk_size)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is sentence one.")
        self.assertEqual(chunks[1], "This is sentence two.")
        self.assertEqual(chunks[2], "This is sentence three.")
    
    def test_forced_breaks(self):
        """
        Test chunking forces breaks at chunk size when no natural breaks are available.
        """
        text = "ThisIsAVeryLongWordWithoutAnyNaturalBreaksInIt"
        chunk_size = 10
        chunk_overlap = 0
        
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], "ThisIsAVer")
        self.assertEqual(chunks[1], "yLongWordW")
        self.assertEqual(chunks[2], "ithoutAnyN")
        self.assertEqual(chunks[3], "aturalBrea")
    
    def test_chunk_with_overlap(self):
        """
        Test chunking with overlap.
        """
        text = "This is a long text that should be split into multiple chunks with overlap."
        chunk_size = 20
        chunk_overlap = 5
        
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        # Check that the chunks have overlap
        self.assertGreater(len(chunks), 1)
        for i in range(1, len(chunks)):
            prev_chunk_end = chunks[i-1][-5:] if len(chunks[i-1]) >= 5 else chunks[i-1]
            curr_chunk_start = chunks[i][:5] if len(chunks[i]) >= 5 else chunks[i]
            
            # Check that there is some overlap between chunks
            overlap = any(c in curr_chunk_start for c in prev_chunk_end)
            self.assertTrue(overlap)
    
    def test_chunk_with_metadata(self):
        """
        Test chunking with metadata.
        """
        text = "This is paragraph one.\n\nThis is paragraph two."
        metadata = {"source": "test", "author": "unittest"}
        
        result = chunk_text_with_metadata(text, metadata)
        
        self.assertEqual(len(result), 2)
        
        # Check that metadata is present in each chunk
        for i, chunk in enumerate(result):
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertEqual(chunk["metadata"]["source"], "test")
            self.assertEqual(chunk["metadata"]["author"], "unittest")
            self.assertEqual(chunk["metadata"]["chunk_index"], i)
            self.assertEqual(chunk["metadata"]["chunk_count"], 2)

if __name__ == "__main__":
    unittest.main() 