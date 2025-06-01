"""
Module for chunking text documents.
"""
import logging
import re
from typing import List, Dict, Any, Optional

import tiktoken

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def get_token_count(text: str, model: str = OPENAI_EMBEDDING_MODEL) -> int:
    """
    Get token count for a text string using the tiktoken library.
    
    Args:
        text: The text to count tokens for.
        model: The model to use for tokenization.
        
    Returns:
        The number of tokens in the text.
    """
    try:
        # Use cl100k_base for text-embedding-3-small
        encoding_name = "cl100k_base"
        if "text-embedding-ada" in model:
            encoding_name = "p50k_base"
            
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens, falling back to character estimate: {e}")
        # Fallback to approximate tokens (1 token ~= 4 chars for English text)
        return len(text) // 4

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
              chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: The text to chunk.
        chunk_size: The maximum size of each chunk in characters.
        chunk_overlap: The overlap between chunks in characters.
        
    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    
    # Clean the text
    text = text.strip()
    
    # If text is smaller than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're at the end of the text, just use the end
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a paragraph
        paragraph_break = text.rfind("\n\n", start, end)
        if paragraph_break != -1 and paragraph_break > start:
            chunks.append(text[start:paragraph_break].strip())
            start = paragraph_break
            continue
        
        # Try to break at a newline
        newline_break = text.rfind("\n", start, end)
        if newline_break != -1 and newline_break > start:
            chunks.append(text[start:newline_break].strip())
            start = newline_break
            continue
        
        # Try to break at a sentence (period, question mark, or exclamation mark followed by a space or newline)
        sentence_end_regex = re.compile(r'[.!?][\s\n]', re.MULTILINE)
        sentence_matches = list(sentence_end_regex.finditer(text[start:end]))
        if sentence_matches:
            # Get the last sentence end match
            last_match = sentence_matches[-1]
            sentence_break = start + last_match.start() + 1  # +1 to include the punctuation mark
            chunks.append(text[start:sentence_break].strip())
            start = sentence_break
            continue
        
        # If all else fails, just break at the chunk size
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    
    # Filter out any empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    
    # Check if any chunk might exceed the token limit for embeddings
    max_token_limit = 8000  # OpenAI embedding model limit for text-embedding-3-small
    
    # Collect chunks that may need further splitting
    final_chunks = []
    for chunk in chunks:
        token_count = get_token_count(chunk)
        if token_count > max_token_limit:
            logger.warning(f"Chunk exceeds token limit ({token_count} > {max_token_limit}), recursively splitting")
            # If chunk is too large, recursively split it using a smaller chunk size
            smaller_chunk_size = int(chunk_size * max_token_limit / token_count)
            smaller_chunks = chunk_text(chunk, smaller_chunk_size, int(chunk_overlap * max_token_limit / token_count))
            final_chunks.extend(smaller_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def chunk_text_with_metadata(text: str, metadata: Optional[Dict[str, Any]] = None, 
                           chunk_size: int = CHUNK_SIZE, 
                           chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into chunks and attach metadata to each chunk.
    
    Args:
        text: The text to chunk.
        metadata: Metadata to attach to each chunk.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The overlap between chunks.
        
    Returns:
        A list of dictionaries containing the chunk text and metadata.
    """
    if metadata is None:
        metadata = {}
    
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Extract document title for context preservation
    document_title = metadata.get("title", "")
    
    result = []
    for i, chunk in enumerate(chunks):
        # Create a new metadata dict for each chunk
        chunk_metadata = metadata.copy()
        
        # Add chunk-specific metadata
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_count"] = len(chunks)
        
        # For non-first chunks, prepend a context header with document title
        # This helps maintain context when embedding middle chunks
        content = chunk
        if i > 0 and document_title:
            # Only add if not already at the beginning of the text
            if document_title not in chunk[:len(document_title) + 10]:
                content = f"Document: {document_title}\n\n{chunk}"
        
        result.append({
            "content": content,
            "metadata": chunk_metadata
        })
    
    # Log token counts for all chunks to help debug embedding issues
    for i, chunk in enumerate(result):
        tokens = get_token_count(chunk["content"])
        logger.debug(f"Chunk {i}: {tokens} tokens, {len(chunk['content'])} characters")
    
    return result 