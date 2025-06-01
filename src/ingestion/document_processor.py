"""
Document processor module for ingesting documents.
"""
import logging
from typing import Dict, Any, List, Optional

from src.utils.database import supabase_manager
from src.utils.embeddings import generate_embedding, generate_batch_embeddings
from src.utils.text_chunking import chunk_text_with_metadata
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Class for processing and ingesting documents.
    """
    
    @staticmethod
    def process_document(content: str, title: str, filename: str, 
                        chunk_size: int = CHUNK_SIZE,
                        chunk_overlap: int = CHUNK_OVERLAP) -> Dict[str, Any]:
        """
        Process a document by chunking it, generating embeddings, and storing in the database.
        
        Args:
            content: The document content.
            title: The document title.
            filename: The original filename.
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The overlap between chunks.
            
        Returns:
            A dictionary containing the document metadata and processing results.
        """
        logger.info(f"Processing document: {title}")
        
        # Store the document
        document = supabase_manager.store_document(title, filename)
        document_id = document["id"]
        
        # Create metadata common to all chunks
        metadata = {
            "document_id": document_id,
            "title": title,
            "filename": filename
        }
        
        # Chunk the document
        chunks = chunk_text_with_metadata(content, metadata, chunk_size, chunk_overlap)
        
        # Process chunks in batches to optimize embedding generation
        if len(chunks) > 0:
            # Extract contents for batch embedding
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings for all chunks at once
            embeddings = generate_batch_embeddings(chunk_texts)
            
            # Store chunks with embeddings
            for i, chunk in enumerate(chunks):
                supabase_manager.store_chunk(
                    document_id=document_id,
                    content=chunk["content"],
                    embedding=embeddings[i],
                    metadata=chunk["metadata"]
                )
            
            logger.info(f"Processed document: {title} with {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "title": title,
                "filename": filename,
                "chunk_count": len(chunks)
            }
        else:
            logger.warning(f"No chunks created for document: {title}")
            return {
                "document_id": document_id,
                "title": title,
                "filename": filename,
                "chunk_count": 0
            }
    
    @staticmethod
    def process_file_content(file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a file's content from file_data dictionary.
        
        Args:
            file_data: Dictionary containing file metadata and content.
            
        Returns:
            A dictionary containing processing results.
        """
        return DocumentProcessor.process_document(
            content=file_data["content"],
            title=file_data["title"],
            filename=file_data["filename"]
        ) 