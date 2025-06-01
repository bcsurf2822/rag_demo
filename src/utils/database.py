"""
Database connection module for Supabase.
"""
import logging
from typing import Optional, Dict, Any, List

from supabase import create_client, Client

from src.utils.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

class SupabaseManager:
    """
    A class to manage Supabase connections and operations.
    """
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern to ensure only one instance of SupabaseManager exists.
        """
        if cls._instance is None:
            cls._instance = super(SupabaseManager, cls).__new__(cls)
            cls._instance._client = None
        return cls._instance
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client, creating it if it doesn't exist.
        
        Returns:
            Client: The Supabase client.
        
        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_KEY is not set.
        """
        if self._client is None:
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            
            self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client created")
        
        return self._client
    
    def store_document(self, title: str, filename: str) -> Dict[str, Any]:
        """
        Store a document in the database.
        
        Args:
            title: The document title.
            filename: The original filename.
            
        Returns:
            The created document record.
        """
        response = self.client.table("documents").insert({
            "title": title,
            "filename": filename
        }).execute()
        
        if len(response.data) == 0:
            raise ValueError("Failed to insert document")
            
        return response.data[0]
    
    def store_chunk(self, document_id: int, content: str, embedding: List[float], 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a document chunk with its embedding in the database.
        
        Args:
            document_id: The parent document ID.
            content: The text content of the chunk.
            embedding: The vector embedding of the chunk.
            metadata: Optional metadata about the chunk.
            
        Returns:
            The created chunk record.
        """
        if metadata is None:
            metadata = {}
            
        response = self.client.table("chunks").insert({
            "document_id": document_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }).execute()
        
        if len(response.data) == 0:
            raise ValueError("Failed to insert chunk")
            
        return response.data[0]
    
    def query_similar_chunks(self, query_embedding: List[float], 
                            match_threshold: float = 0.7, 
                            match_count: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to the query embedding.
        
        Args:
            query_embedding: The embedding vector to match against.
            match_threshold: The similarity threshold (0-1).
            match_count: The maximum number of matches to return.
            
        Returns:
            A list of matching chunks with their similarity scores.
        """
        response = self.client.rpc(
            "match_chunks", 
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }
        ).execute()
        
        return response.data
    
    def get_document_by_id(self, document_id: int) -> Dict[str, Any]:
        """
        Get a document by its ID.
        
        Args:
            document_id: The document ID.
            
        Returns:
            The document record.
        """
        response = self.client.table("documents").select("*").eq("id", document_id).execute()
        
        if len(response.data) == 0:
            raise ValueError(f"Document with ID {document_id} not found")
            
        return response.data[0]
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the database.
        
        Returns:
            A list of document records.
        """
        response = self.client.table("documents").select("*").order("created_at", desc=True).execute()
        return response.data
    
    def log_some_chunk_embeddings(self):
        """
        Log the first 3 stored chunk embeddings and their content snippets for debugging.
        """
        response = self.client.table("chunks").select("embedding,content").limit(3).execute()
        for i, row in enumerate(response.data):
            logger.info(f"Chunk {i} embedding (first 5): {row['embedding'][:5]}, content: {row['content'][:100]}")

# Create a singleton instance
supabase_manager = SupabaseManager() 