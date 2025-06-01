"""
Observability module for tracking RAG system metrics.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """
    Token usage tracking for LLM interactions.
    """
    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    
    def __post_init__(self):
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.request_tokens + self.response_tokens

@dataclass
class SearchMetrics:
    """
    Search performance metrics.
    """
    query: str = ""
    search_time_ms: float = 0.0
    num_results: int = 0
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0
    used_reranking: bool = False
    used_multi_query: bool = False
    used_cache: bool = False  # Add caching support
    expansion_queries: List[str] = field(default_factory=list)

@dataclass
class RAGMetrics:
    """
    Complete RAG pipeline metrics.
    """
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    query: str = ""
    response: str = ""
    confidence_score: float = 0.0
    
    # Timing metrics
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    
    # Token usage
    token_usage: Optional[TokenUsage] = None
    
    # Search metrics
    search_metrics: Optional[SearchMetrics] = None
    
    # Sources and context
    num_sources: int = 0
    context_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "response_length": len(self.response),
            "confidence_score": self.confidence_score,
            "total_time_ms": self.total_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "token_usage": {
                "request_tokens": self.token_usage.request_tokens if self.token_usage else 0,
                "response_tokens": self.token_usage.response_tokens if self.token_usage else 0,
                "total_tokens": self.token_usage.total_tokens if self.token_usage else 0,
                "model_name": self.token_usage.model_name if self.token_usage else ""
            },
            "search_metrics": {
                "search_time_ms": self.search_metrics.search_time_ms if self.search_metrics else 0,
                "num_results": self.search_metrics.num_results if self.search_metrics else 0,
                "avg_similarity": self.search_metrics.avg_similarity if self.search_metrics else 0,
                "used_reranking": self.search_metrics.used_reranking if self.search_metrics else False,
                "used_multi_query": self.search_metrics.used_multi_query if self.search_metrics else False
            },
            "num_sources": self.num_sources
        }

class MetricsCollector:
    """
    Centralized metrics collection for the RAG system.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics_history: List[RAGMetrics] = []
        self.current_session: Optional[str] = None
    
    def start_session(self, session_id: str):
        """Start a new metrics collection session."""
        self.current_session = session_id
        logger.info(f"Started metrics session: {session_id}")
    
    def log_rag_metrics(self, metrics: RAGMetrics):
        """
        Log RAG pipeline metrics.
        
        Args:
            metrics: The RAG metrics to log.
        """
        # Set session ID if not provided
        if not metrics.session_id and self.current_session:
            metrics.session_id = self.current_session
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log summary
        logger.info(
            f"RAG Query Complete - "
            f"Time: {metrics.total_time_ms:.1f}ms, "
            f"Tokens: {metrics.token_usage.total_tokens if metrics.token_usage else 0}, "
            f"Sources: {metrics.num_sources}, "
            f"Confidence: {metrics.confidence_score:.2f}"
        )
        
        # Log detailed metrics in debug mode
        logger.debug(f"Detailed RAG metrics: {json.dumps(metrics.to_dict(), indent=2)}")
    
    def record_rag_interaction(self, metrics: RAGMetrics):
        """
        Alias for log_rag_metrics to maintain compatibility with fast agent.
        
        Args:
            metrics: The RAG metrics to record.
        """
        self.log_rag_metrics(metrics)
    
    def get_session_metrics(self, session_id: str) -> List[RAGMetrics]:
        """
        Get all metrics for a specific session.
        
        Args:
            session_id: The session ID to filter by.
            
        Returns:
            List of metrics for the session.
        """
        return [m for m in self.metrics_history if m.session_id == session_id]
    
    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            last_n: Number of recent queries to analyze (None for all).
            
        Returns:
            Performance summary dictionary.
        """
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics:
            return {"message": "No metrics available"}
        
        # Calculate averages
        avg_total_time = sum(m.total_time_ms for m in metrics) / len(metrics)
        avg_retrieval_time = sum(m.retrieval_time_ms for m in metrics) / len(metrics)
        avg_generation_time = sum(m.generation_time_ms for m in metrics) / len(metrics)
        avg_tokens = sum(m.token_usage.total_tokens if m.token_usage else 0 for m in metrics) / len(metrics)
        avg_sources = sum(m.num_sources for m in metrics) / len(metrics)
        avg_confidence = sum(m.confidence_score for m in metrics) / len(metrics)
        
        return {
            "total_queries": len(metrics),
            "time_period": f"{metrics[0].timestamp} to {metrics[-1].timestamp}",
            "average_metrics": {
                "total_time_ms": round(avg_total_time, 2),
                "retrieval_time_ms": round(avg_retrieval_time, 2),
                "generation_time_ms": round(avg_generation_time, 2),
                "total_tokens": round(avg_tokens, 1),
                "num_sources": round(avg_sources, 1),
                "confidence_score": round(avg_confidence, 3)
            },
            "performance_percentiles": {
                "p50_total_time": self._percentile([m.total_time_ms for m in metrics], 50),
                "p95_total_time": self._percentile([m.total_time_ms for m in metrics], 95),
                "p99_total_time": self._percentile([m.total_time_ms for m in metrics], 99)
            }
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class PerformanceTimer:
    """
    Context manager for timing operations.
    """
    
    def __init__(self, operation_name: str = "operation"):
        """
        Initialize the performance timer.
        
        Args:
            operation_name: Name of the operation being timed.
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0.0
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        logger.debug(f"{self.operation_name} took {self.duration_ms:.2f}ms")

# Global metrics collector instance
metrics_collector = MetricsCollector() 