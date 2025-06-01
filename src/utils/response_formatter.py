"""
Response formatting utilities for enhanced RAG output.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """
    Represents a citation with source information.
    """
    source_id: int
    title: str
    similarity_score: float
    chunk_content: str
    page_number: Optional[int] = None
    section: Optional[str] = None

class ResponseFormatter:
    """
    Formats RAG responses with markdown, citations, and improved structure.
    """
    
    @staticmethod
    def format_response_with_citations(response: str, 
                                     sources: List[Dict[str, Any]],
                                     include_confidence: bool = True,
                                     confidence_score: Optional[float] = None) -> str:
        """
        Format a response with proper citations and markdown.
        
        Args:
            response: The raw agent response.
            sources: List of source chunks used.
            include_confidence: Whether to include confidence score.
            confidence_score: The confidence score (0-1).
            
        Returns:
            Formatted response with citations.
        """
        # Extract unique sources for citation
        citations = ResponseFormatter._extract_citations(sources)
        
        # Format the main response with markdown
        formatted_response = ResponseFormatter._apply_markdown_formatting(response)
        
        # Add inline citations if sources are mentioned
        formatted_response = ResponseFormatter._add_inline_citations(formatted_response, citations)
        
        # Add sources section
        if citations:
            formatted_response += "\n\n" + ResponseFormatter._format_sources_section(citations)
        
        # Add confidence indicator
        if include_confidence and confidence_score is not None:
            formatted_response += f"\n\n**Confidence Score:** {confidence_score:.1%}"
        
        return formatted_response
    
    @staticmethod
    def _extract_citations(sources: List[Dict[str, Any]]) -> List[Citation]:
        """
        Extract and deduplicate citations from sources.
        
        Args:
            sources: List of source chunks.
            
        Returns:
            List of unique citations.
        """
        citations = []
        seen_titles: Set[str] = set()
        
        for i, source in enumerate(sources):
            metadata = source.get('metadata', {})
            title = metadata.get('title', 'Unknown Source')
            
            # Only add if we haven't seen this title
            if title not in seen_titles:
                citations.append(Citation(
                    source_id=i + 1,
                    title=title,
                    similarity_score=source.get('rerank_score', source.get('similarity', 0)),
                    chunk_content=source.get('content', ''),
                    page_number=metadata.get('page_number'),
                    section=metadata.get('section')
                ))
                seen_titles.add(title)
        
        return citations
    
    @staticmethod
    def _apply_markdown_formatting(text: str) -> str:
        """
        Apply basic markdown formatting to improve readability.
        
        Args:
            text: The raw text to format.
            
        Returns:
            Text with markdown formatting.
        """
        # Convert numbered lists to proper markdown
        text = re.sub(r'^(\d+)\.\s+(.+)$', r'1. **\2**', text, flags=re.MULTILINE)
        
        # Convert bullet points to proper markdown
        text = re.sub(r'^[-*]\s+(.+)$', r'- \1', text, flags=re.MULTILINE)
        
        # Emphasize key terms (simple heuristic)
        key_terms = ['important', 'note', 'key', 'critical', 'essential', 'significant']
        for term in key_terms:
            pattern = rf'\b({term})\b'
            text = re.sub(pattern, r'**\1**', text, flags=re.IGNORECASE)
        
        # Add emphasis to questions
        text = re.sub(r'^(.+\?)\s*$', r'**\1**', text, flags=re.MULTILINE)
        
        return text
    
    @staticmethod
    def _add_inline_citations(text: str, citations: List[Citation]) -> str:
        """
        Add inline citations where source titles are mentioned.
        
        Args:
            text: The text to add citations to.
            citations: List of available citations.
            
        Returns:
            Text with inline citations added.
        """
        for citation in citations:
            # Look for mentions of the source title (case insensitive)
            title_words = citation.title.lower().split()
            
            # Create different patterns to match title mentions
            patterns = [
                citation.title,  # Exact title
                ' '.join(title_words[:3]),  # First 3 words
                ' '.join(title_words[-2:]),  # Last 2 words
            ]
            
            for pattern in patterns:
                if len(pattern) > 5:  # Only for meaningful patterns
                    # Add citation reference
                    pattern_regex = re.escape(pattern)
                    replacement = f"{pattern} [{citation.source_id}]"
                    text = re.sub(f'\\b{pattern_regex}\\b', replacement, text, 
                                count=1, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _format_sources_section(citations: List[Citation]) -> str:
        """
        Format the sources section with proper citations.
        
        Args:
            citations: List of citations to format.
            
        Returns:
            Formatted sources section.
        """
        sources_section = "## 📚 Sources\n\n"
        
        for citation in citations:
            sources_section += f"**[{citation.source_id}]** {citation.title}"
            
            if citation.similarity_score > 0:
                sources_section += f" *(relevance: {citation.similarity_score:.1%})*"
            
            if citation.section:
                sources_section += f" - Section: {citation.section}"
            
            if citation.page_number:
                sources_section += f" - Page {citation.page_number}"
            
            sources_section += "\n\n"
        
        return sources_section.rstrip()
    
    @staticmethod
    def format_search_debug_info(sources: List[Dict[str, Any]], 
                                search_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Format debug information about the search process.
        
        Args:
            sources: List of source chunks.
            search_metrics: Optional search metrics.
            
        Returns:
            Formatted debug information.
        """
        debug_info = "## 🔍 Search Debug Information\n\n"
        
        if search_metrics:
            debug_info += f"**Search Time:** {search_metrics.get('search_time_ms', 0):.1f}ms\n"
            debug_info += f"**Results Found:** {search_metrics.get('num_results', 0)}\n"
            debug_info += f"**Average Relevance:** {search_metrics.get('avg_similarity', 0):.2f}\n"
            
            if search_metrics.get('used_reranking'):
                debug_info += "**Re-ranking:** ✅ Used\n"
            
            if search_metrics.get('used_multi_query'):
                debug_info += "**Multi-query Expansion:** ✅ Used\n"
                expansion_queries = search_metrics.get('expansion_queries', [])
                if expansion_queries:
                    debug_info += f"**Expansion Queries:** {', '.join(expansion_queries)}\n"
            
            debug_info += "\n"
        
        if sources:
            debug_info += "### Retrieved Chunks\n\n"
            for i, source in enumerate(sources):
                metadata = source.get('metadata', {})
                similarity = source.get('rerank_score', source.get('similarity', 0))
                content_preview = source.get('content', '')[:100] + "..."
                
                debug_info += f"**Chunk {i+1}** from *{metadata.get('title', 'Unknown')}*\n"
                debug_info += f"- Relevance: {similarity:.3f}\n"
                debug_info += f"- Content: {content_preview}\n\n"
        
        return debug_info
    
    @staticmethod
    def create_response_summary(query: str, response: str, 
                              num_sources: int, 
                              confidence: float,
                              processing_time_ms: float) -> str:
        """
        Create a summary card for the response.
        
        Args:
            query: The original query.
            response: The generated response.
            num_sources: Number of sources used.
            confidence: Confidence score.
            processing_time_ms: Processing time in milliseconds.
            
        Returns:
            Formatted summary card.
        """
        summary = "## 📊 Response Summary\n\n"
        summary += f"**Query:** {query}\n"
        summary += f"**Response Length:** {len(response)} characters\n"
        summary += f"**Sources Used:** {num_sources}\n"
        summary += f"**Confidence:** {confidence:.1%}\n"
        summary += f"**Processing Time:** {processing_time_ms:.1f}ms\n"
        
        # Add quality indicators
        if confidence >= 0.8:
            summary += "**Quality:** 🟢 High confidence\n"
        elif confidence >= 0.6:
            summary += "**Quality:** 🟡 Medium confidence\n"
        else:
            summary += "**Quality:** 🔴 Low confidence\n"
        
        return summary 