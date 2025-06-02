"""
Enhanced Streamlit UI for the RAG AI Agent with advanced features.
"""
import streamlit as st
import asyncio
import time
import json
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


from src.agent.enhanced_rag_agent import EnhancedRAGAgent
from src.agent.fast_rag_agent import FastRAGAgent
from src.ingestion.document_processor import DocumentProcessor
from src.utils.file_loader import extract_file_content
from src.utils.observability import metrics_collector
from src.utils.response_formatter import ResponseFormatter
from src.utils.database import supabase_manager

# Configure page
st.set_page_config(
    page_title="Enhanced RAG AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitRAGInterface:
    """
    Enhanced Streamlit interface for the RAG AI Agent.
    """
    
    def __init__(self):
        """Initialize the interface."""
        # Initialize both agents
        self.enhanced_agent = EnhancedRAGAgent()
        self.fast_agent = FastRAGAgent()
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = None
        if "uploaded_docs" not in st.session_state:
            st.session_state.uploaded_docs = []
        if "selected_agent" not in st.session_state:
            st.session_state.selected_agent = "Fast"  # Default to fast agent
    
    def run(self):
        """Run the Streamlit interface."""
        st.title("RAG AI Agent Demonstration")
        st.markdown("*Retrieval-Augmented Generation with Semantic Search and Reranking*")
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Documents", "Analytics", "Debug"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_document_interface()
        
        with tab3:
            self.render_analytics_interface()
        
        with tab4:
            self.render_debug_interface()
    
    def render_sidebar(self):
        """Render the configuration sidebar."""
        st.sidebar.header("Agent Selection")
        
        # Agent selection
        agent_type = st.sidebar.selectbox(
            "Choose Agent Type",
            ["Fast", "Enhanced"],
            index=0 if st.session_state.selected_agent == "Fast" else 1,
            help="Fast: Optimized for speed (~2-5s response), Enhanced: Maximum accuracy (~5-15s response)"
        )
        st.session_state.selected_agent = agent_type
        
        # Performance indicator
        if agent_type == "Fast":
            st.sidebar.success("Fast Mode")
            st.sidebar.info("• 2-5 second response times\n• Intelligent caching\n• Parallel processing\n• Reduced token usage")
        else:
            st.sidebar.warning("Enhanced Mode - Better accuracy") 
            st.sidebar.info("• 5-15 second response \n• Adaptive search strategy\n•  Reranking\n• Comprehensive analysis")
        
        st.sidebar.header("Search Configuration")
        
        # Optimized settings based on agent type
        if agent_type == "Fast":
            match_count = st.sidebar.slider("Max Results", 3, 8, 5, help="Fewer results for faster processing")
            match_threshold = st.sidebar.slider("Similarity Threshold", 0.2, 0.8, 0.3, 0.1)
            rerank_top_k = st.sidebar.slider("Final Results", 2, 5, 3, help="Top results after reranking")
            use_caching = st.sidebar.checkbox("Enable Caching", True, help="Cache results for faster repeat queries")
        else:
            # Enhanced settings - simplified to use adaptive strategy only
            match_count = st.sidebar.slider("Max Results", 5, 15, 10, help="More results for better accuracy")
            match_threshold = st.sidebar.slider("Similarity Threshold", 0.3, 0.8, 0.7, 0.05, help="Higher threshold for quality")
            # No rerank_top_k needed for enhanced (it uses adaptive logic)
            # No caching option for enhanced (not implemented yet)
        
        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            include_debug = st.sidebar.checkbox("Include Debug Info", False)
            if agent_type == "Fast":
                parallel_timeout = st.sidebar.slider("Timeout (seconds)", 5, 15, 8)
        
        # Store in session state
        if agent_type == "Fast":
            st.session_state.search_config = {
                "agent_type": "Fast",
                "match_count": match_count,
                "match_threshold": match_threshold,
                "rerank_top_k": rerank_top_k,
                "use_caching": use_caching,
                "include_debug": include_debug,
                "parallel_timeout": parallel_timeout
            }
        else:
            st.session_state.search_config = {
                "agent_type": "Enhanced",
                "strategy": "Adaptive",  # Always use the most efficient strategy
                "match_count": match_count,
                "match_threshold": match_threshold,
                "include_debug": include_debug
            }
        
        # Performance summary
        st.sidebar.header("Performance")
        try:
            # Get summary from the appropriate agent
            if agent_type == "Fast":
                summary = self.fast_agent.get_performance_summary()
            else:
                summary = self.enhanced_agent.get_performance_summary()
                
            if "average_metrics" in summary:
                metrics = summary["average_metrics"]
                st.sidebar.metric("Avg Response Time", f"{metrics['total_time_ms']:.0f}ms")
                st.sidebar.metric("Avg Tokens", f"{metrics['total_tokens']:.0f}")
                st.sidebar.metric("Avg Confidence", f"{metrics['confidence_score']:.1%}")
                
                # Performance badge
                if metrics['total_time_ms'] < 3000:
                    st.sidebar.success("Excellent Performance")
                elif metrics['total_time_ms'] < 8000:
                    st.sidebar.warning("Good Performance")
                else:
                    st.sidebar.error("Slow Performance")
        except Exception:
            st.sidebar.info("No performance data available yet")
        
        # Cache management for fast agent
        if agent_type == "Fast":
            st.sidebar.header("Cache Management")
            if st.sidebar.button("Clear Cache"):
                self.fast_agent.clear_cache()
                st.sidebar.success("Cache cleared!")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("Response Details"):
                        metadata = message["metadata"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sources Used", metadata.get("num_sources_used", 0))
                        with col2:
                            st.metric("Processing Time", f"{metadata.get('processing_time_ms', 0):.0f}ms")
                        with col3:
                            st.metric("Confidence", f"{metadata.get('confidence', 0):.1%}")
                        
                        st.text(f"Search Method: {metadata.get('search_method_used', 'unknown')}")
                        
                        # Token usage
                        token_usage = metadata.get("token_usage", {})
                        if token_usage:
                            st.text(f"Tokens: {token_usage.get('total_tokens', 0)} "
                                   f"({token_usage.get('request_tokens', 0)} request + "
                                   f"{token_usage.get('response_tokens', 0)} response)")
                        
                        # Debug info if available - no nested expander
                        if metadata.get("debug_info"):
                            st.subheader("Debug Information")
                            st.text_area(
                                "Debug Details",
                                value=metadata["debug_info"],
                                height=150,
                                disabled=True,
                                label_visibility="collapsed"
                            )
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(self._generate_response(prompt))
                
                if response:
                    st.markdown(response.answer)
                    
                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "metadata": {
                            "num_sources_used": response.num_sources_used,
                            "search_method_used": response.search_method_used,
                            "processing_time_ms": response.processing_time_ms,
                            "confidence": response.confidence,
                            "token_usage": response.token_usage,
                            "debug_info": response.debug_info
                        }
                    })
                else:
                    st.error("Failed to generate response. Please try again.")
    
    async def _generate_response(self, prompt: str):
        """Generate response using the appropriate agent."""
        try:
            config = st.session_state.search_config
            agent_type = config.get("agent_type", "Fast")
            
            if agent_type == "Fast":
                # Use FastRAGAgent
                response = await self.fast_agent.answer_question(
                    question=prompt,
                    match_count=config["match_count"],
                    match_threshold=config["match_threshold"],
                    rerank_top_k=config["rerank_top_k"],
                    use_caching=config["use_caching"],
                    include_debug_info=config["include_debug"],
                    session_id=st.session_state.session_id
                )
            else:
                # Use EnhancedRAGAgent with existing logic
                strategy_map = {
                    "Adaptive": {"adaptive_search": True, "use_reranking": False, "use_multi_query": False},
                    "Multi-Query + Re-ranking": {"adaptive_search": False, "use_reranking": True, "use_multi_query": True},
                    "Re-ranking Only": {"adaptive_search": False, "use_reranking": True, "use_multi_query": False},
                    "Basic": {"adaptive_search": False, "use_reranking": False, "use_multi_query": False}
                }
                
                strategy_params = strategy_map.get(config["strategy"], strategy_map["Adaptive"])
                
                response = await self.enhanced_agent.answer_question(
                    question=prompt,
                    match_count=config["match_count"],
                    match_threshold=config["match_threshold"],
                    include_debug_info=config["include_debug"],
                    session_id=st.session_state.session_id,
                    **strategy_params
                )
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
    
    def render_document_interface(self):
        """Render the document upload and management interface."""
        st.header("Document Management")
        st.markdown("""
        This interface shows all documents in your knowledge base and allows you to upload new ones.
        
        - **Database Documents**: All permanently stored documents that can be queried
        - **Session Uploads**: Documents uploaded in the current session (also stored in database)
        - Click the 📋 button to view document details and content chunks
        """)
        
        # File upload section
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload text, PDF, or DOCX files to add to the knowledge base"
        )
        
        if uploaded_files:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file not in st.session_state.uploaded_docs:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = self._process_uploaded_file(uploaded_file)
                        if success:
                            st.session_state.uploaded_docs.append(uploaded_file)
                            st.success(f"Successfully processed {uploaded_file.name}")
                        else:
                            st.error(f"Failed to process {uploaded_file.name}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Show all available documents (database + current session)
        st.subheader("Knowledge Base Documents")
        
        # Get documents from database
        try:
            db_documents = supabase_manager.list_documents()
        except Exception as e:
            st.error(f"Error loading documents from database: {e}")
            db_documents = []
        
        # Refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh", help="Refresh document list from database"):
                st.rerun()
        
        # Show database documents
        if db_documents:
            st.write(f"**Database Documents ({len(db_documents)} total):**")
            
            # Add column headers
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
            with col1:
                st.write("**Document Title**")
            with col2:
                st.write("**Filename**")
            with col3:
                st.write("**Created Date**")
            with col4:
                st.write("**Chunks**")
            with col5:
                st.write("**Actions**")
            
            st.write("---")  # Separator line
            
            # Create a table-like display
            for i, doc in enumerate(db_documents):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                    
                    with col1:
                        st.text(f"📄 {doc.get('title', 'Unknown Title')}")
                    
                    with col2:
                        st.text(f"{doc.get('filename', 'Unknown File')}")
                    
                    with col3:
                        created_at = doc.get('created_at', '')
                        if created_at:
                            # Format date nicely
                            from datetime import datetime
                            try:
                                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                                st.text(formatted_date)
                            except:
                                st.text(created_at[:10])  # Just show date part
                        else:
                            st.text("Unknown Date")
                    
                    with col4:
                        # Get chunk count for this document
                        try:
                            chunks_response = supabase_manager.client.table("chunks").select("id").eq("document_id", doc['id']).execute()
                            chunk_count = len(chunks_response.data)
                            st.text(f"{chunk_count} chunks")
                        except:
                            st.text("? chunks")
                    
                    with col5:
                        # View details button
                        if st.button("📋", key=f"view_db_{doc['id']}", help="View document details"):
                            st.session_state.selected_doc_id = doc['id']
                
                # Show document details if selected
                if hasattr(st.session_state, 'selected_doc_id') and st.session_state.selected_doc_id == doc['id']:
                    with st.expander(f"📋 Details for: {doc.get('title', 'Unknown')}", expanded=True):
                        st.write(f"**Document ID:** {doc['id']}")
                        st.write(f"**Title:** {doc.get('title', 'Unknown')}")
                        st.write(f"**Filename:** {doc.get('filename', 'Unknown')}")
                        st.write(f"**Created:** {doc.get('created_at', 'Unknown')}")
                        
                        # Show chunks for this document
                        try:
                            chunks_response = supabase_manager.client.table("chunks").select("id, content, metadata").eq("document_id", doc['id']).execute()
                            chunks = chunks_response.data
                            
                            if chunks:
                                st.write(f"**Chunks ({len(chunks)} total):**")
                                for j, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                                    with st.container():
                                        st.write(f"*Chunk {j+1}:*")
                                        content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                                        st.text_area(f"Content Preview", content_preview, height=100, disabled=True, key=f"chunk_{doc['id']}_{j}")
                                
                                if len(chunks) > 3:
                                    st.info(f"... and {len(chunks) - 3} more chunks")
                            else:
                                st.warning("No chunks found for this document")
                        except Exception as e:
                            st.error(f"Error loading chunks: {e}")
                        
                        # Close button
                        if st.button("Close Details", key=f"close_{doc['id']}"):
                            if hasattr(st.session_state, 'selected_doc_id'):
                                delattr(st.session_state, 'selected_doc_id')
                            st.rerun()
        else:
            st.info("No documents found in the database.")
        
        # Show current session uploads if any
        if st.session_state.uploaded_docs:
            st.write("---")
            st.write(f"**Current Session Uploads ({len(st.session_state.uploaded_docs)} total):**")
            st.info("These documents have been uploaded in this session and are stored in the database.")
            
            # Add column headers for session uploads
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write("**Document Name**")
            with col2:
                st.write("**File Size**")
            with col3:
                st.write("**Actions**")
            
            st.write("---")  # Separator line
            
            # Create a more detailed display
            for i, doc in enumerate(st.session_state.uploaded_docs):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📄 {doc.name}")
                
                with col2:
                    st.text(f"{doc.size:,} bytes")
                
                with col3:
                    if st.button(f"Remove from View", key=f"remove_{i}", help="Remove from session view (document remains in database)"):
                        st.session_state.uploaded_docs.remove(doc)
                        st.rerun()
        
        # Document statistics
        total_db_docs = len(db_documents)
        total_session_docs = len(st.session_state.uploaded_docs)
        total_unique_docs = total_db_docs  # DB documents are the source of truth
        
        st.subheader("Document Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Documents", total_db_docs)
        with col2:
            st.metric("Session Uploads", total_session_docs)
        with col3:
            if st.session_state.uploaded_docs:
                total_session_size = sum(doc.size for doc in st.session_state.uploaded_docs)
                st.metric("Session Upload Size", f"{total_session_size:,} bytes")
            else:
                st.metric("Total in Knowledge Base", total_unique_docs)
    
    def _process_uploaded_file(self, uploaded_file) -> bool:
        """Process an uploaded file."""
        try:
            # Use the existing file loader utility to extract content
            file_data = extract_file_content(uploaded_file)
            
            # Process the document using the DocumentProcessor
            result = DocumentProcessor.process_file_content(file_data)
            
            # Check if processing was successful
            if result and result.get("chunk_count", 0) > 0:
                return True
            else:
                st.warning(f"Document processed but no chunks were created from {uploaded_file.name}")
                return False
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return False
    
    def render_analytics_interface(self):
        """Render the analytics and metrics interface."""
        st.header("Analytics Dashboard")
        
        try:
            # Get performance summary
            summary = self.enhanced_agent.get_performance_summary()
            
            if "message" in summary:
                st.info(summary["message"])
                return
            
            # Key metrics
            st.subheader("Key Performance Indicators")
            metrics = summary["average_metrics"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", summary["total_queries"])
            with col2:
                st.metric("Avg Response Time", f"{metrics['total_time_ms']:.1f}ms")
            with col3:
                st.metric("Avg Token Usage", f"{metrics['total_tokens']:.0f}")
            with col4:
                st.metric("Avg Confidence", f"{metrics['confidence_score']:.1%}")
            
            # Performance trends (mock data for demonstration)
            st.subheader("Performance Trends")
            
            # Create sample trend data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            trend_data = pd.DataFrame({
                'Date': dates,
                'Response Time (ms)': [metrics['total_time_ms'] + (i % 5 - 2) * 100 for i in range(30)],
                'Confidence Score': [metrics['confidence_score'] + (i % 7 - 3) * 0.02 for i in range(30)]
            })
            
            # Response time chart
            fig_time = px.line(trend_data, x='Date', y='Response Time (ms)', 
                              title='Response Time Trend')
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Confidence score chart
            fig_conf = px.line(trend_data, x='Date', y='Confidence Score', 
                              title='Confidence Score Trend')
            st.plotly_chart(fig_conf, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def render_debug_interface(self):
        """Render the debug and system information interface."""
        st.header("Debug Information")
        
        # System status
        st.subheader("System Status")
        st.success("Enhanced RAG Agent - Online")
        st.success("Vector Search - Active")
        st.success("Semantic Re-ranking - Available")
        st.success("Multi-Query Expansion - Available")
        
        # Configuration display
        st.subheader("Current Configuration")
        if "search_config" in st.session_state:
            config = st.session_state.search_config
            st.json(config)
        
        # Session information
        st.subheader("Session Information")
        st.text(f"Session ID: {st.session_state.session_id or 'Not started'}")
        st.text(f"Messages in History: {len(st.session_state.messages)}")
        st.text(f"Documents Uploaded: {len(st.session_state.uploaded_docs)}")
        
        # Raw metrics data
        st.subheader("Raw Metrics Data")
        try:
            summary = self.enhanced_agent.get_performance_summary()
            st.json(summary)
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")
        
        # Clear session data
        st.subheader("Session Management")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
        
        if st.button("Reset Session"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.uploaded_docs = []
            st.success("Session reset!")

def main():
    """Main function to run the Streamlit app."""
    interface = StreamlitRAGInterface()
    interface.run()

if __name__ == "__main__":
    main() 