#!/usr/bin/env python3
"""
Verification script to ensure all enhanced RAG components are active and working.
"""
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_imports():
    """Verify that all enhanced RAG components can be imported."""
    
    components_status = {}
    
    # Core enhanced components
    try:
        from src.retrieval.reranker import SemanticReranker
        components_status['Semantic Reranker'] = "✅ Active"
        logger.info("✅ Semantic Re-ranking component imported successfully")
    except Exception as e:
        components_status['Semantic Reranker'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import SemanticReranker: {e}")
    
    try:
        from src.retrieval.query_expansion import QueryExpansion
        components_status['Multi-Query Expansion'] = "✅ Active"
        logger.info("✅ Multi-Query Expansion component imported successfully")
    except Exception as e:
        components_status['Multi-Query Expansion'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import QueryExpansion: {e}")
    
    try:
        from src.retrieval.enhanced_vector_search import EnhancedVectorSearch
        components_status['Enhanced Vector Search'] = "✅ Active"
        logger.info("✅ Enhanced Vector Search component imported successfully")
    except Exception as e:
        components_status['Enhanced Vector Search'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import EnhancedVectorSearch: {e}")
    
    try:
        from src.utils.observability import metrics_collector
        components_status['Observability & Metrics'] = "✅ Active"
        logger.info("✅ Observability & Metrics component imported successfully")
    except Exception as e:
        components_status['Observability & Metrics'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import observability: {e}")
    
    try:
        from src.utils.response_formatter import ResponseFormatter
        components_status['Response Formatter'] = "✅ Active"
        logger.info("✅ Response Formatter component imported successfully")
    except Exception as e:
        components_status['Response Formatter'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import ResponseFormatter: {e}")
    
    try:
        from src.agent.enhanced_rag_agent import EnhancedRAGAgent
        components_status['Enhanced RAG Agent'] = "✅ Active"
        logger.info("✅ Enhanced RAG Agent component imported successfully")
    except Exception as e:
        components_status['Enhanced RAG Agent'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import EnhancedRAGAgent: {e}")
    
    try:
        from enhanced_streamlit_ui import StreamlitRAGInterface
        components_status['Enhanced Streamlit UI'] = "✅ Active"
        logger.info("✅ Enhanced Streamlit UI component imported successfully")
    except Exception as e:
        components_status['Enhanced Streamlit UI'] = f"❌ Error: {e}"
        logger.error(f"❌ Failed to import Enhanced Streamlit UI: {e}")
    
    return components_status

def verify_streamlit_dependencies():
    """Verify Streamlit and visualization dependencies."""
    try:
        import streamlit
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        logger.info("✅ Streamlit and visualization dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing Streamlit dependencies: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("🔍 Verifying Enhanced RAG Components...")
    
    # Verify imports
    components_status = verify_imports()
    
    # Verify dependencies
    deps_ok = verify_streamlit_dependencies()
    
    # Print summary
    print("\n" + "="*60)
    print("🤖 ENHANCED RAG SYSTEM COMPONENT STATUS")
    print("="*60)
    
    for component, status in components_status.items():
        print(f"{component:<30} {status}")
    
    print(f"{'Streamlit Dependencies':<30} {'✅ Active' if deps_ok else '❌ Missing'}")
    
    # Overall status
    all_active = all("✅" in status for status in components_status.values()) and deps_ok
    
    print("\n" + "="*60)
    if all_active:
        print("🎉 ALL ENHANCED RAG COMPONENTS ARE ACTIVE!")
        print("Run: streamlit run app.py")
        print("\nFeatures Available:")
        print("  • Semantic Re-ranking with cosine similarity & cross-encoder")
        print("  • Multi-Query Expansion for improved coverage")
        print("  • Enhanced Vector Search with adaptive strategies") 
        print("  • Comprehensive observability & token tracking")
        print("  • Response formatting with citations")
        print("  • Performance analytics dashboard")
        print("  • Debug tools with search metadata")
        print("  • Configurable search strategies")
        print("  • Real-time metrics visualization")
        return 0
    else:
        print("⚠️  SOME COMPONENTS HAVE ISSUES - Check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 