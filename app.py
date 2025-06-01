"""
Main Streamlit application for the Enhanced RAG AI Agent.
Runs the advanced UI with all RAG components active.
"""
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import and run the enhanced UI
from enhanced_streamlit_ui import main

if __name__ == "__main__":
    logger.info("Starting Enhanced RAG AI Agent with all components active...")
    main() 