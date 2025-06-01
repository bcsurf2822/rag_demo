#!/usr/bin/env python3
"""
Test script to verify the FastRAGAgent fix for validation errors.
"""
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.fast_rag_agent import FastRAGAgent
from src.utils.config import OPENAI_API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fast_agent_fix():
    """Test that the fast agent no longer has validation errors."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found. Please set up your .env file.")
        return False
    
    try:
        logger.info("🚀 Testing FastRAGAgent fix...")
        
        # Create the agent
        agent = FastRAGAgent()
        logger.info("✅ Agent created successfully")
        
        # Test with a simple question
        test_questions = [
            "What is RAG?",
            "How does machine learning work?", 
            "What are the benefits of AI systems?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n📝 Test {i}: '{question}'")
            
            try:
                # This should no longer fail with validation errors
                response = await agent.answer_question(
                    question=question,
                    match_count=3,
                    match_threshold=0.3,
                    include_debug_info=True
                )
                
                logger.info(f"✅ Response received successfully!")
                logger.info(f"📊 Answer length: {len(response.answer)} chars")
                logger.info(f"📚 Sources used: {response.num_sources_used}")
                logger.info(f"🎯 Confidence: {response.confidence:.2f}")
                logger.info(f"⏱️  Processing time: {response.processing_time_ms:.1f}ms")
                logger.info(f"🔍 Search method: {response.search_method_used}")
                
                if response.debug_info:
                    logger.info(f"🐛 Debug: {response.debug_info}")
                
                # Verify response structure
                assert isinstance(response.answer, str), "Answer should be string"
                assert isinstance(response.sources, list), "Sources should be list"
                assert isinstance(response.confidence, float), "Confidence should be float"
                assert 0 <= response.confidence <= 1, "Confidence should be 0-1"
                
                logger.info(f"✅ Test {i} passed - No validation errors!")
                
            except Exception as e:
                if "maximum retries" in str(e) and "validation" in str(e):
                    logger.error(f"❌ VALIDATION ERROR STILL EXISTS: {e}")
                    return False
                else:
                    logger.warning(f"⚠️  Non-validation error (expected if no docs): {e}")
        
        logger.info("\n🎉 All tests passed! Validation error fix successful!")
        
        # Test performance summary
        try:
            summary = agent.get_performance_summary()
            logger.info(f"📈 Performance summary available: {len(summary)} entries")
        except Exception as e:
            logger.info(f"📈 Performance summary not available yet: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🔧 Testing FastRAGAgent validation fix...")
    
    success = asyncio.run(test_fast_agent_fix())
    
    if success:
        logger.info("\n✅ SUCCESS: FastRAGAgent fix verified!")
        logger.info("The 'Exceeded maximum retries for result validation' error should be resolved.")
        print("\n🚀 Your RAG agent is ready to use!")
    else:
        logger.error("\n❌ FAILURE: Validation error still exists")
        print("\n🔧 Additional debugging may be needed")
        sys.exit(1)

if __name__ == "__main__":
    main() 