"""
Performance testing script to compare original vs optimized RAG agents.
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any

from src.agent.enhanced_rag_agent import EnhancedRAGAgent
from src.agent.fast_rag_agent import FastRAGAgent

class PerformanceTester:
    """
    Class to test and compare RAG agent performance.
    """
    
    def __init__(self):
        """Initialize both agents for testing."""
        print("Initializing agents...")
        self.enhanced_agent = EnhancedRAGAgent()
        self.fast_agent = FastRAGAgent()
        
        # Test questions of varying complexity
        self.test_questions = [
            "What is machine learning?",
            "How does natural language processing work?",
            "Explain the differences between supervised and unsupervised learning",
            "What are the key components of a neural network and how do they work together?",
            "How can I implement a recommendation system using collaborative filtering and what are the main challenges?",
        ]
    
    async def benchmark_agent(self, agent, agent_name: str, questions: List[str]) -> Dict[str, Any]:
        """
        Benchmark an agent with a list of questions.
        
        Args:
            agent: The agent to test (EnhancedRAGAgent or FastRAGAgent).
            agent_name: Name for logging.
            questions: List of test questions.
            
        Returns:
            Performance metrics dictionary.
        """
        print(f"\n{'='*50}")
        print(f"Testing {agent_name}")
        print(f"{'='*50}")
        
        response_times = []
        token_counts = []
        confidence_scores = []
        successful_responses = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {question[:50]}...")
            
            start_time = time.time()
            
            try:
                if agent_name == "FastRAGAgent":
                    response = await agent.answer_question(
                        question=question,
                        match_count=5,
                        rerank_top_k=3,
                        use_caching=True
                    )
                else:
                    response = await agent.answer_question(
                        question=question,
                        match_count=10,
                        use_reranking=True,
                        adaptive_search=True
                    )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                response_times.append(response_time)
                token_counts.append(response.token_usage.get('total_tokens', 0))
                confidence_scores.append(response.confidence)
                successful_responses += 1
                
                print(f"  ✅ Response time: {response_time:.1f}ms")
                print(f"  📊 Confidence: {response.confidence:.2f}")
                print(f"  🔤 Tokens: {response.token_usage.get('total_tokens', 0)}")
                print(f"  📝 Sources: {response.num_sources_used}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
        
        # Calculate statistics
        if response_times:
            metrics = {
                "agent_name": agent_name,
                "successful_responses": successful_responses,
                "total_questions": len(questions),
                "success_rate": successful_responses / len(questions),
                "avg_response_time_ms": statistics.mean(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "std_response_time_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "avg_tokens": statistics.mean(token_counts) if token_counts else 0,
                "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
                "response_times": response_times,
                "token_counts": token_counts,
                "confidence_scores": confidence_scores
            }
        else:
            metrics = {
                "agent_name": agent_name,
                "successful_responses": 0,
                "total_questions": len(questions),
                "success_rate": 0.0,
                "error": "No successful responses"
            }
        
        return metrics
    
    def print_comparison(self, enhanced_metrics: Dict[str, Any], fast_metrics: Dict[str, Any]):
        """
        Print a detailed comparison of the two agents.
        
        Args:
            enhanced_metrics: Metrics from EnhancedRAGAgent.
            fast_metrics: Metrics from FastRAGAgent.
        """
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        print(f"\n📊 RESPONSE TIMES:")
        print(f"  Enhanced Agent: {enhanced_metrics.get('avg_response_time_ms', 0):.1f}ms avg")
        print(f"  Fast Agent:     {fast_metrics.get('avg_response_time_ms', 0):.1f}ms avg")
        
        # Calculate improvement
        if enhanced_metrics.get('avg_response_time_ms', 0) > 0:
            improvement = ((enhanced_metrics['avg_response_time_ms'] - fast_metrics.get('avg_response_time_ms', 0)) 
                          / enhanced_metrics['avg_response_time_ms']) * 100
            print(f"  🚀 Speed Improvement: {improvement:.1f}%")
        
        print(f"\n📈 ACCURACY METRICS:")
        print(f"  Enhanced Agent: {enhanced_metrics.get('avg_confidence', 0):.3f} confidence")
        print(f"  Fast Agent:     {fast_metrics.get('avg_confidence', 0):.3f} confidence")
        
        print(f"\n🔤 TOKEN USAGE:")
        print(f"  Enhanced Agent: {enhanced_metrics.get('avg_tokens', 0):.0f} tokens avg")
        print(f"  Fast Agent:     {fast_metrics.get('avg_tokens', 0):.0f} tokens avg")
        
        print(f"\n✅ SUCCESS RATES:")
        print(f"  Enhanced Agent: {enhanced_metrics.get('success_rate', 0):.1%}")
        print(f"  Fast Agent:     {fast_metrics.get('success_rate', 0):.1%}")
        
        print(f"\n⏱️  RESPONSE TIME DISTRIBUTION:")
        for agent_name, metrics in [("Enhanced", enhanced_metrics), ("Fast", fast_metrics)]:
            if 'response_times' in metrics:
                times = metrics['response_times']
                print(f"  {agent_name}:")
                print(f"    Min: {min(times):.1f}ms")
                print(f"    Max: {max(times):.1f}ms")
                print(f"    Median: {statistics.median(times):.1f}ms")
                print(f"    Std Dev: {statistics.stdev(times) if len(times) > 1 else 0:.1f}ms")
        
        # Performance recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        fast_avg = fast_metrics.get('avg_response_time_ms', float('inf'))
        enhanced_avg = enhanced_metrics.get('avg_response_time_ms', float('inf'))
        fast_conf = fast_metrics.get('avg_confidence', 0)
        enhanced_conf = enhanced_metrics.get('avg_confidence', 0)
        
        if fast_avg < enhanced_avg * 0.5:  # Fast is significantly faster
            print("  🚀 Use Fast Agent for real-time applications")
        
        if abs(fast_conf - enhanced_conf) < 0.05:  # Similar confidence
            print("  ⚖️  Both agents provide similar accuracy")
        elif enhanced_conf > fast_conf + 0.1:
            print("  🎯 Use Enhanced Agent for maximum accuracy")
        
        if fast_avg < 5000:  # Under 5 seconds
            print("  ✅ Fast Agent meets interactive response requirements")
        
    async def run_full_benchmark(self):
        """Run complete benchmark comparing both agents."""
        print("🧪 Starting RAG Agent Performance Benchmark")
        print("=" * 60)
        
        # Test Enhanced Agent
        enhanced_metrics = await self.benchmark_agent(
            self.enhanced_agent, 
            "EnhancedRAGAgent", 
            self.test_questions
        )
        
        # Clear any caches between tests
        if hasattr(self.fast_agent, 'clear_cache'):
            self.fast_agent.clear_cache()
        
        # Test Fast Agent
        fast_metrics = await self.benchmark_agent(
            self.fast_agent, 
            "FastRAGAgent", 
            self.test_questions
        )
        
        # Print comparison
        self.print_comparison(enhanced_metrics, fast_metrics)
        
        return enhanced_metrics, fast_metrics

async def main():
    """Main function to run the performance test."""
    tester = PerformanceTester()
    
    try:
        enhanced_metrics, fast_metrics = await tester.run_full_benchmark()
        
        print(f"\n🎉 Benchmark Complete!")
        print(f"Enhanced Agent average: {enhanced_metrics.get('avg_response_time_ms', 0):.1f}ms")
        print(f"Fast Agent average: {fast_metrics.get('avg_response_time_ms', 0):.1f}ms")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 