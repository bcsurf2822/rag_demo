# Enhanced RAG AI Agent: How It Works

## Architecture Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) AI agent with advanced search capabilities, semantic re-ranking, and comprehensive observability. The system is built using Pydantic AI and Supabase with pgvector.

## Core Components

### 1. Document Ingestion Pipeline

- **Input**: Accepts TXT, PDF, and DOCX files
- **Processing**:
  - Extracts text using PyPDF2 (for PDFs) and simple text processing
  - Implements smart chunking to break documents into manageable pieces
  - Generates embeddings using OpenAI's embedding models
- **Storage**: Stores document chunks and their embeddings in Supabase with pgvector

### 2. RAG Agents

The system offers two agent implementations:

#### Enhanced RAG Agent

- Full-featured agent with comprehensive capabilities
- Multiple search strategies (adaptive, multi-query, reranking)
- Detailed metadata and observability
- Response time: 8-15 seconds

#### Fast RAG Agent

- Optimized for speed (60-80% faster)
- Intelligent caching with 5-minute TTL
- Parallel processing with asyncio
- Response time: 2-5 seconds
- Reduced token usage (45% less)

### 3. Advanced Search & Retrieval

- **Semantic Re-ranking**: Improves result relevance using cosine similarity and cross-encoder methods
- **Multi-Query Expansion**: Generates related queries for better coverage
- **Adaptive Search**: Intelligently selects search strategies based on query complexity
- **Enhanced Vector Search**: Multiple fallback strategies for robust retrieval

### 4. Observability & Analytics

- **Performance Metrics**: Tracks response times, token usage, and search quality
- **Confidence Scoring**: Assesses AI confidence for each response
- **Search Analytics**: Detailed metrics on search performance
- **Debug Tools**: Comprehensive debugging interface with metadata

### 5. User Interface

- **Streamlit UI**: Advanced interface with multiple tabs:
  - 💬 Chat: Interactive conversation with the AI agent
  - 📄 Documents: Upload and manage knowledge base documents
  - 📊 Analytics: Performance metrics and trends
  - 🔧 Debug: Detailed system information and debugging tools

## Data Flow

1. **Document Upload**:

   - User uploads documents through the UI
   - Documents are processed, chunked, and embedded
   - Chunks and embeddings are stored in Supabase

2. **Query Processing**:

   - User submits a question
   - Question is converted to an embedding
   - System performs vector search with optional:
     - Multi-query expansion
     - Semantic re-ranking
     - Adaptive strategy selection

3. **Response Generation**:

   - Retrieved contexts are formatted and sent to the LLM
   - LLM generates a response with citations
   - Response is formatted with markdown and source attribution
   - Performance metrics are collected and stored

4. **Result Presentation**:
   - Response is displayed with expandable metadata
   - Sources are highlighted with citations
   - Performance metrics are updated in real-time

## Performance Optimization

The system includes significant performance optimizations:

- **Parallel Processing**: Concurrent search operations
- **Intelligent Caching**: Query result caching with 5-minute TTL
- **Reduced Parameters**: Optimized defaults for speed
- **Vectorized Operations**: Batch similarity computations
- **Smart Strategy Selection**: Adaptive query routing

## Configuration Options

Users can configure:

- **Agent Type**: Enhanced (comprehensive) or Fast (speed-optimized)
- **Search Strategy**: Adaptive, Multi-Query + Re-ranking, Re-ranking Only, or Basic
- **Match Parameters**: Number of results and similarity thresholds
- **Performance Options**: Caching, timeout settings, and debug information

## Getting Started

1. Set up environment variables (OpenAI API key, Supabase credentials)
2. Install dependencies with `pip install -r requirements.txt`
3. Run `streamlit run app.py` to start the application
4. Upload documents to build your knowledge base
5. Start asking questions in the chat interface

## Technology Stack

- **Language**: Python 3.11+
- **AI Framework**: Pydantic AI for agent implementation
- **Database**: Supabase with pgvector extension
- **Embeddings**: OpenAI embeddings API
- **LLM Provider**: OpenAI (GPT-4o-mini or similar)
- **UI**: Streamlit
- **Document Processing**: PyPDF2 for PDF extraction
