# Enhanced RAG AI Agent with Pydantic AI and Supabase

A sophisticated Retrieval-Augmented Generation (RAG) AI agent with advanced search capabilities, semantic re-ranking, and comprehensive observability.

## 🚀 Features

### Core RAG Capabilities

- **Document Ingestion**: TXT, PDF, and DOCX support with smart chunking
- **Vector Search**: Semantic similarity search with pgvector
- **AI Generation**: OpenAI-powered responses with retrieved context
- **Web Interface**: Advanced Streamlit UI with tabbed interface

### 🧠 Advanced Search & Retrieval

- **Semantic Re-ranking**: Cosine similarity and cross-encoder re-ranking
- **Multi-Query Expansion**: Automatic query expansion for better coverage
- **Adaptive Search**: Intelligent search strategy selection
- **Enhanced Vector Search**: Multiple search strategies with fallback

### 📊 Observability & Analytics

- **Performance Metrics**: Real-time response time and token usage tracking
- **Confidence Scoring**: AI confidence assessment for each response
- **Search Analytics**: Detailed search performance and relevance scores
- **Debug Tools**: Comprehensive debugging interface with metadata

### 🎨 User Experience

- **Tabbed Interface**: Chat, Documents, Analytics, and Debug tabs
- **Search Strategy Selection**: Choose between Adaptive, Multi-Query + Re-ranking, etc.
- **Response Metadata**: Expandable sections showing sources, timing, and confidence
- **Performance Dashboard**: Charts and trends for system performance

## Quick Start

### Prerequisites

- Python 3.11+
- Supabase account with pgvector enabled
- OpenAI API key

### Installation & Setup

1. **Clone and Setup Environment**:

   ```bash
   git clone <repository-url>
   cd supabase_mcp_server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # OPENAI_API_KEY=your_openai_key
   # SUPABASE_URL=your_supabase_url
   # SUPABASE_KEY=your_supabase_key
   ```

3. **Verify Components** (Optional):

   ```bash
   python verify_components.py
   ```

4. **Run the Enhanced Application**:
   ```bash
   streamlit run app.py
   ```

## 🔧 Usage

### Document Management

1. **Upload Documents**: Use the Documents tab to upload TXT, PDF, or DOCX files
2. **View Processing**: Monitor document processing with progress indicators
3. **Manage Knowledge Base**: View uploaded documents and their status

### Interactive Chat

1. **Configure Search**: Select search strategy in the sidebar (Adaptive, Multi-Query + Re-ranking, etc.)
2. **Ask Questions**: Chat interface with your documents
3. **View Details**: Expand response metadata to see sources, confidence, and timing

### Analytics & Performance

1. **Performance Metrics**: View real-time KPIs (response time, token usage, confidence)
2. **Trend Analysis**: Charts showing performance trends over time
3. **Search Analysis**: Detailed breakdown of search strategies and effectiveness

### Debugging & System Info

1. **Component Status**: View status of all enhanced RAG components
2. **Session Management**: Clear chat history or reset sessions
3. **Raw Metrics**: Access detailed system metrics and configuration

## 🏗️ Architecture

```
├── app.py                          # Enhanced main application entry point
├── enhanced_streamlit_ui.py        # Advanced Streamlit interface
├── verify_components.py            # Component verification script
├── src/
│   ├── agent/
│   │   ├── enhanced_rag_agent.py   # Enhanced agent with observability
│   │   └── rag_agent.py            # Basic agent implementation
│   ├── retrieval/
│   │   ├── enhanced_vector_search.py # Multi-strategy search
│   │   ├── reranker.py             # Semantic re-ranking
│   │   ├── query_expansion.py      # Multi-query expansion
│   │   └── vector_search.py        # Basic vector search
│   ├── utils/
│   │   ├── observability.py        # Metrics & performance tracking
│   │   ├── response_formatter.py   # Response formatting & citations
│   │   └── ...                     # Other utilities
│   ├── ingestion/                  # Document processing pipeline
│   └── database/                   # Database connections & setup
```

## 🎯 Search Strategies

### Adaptive Search

Intelligently escalates search techniques based on result quality:

1. Start with basic vector search
2. Apply re-ranking if needed
3. Use multi-query expansion as fallback

### Multi-Query + Re-ranking

- Generates multiple related queries
- Combines results from all queries
- Re-ranks using semantic similarity

### Re-ranking Only

- Performs standard vector search
- Applies semantic re-ranking to improve relevance

### Basic Search

- Standard vector similarity search
- Fastest option with good baseline performance

## 📈 Performance Features

- **Token Tracking**: Request, response, and total token usage
- **Response Time**: End-to-end and component-level timing
- **Confidence Scoring**: AI confidence in each response
- **Search Quality**: Relevance scores and search effectiveness
- **Real-time Metrics**: Live performance indicators in UI

## 🛠️ Configuration

Environment variables in `.env`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## 🧪 Testing

Run component verification:

```bash
python verify_components.py
```

Run tests:

```bash
pytest tests/
```

## License

MIT
# rag_demo
