# Confluence Search with Advanced RAG and Query Expansion

A powerful Confluence search tool for Open WebUI with intelligent query expansion, semantic search, reranking, and advanced relevance filtering.

## 🚀 Key Features

### **🔍 Intelligent Query Expansion (NEW in v0.7.0)**
- **LLM-Powered Search Enhancement**: Automatically generates alternative search queries using AI
- **Smart Variations**: Creates synonyms, abbreviations, related terms, and rephrased queries
- **Technical Documentation Focus**: Optimized prompts for finding business and technical content
- **Configurable**: Control number of variations (1-10) and customize the expansion model

### **🧠 Advanced Semantic Search**
- **Remote Embeddings**: OpenAI-compatible API for embeddings (supports local models)
- **Hybrid Search**: Combines semantic similarity with keyword matching (BM25)
- **Smart Chunking**: Breaks large documents into optimal-sized sections with overlap
- **Relevance Filtering**: Configurable similarity thresholds to filter irrelevant results

### **📊 Intelligent Reranking**
- **Cross-Encoder Models**: Advanced reranking using models like Qwen2.5-Reranker or BGE-Reranker
- **Relevance Scoring**: Display confidence scores for each result
- **Quality Filtering**: Filter out low-relevance chunks automatically
- **Multiple Model Support**: Compatible with various reranking APIs

### **⚡ Performance & Reliability**
- **Memory Optimization**: Batch processing and smart memory management
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Space Control**: Include/exclude specific Confluence spaces
- **SSL & Authentication**: Support for API keys and Personal Access Tokens

## 🎯 How Query Expansion Works

When you search for **"API authentication"**, the system:

1. **Original Search**: Searches Confluence with "API authentication"
2. **AI Enhancement**: Generates variations like:
   - "API auth"
   - "REST API security" 
   - "authentication endpoints"
   - "API token configuration"
3. **Multi-Query Search**: Searches with each variation
4. **Smart Aggregation**: Combines unique results from all searches
5. **Relevance Ranking**: Returns the most relevant content

This dramatically improves search coverage and finds content you might otherwise miss!

## 📋 Requirements

```
markdownify
openai
tiktoken
numpy
rank_bm25
scikit-learn
requests
```

## ⚙️ Configuration

### **Global Settings (Valves)**

#### **Basic Confluence Settings**
- `base_url`: Your Confluence instance URL
- `username`: Default username/email
- `api_key`: Default API key or Personal Access Token
- `ssl_verify`: Enable/disable SSL verification
- `api_result_limit`: Max pages to retrieve from Confluence API

#### **Embedding & AI Settings**
- `openai_api_key`: API key for embeddings (leave empty for local servers)
- `openai_api_base`: Embedding server URL (e.g., `http://localhost:8000/v1`)
- `embedding_model_name`: Model name (e.g., `text-embedding-ada-002`, `your-embedding-model`)

#### **🆕 Query Expansion Settings**
- `enable_query_expansion`: Enable AI-powered query expansion
- `query_expansion_api_key`: API key for expansion LLM (uses OpenAI key if empty)
- `query_expansion_api_base`: LLM server URL (uses OpenAI base if empty)
- `query_expansion_model`: Model for expansion (e.g., `gpt-3.5-turbo`, `your-llm-model`)
- `query_expansion_max_variations`: Number of alternative queries (1-10)
- `query_expansion_system_prompt`: Custom prompt template for query generation

#### **Reranking Settings**
- `enable_reranking`: Enable cross-encoder reranking
- `reranker_api_key`: Reranker API key (uses embedding key if empty)
- `reranker_api_base`: Reranker server URL (uses embedding base if empty)
- `reranker_model_name`: Reranker model (e.g., `your-reranker-model`)
- `minimum_relevance_score`: Filter threshold for relevance (0-1)

#### **RAG Processing Settings**
- `chunk_size`: Maximum chunk size for documents (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 100)
- `max_results`: Number of relevant chunks to return (default: 3)
- `similarity_threshold`: Minimum similarity score for chunks
- `ensemble_weighting`: Balance between semantic (1.0) and keyword (0.0) search
- `enable_hybrid_search`: Enable combined semantic + keyword search
- `full_context`: Return complete pages instead of relevant chunks

### **User Settings (UserValves)**

- `api_key_auth`: Use API key vs Personal Access Token
- `username`: User-specific username (overrides global)
- `api_key`: User-specific API key (overrides global)
- `split_terms`: Split search queries into individual words
- `included_confluence_spaces`: **REQUIRED** - Comma-separated spaces to search
- `excluded_confluence_spaces`: Spaces to exclude from search

## 🔧 Setup Instructions

### **1. Basic Setup**
1. Install the tool in Open WebUI
2. Configure your Confluence URL and credentials in Global Settings
3. **Important**: Set `included_confluence_spaces` in User Settings (required)

### **2. Embedding Server Setup**

#### **Option A: OpenAI**
```python
openai_api_key = "sk-your-openai-api-key"
openai_api_base = "https://api.openai.com/v1"
embedding_model_name = "text-embedding-ada-002"
```

#### **Option B: Local Embedding Server**
```python
openai_api_key = ""  # Leave empty
openai_api_base = "http://your-server:8000/v1"
embedding_model_name = "your-embedding-model"
```

### **3. Query Expansion Setup (Optional)**
```python
enable_query_expansion = True
query_expansion_model = "your-llm-model"
query_expansion_max_variations = 3
```

### **4. Reranking Setup (Optional)**
```python
enable_reranking = True
reranker_model_name = "your-reranker-model"
minimum_relevance_score = 0.7
```

## 📖 Usage Examples

### **Basic Search**
```python
# Search in page content
await search_confluence("API authentication", "content")

# Search in titles only  
await search_confluence("deployment guide", "title")

# Search both title and content
await search_confluence("database setup", "title_and_content")
```

### **Advanced Search Flow**
1. **Query Expansion**: "API auth" → ["API authentication", "REST API security", "authentication endpoints"]
2. **Multi-Search**: Searches Confluence with all variations
3. **Content Retrieval**: Fetches full content of matching pages
4. **Smart Chunking**: Breaks content into relevant sections
5. **Semantic Analysis**: Finds content that matches query meaning
6. **Reranking**: Ranks results by relevance confidence
7. **Filtering**: Removes low-relevance content
8. **Results**: Returns top relevant sections with scores

## 🔄 Version History

- **0.7.0** - Added LLM-powered query expansion for improved search coverage
- **0.6.1** - Added minimum relevance score filtering and score display in citations
- **0.6.0** - Added reranking support with cross-encoder models
- **0.5.0** - Replaced local sentence transformers with remote OpenAI API embeddings

## 🏗️ Architecture

```
Query Input
    ↓
Query Expansion (LLM) → Multiple Search Queries
    ↓
Confluence API Search → Page IDs
    ↓  
Content Retrieval → Full Page Content
    ↓
Document Chunking → Manageable Sections
    ↓
Embedding Generation → Vector Representations
    ↓
Hybrid Search (Semantic + Keyword) → Candidate Results
    ↓
Cross-Encoder Reranking → Relevance Scores
    ↓
Relevance Filtering → High-Quality Results
    ↓
Citation Generation → Final Output
```

## 🚨 Important Notes

- **Space Configuration**: You MUST set `included_confluence_spaces` in User Settings
- **Memory Management**: Large documents are automatically chunked and processed in batches
- **API Compatibility**: Works with OpenAI API and OpenAI-compatible local servers
- **Authentication**: Supports both API keys and Personal Access Tokens
- **Error Handling**: Graceful fallbacks when individual components fail

## 🛠️ Troubleshooting

### **No Search Results**
- Check that `included_confluence_spaces` is set in User Settings
- Verify Confluence credentials and permissions
- Ensure the specified spaces exist and are accessible

### **Embedding Errors**
- Verify embedding server URL and API key
- Test connection to embedding endpoint
- Check model name compatibility

### **Query Expansion Issues**
- Verify LLM server configuration
- Check API key and model availability
- Monitor logs for expansion errors (falls back to original query)

### **Reranking Problems**
- Ensure reranker endpoint is available (`/rerank` or `/v1/rerank`)
- Verify model compatibility with cross-encoder format
- Check relevance score thresholds

## 🤝 Contributing

This tool is a community effort! Original work by [@romainneup](https://github.com/RomainNeup), enhanced with remote embeddings, reranking, and query expansion features.

- **Repository**: https://github.com/RomainNeup/open-webui-utilities
- **Funding**: https://github.com/sponsors/RomainNeup

## 📄 License

See the original repository for license information.
