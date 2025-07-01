# Confluence Search with Reranking Tool

A powerful Open WebUI tool that enables intelligent search and retrieval of content from Confluence with advanced reranking capabilities for improved relevance.

## Features

- **Smart Search**: Search by title, content, or both across your Confluence spaces
- **RAG Processing**: Retrieval Augmented Generation for finding the most relevant content chunks
- **Hybrid Search**: Combines semantic search (embeddings) with keyword search (BM25) for optimal results
- **Cross-Encoder Reranking**: Uses advanced reranking models to improve result relevance
- **Relevance Filtering**: Automatically filters out low-relevance content based on configurable thresholds
- **Flexible Authentication**: Supports both API key and Personal Access Token authentication
- **Memory Optimization**: Handles large Confluence pages efficiently with configurable chunking

## Installation

### Requirements

```
markdownify
openai
numpy
rank_bm25
scikit-learn
requests
```

### Setup

1. Install the tool in your Open WebUI instance
2. Configure the admin valves (see Configuration section)
3. Users can override settings with their own credentials

## Configuration

### Admin Configuration (Valves)

Administrators can set default values that apply to all users:

#### Confluence Settings
- **`base_url`**: Your Confluence instance URL (e.g., `https://company.atlassian.net/wiki`)
- **`ssl_verify`**: Enable/disable SSL verification (default: `true`)
- **`username`**: Default username for API authentication
- **`api_key`**: Default API key or Personal Access Token
- **`api_result_limit`**: Maximum pages to retrieve from Confluence API (default: `5`)

#### Embedding Settings
- **`openai_api_key`**: API key for OpenAI embeddings (leave empty for local servers)
- **`openai_api_base`**: Embedding server URL (default: `https://api.openai.com/v1`)
  - For local servers: `http://your-server:8000/v1`
- **`embedding_model_name`**: Model to use for embeddings (e.g., `text-embedding-ada-002`, `bge-m3`)

#### RAG Processing Settings
- **`chunk_size`**: Maximum chunk size for splitting pages (default: `1000`)
- **`chunk_overlap`**: Overlap between chunks for context preservation (default: `100`)
- **`max_results`**: Maximum relevant chunks to return (default: `3`)
- **`similarity_threshold`**: Minimum similarity score for semantic search (default: `0.0`)
- **`ensemble_weighting`**: Balance between semantic and keyword search (default: `0.5`)
  - `0.0` = Pure keyword search
  - `1.0` = Pure semantic search
- **`enable_hybrid_search`**: Enable/disable hybrid search (default: from environment)
- **`full_context`**: Return complete pages instead of chunks (default: `false`)

#### Reranking Settings
- **`enable_reranking`**: Enable cross-encoder reranking (default: `false`)
- **`reranker_api_key`**: API key for reranker (defaults to embedding API key)
- **`reranker_api_base`**: Reranker server URL (defaults to embedding server)
- **`reranker_model_name`**: Reranking model (e.g., `Qwen/Qwen2.5-Reranker-0.6B`)
- **`reranker_top_k`**: Number of results to keep after reranking (default: `5`)
- **`minimum_relevance_score`**: Minimum score (0-1) for chunks to be included (default: `0.7`)
  - Only applies when reranking is enabled
  - `0.7` = 70% relevance threshold

#### Memory Management
- **`max_page_size`**: Maximum characters per Confluence page (default: `10000`)
- **`batch_size`**: Documents to process at once for embeddings (default: `16`)

### User Configuration (UserValves)

Individual users can override admin settings with their own values:

#### Authentication
- **`api_key_auth`**: Use API key authentication (default: `true`)
  - Set to `false` to use Personal Access Token
- **`username`**: Your Confluence username/email
  - Leave empty for Personal Access Token auth
- **`api_key`**: Your API key or Personal Access Token

#### Search Settings
- **`split_terms`**: Split search queries into words (default: `true`)
  - Improves search results for multi-word queries
- **`included_confluence_spaces`**: Comma-separated list of spaces to search
  - **REQUIRED**: No spaces will be searched if empty
  - Example: `TECH,DOCS,KB`
- **`excluded_confluence_spaces`**: Spaces to exclude from search
  - Only applies to included spaces
  - Example: `ARCHIVE,OLD`

## Usage

### Basic Search

To search Confluence, use natural language queries like:
- "Search Confluence for project documentation"
- "Find pages about API integration in Confluence"
- "Look for deployment guides"

### Search Types

The tool will automatically detect the appropriate search type:
- **Title search**: When looking for specific page titles
- **Content search**: When searching within page content
- **Combined search**: Default behavior for general queries

### Understanding Results

#### With Reranking Enabled
Each result shows a relevance score:
```
📊 Relevance Score: 92.3%
---
[Content follows...]
```

#### Relevance Scores
- **90-100%**: Highly relevant - exact match to your query
- **70-90%**: Very relevant - strong connection to your query
- **50-70%**: Moderately relevant - some useful information
- **Below 50%**: Less relevant - filtered out by default

## Best Practices

### For Administrators

1. **Embedding Server Setup**
   - Use a local embedding server for better performance and privacy
   - Recommended models: `bge-m3`, `e5-large-v2`

2. **Reranking Configuration**
   - Enable reranking for better result quality
   - Start with `minimum_relevance_score: 0.7` and adjust based on results
   - Lower the threshold if getting too few results
   - Raise it if getting irrelevant content

3. **Memory Management**
   - Increase `chunk_size` for technical documentation (better context)
   - Decrease for general content (more precise matching)
   - Adjust `batch_size` based on your server capacity

### For Users

1. **Space Configuration**
   - Always set `included_confluence_spaces` to limit search scope
   - Use space keys, not space names

2. **Search Queries**
   - Be specific but not too narrow
   - Use keywords that likely appear in your documentation
   - Try different phrasings if first search doesn't yield results

3. **Authentication**
   - Personal Access Tokens are more secure than API keys
   - Create tokens with read-only permissions for safety

## Troubleshooting

### No Results Found
- Check if `included_confluence_spaces` is set correctly
- Verify space keys are correct (not space names)
- Try broader search terms
- Lower the `minimum_relevance_score` if reranking is enabled

### Authentication Errors
- Verify your Confluence URL includes `/wiki`
- Check API key or token validity
- Ensure user has read access to specified spaces

### Slow Performance
- Reduce `api_result_limit` to fetch fewer pages
- Decrease `chunk_size` for faster processing
- Consider using a local embedding server

### Reranking Issues
- Ensure reranker server is running and accessible
- Check if the model name is correct
- Verify the reranker endpoint (usually `/rerank` or `/v1/rerank`)

## Version History

- **0.6.1** - Added minimum relevance score filtering and score display
- **0.6.0** - Added cross-encoder reranking support
- **0.5.0** - Switched to remote embedding servers
- **0.4.0** - Added space inclusion/exclusion support

## Credits

- **Original Author**: [@romainneup](https://github.com/RomainNeup)
- **Current Branch**: Gil Gedje
- **Note**: This tool is a branch of the original work with added reranking and relevance scoring features
