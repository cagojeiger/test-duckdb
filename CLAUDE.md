# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DuckDB Vector Search benchmarking project focused on systematically analyzing the performance characteristics of DuckDB's VSS (Vector Similarity Search) extension for text vector search scenarios. The project aims to provide comprehensive performance benchmarks and practical recommendations for real-world usage.

## Architecture

The project follows a structured approach to benchmark DuckDB VSS extension across 48 experimental combinations:
- **Data Scales**: 10K, 100K, 250K vectors
- **Vector Dimensions**: 128, 256, 512, 1024
- **Search Types**: Pure vector search, Hybrid (vector + BM25)
- **Filter Conditions**: With/without metadata filtering

Key technical components:
- **Database**: DuckDB with VSS extension (HNSW indexing)
- **Text Data**: Korean text generated using Faker library
- **Performance Metrics**: Throughput (QPS), response time, accuracy (Recall@K), resource usage

## Common Commands

### Environment Setup
```bash
# Install Python dependencies (when requirements.txt is created)
pip install -r requirements.txt

# Install DuckDB VSS extension
python -c "import duckdb; conn = duckdb.connect(); conn.execute('INSTALL vss'); conn.execute('LOAD vss')"
```

### Running Experiments
```bash
# Run specific experiment (when main.py is implemented)
python main.py --experiment text_similarity --scale small --dimensions 128

# Run all benchmarks
python benchmarks/run_all.py

# Run tests
python -m pytest tests/
```

### DuckDB VSS Operations
```sql
-- Create HNSW index
CREATE INDEX idx_name ON table_name USING HNSW(vector_column) 
WITH (ef_construction = 128, ef_search = 64, M = 16, metric = 'cosine');

-- Vector similarity search
SELECT * FROM table_name 
ORDER BY array_distance(vector_column, query_vector::FLOAT[n])
LIMIT k;
```

## Key Implementation Considerations

### DuckDB VSS Limitations
- VSS extension is experimental, not production-ready
- Requires vectors as FLOAT[n] arrays (32-bit floats only)
- HNSW indexes must fit entirely in RAM
- Updates mark deletions only; periodic compression needed

### Performance Optimization
- Test different HNSW parameters (ef_construction, ef_search, M)
- Monitor memory usage carefully as indexes are RAM-resident
- Use connection pooling for concurrent operations
- Profile queries with EXPLAIN ANALYZE

### Korean Text Data Generation
- Use Faker with Korean locale (`ko_KR`)
- Generate realistic text patterns: news articles, product reviews, user profiles
- Include metadata for filtering tests (category, timestamp, user_id)