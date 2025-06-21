# DuckDB Vector Search Benchmarking Project - AI Documentation

## Project Overview
This is a **Python-based benchmarking platform** for systematically analyzing the performance characteristics of **DuckDB's VSS (Vector Similarity Search) extension** with a focus on **text vector search scenarios**.

## Core Purpose
- **Performance Benchmarking**: Measure search performance across various data scales and vector dimensions
- **Optimization Research**: Determine optimal HNSW index parameters and configurations
- **Scalability Analysis**: Identify performance limits based on memory constraints and data size
- **Hybrid Search Evaluation**: Test combination of vector search + BM25 text search
- **Production Guidelines**: Provide practical recommendations for real-world usage

## Technical Stack
- **Database**: DuckDB + VSS extension (HNSW indexing)
- **Language**: Python 3.12+
- **Data Generation**: Faker library (Korean locale)
- **Performance Measurement**: psutil, time, DuckDB EXPLAIN ANALYZE
- **Visualization**: matplotlib, seaborn, plotly
- **Data Processing**: pandas, numpy

## Experimental Design Framework

### Test Matrix (48 Total Combinations)
- **Data Sizes**: 3 levels (10K, 100K, 250K vectors)
- **Vector Dimensions**: 4 levels (128, 256, 512, 1024)
- **Search Types**: 2 types (Pure Vector, Hybrid Vector+BM25)
- **Filter Conditions**: 2 types (No Filter, With Metadata Filter)

### Performance Measurement Areas (5 Categories)
1. **Bulk Data Upload Speed**: Vector insertion performance
2. **HNSW Index Construction Time**: Index building performance
3. **Vector Search Performance**: Query execution speed
4. **Hybrid Search Performance**: Combined vector+text search
5. **Filtered Search Performance**: Conditional search with metadata filters

### Key Metrics Collected
- **Throughput**: vectors/sec, QPS (queries per second)
- **Response Time**: Query execution time (ms)
- **Accuracy**: Recall@K (K=1,5,10)
- **Resource Usage**: CPU, memory utilization
- **Index Efficiency**: Build time, size, compression ratio

## DuckDB VSS Technical Specifications

### Core Capabilities
- **Data Type**: FLOAT 32-bit vectors only (`FLOAT[n]` arrays)
- **Indexing**: HNSW (Hierarchical Navigable Small Worlds) algorithm
- **Distance Metrics**: L2 (Euclidean), Cosine, Inner Product
- **Memory Constraint**: Indexes must be fully loaded in RAM
- **Vector Dimensions**: No explicit maximum, limited by available memory

### HNSW Index Parameters
```sql
CREATE INDEX idx_name ON table_name USING HNSW(vector_column) 
WITH (
    ef_construction = 128,  -- Candidate vertices during index construction
    ef_search = 64,         -- Candidate vertices during search
    M = 16,                 -- Maximum connections per vertex
    M0 = 32,                -- Base connectivity for level 0
    metric = 'cosine'       -- Distance metric (l2sq, cosine, ip)
);
```

### Key Limitations
- **Experimental Status**: Not production-ready, stability concerns
- **Memory Dependency**: System RAM size limits dataset scale
- **Update Performance**: Deletions are marked only, require periodic compression
- **Data Persistence**: Risk of data loss due to experimental nature

## Data Generation Strategy

### Korean Text Data (Faker-based)
- **News Articles**: `fake.text()`, `fake.sentence()`
- **Product Reviews**: `fake.catch_phrase()`, `fake.company()`
- **User Profiles**: `fake.profile()`, `fake.name()`
- **Categories**: News, Reviews, Documents, Social Media
- **Locale**: Korean (`ko_KR`) for realistic text patterns

### Vector Embedding Simulation
- **Random Normalized Vectors**: Simulating text embeddings
- **Dimension-specific Patterns**: Different characteristics per dimension level
- **Metadata Association**: Category, timestamp, user_id for filtering tests

## Expected Research Outcomes

### Performance Hypotheses
1. **Curse of Dimensionality**: Diminishing returns after 512 dimensions
2. **Memory Wall**: Sharp performance degradation beyond 100K vectors
3. **Hybrid Trade-off**: 20% quality improvement, 30% performance cost
4. **Filter Optimization**: Effective only with <10% selectivity

### Deliverables
- **Performance Benchmark Report**: Detailed analysis and recommendations
- **Optimization Guide**: HNSW parameter tuning guidelines
- **Visualization Dashboard**: Interactive performance analysis tools
- **Production Guide**: Real-world deployment recommendations

## File Structure Context

### Documentation
- `docs/01-draft-duckdb-vector-search-investigation.md`: Technical investigation of DuckDB VSS capabilities
- `docs/02-experimental-design.md`: Detailed 48-combination experimental framework
- `README.md`: Project overview and goals (no implementation structure)
- `devin.md`: This AI-friendly project summary

### Implementation Areas (To Be Developed)
- **Data Generation**: Faker-based Korean text and vector generation
- **Benchmarking Engine**: Performance measurement and analysis
- **Scenario Testing**: Text similarity search implementations
- **Utilities**: DuckDB connection management and vector operations
- **Testing**: Unit tests for all components

## AI Assistant Guidelines

### When Working on This Project
1. **Focus on Text Search**: This is specifically for text vector search benchmarking
2. **Korean Data**: Use Korean locale for realistic text generation
3. **Performance First**: All implementations should be optimized for benchmarking
4. **Memory Awareness**: Consider DuckDB VSS memory constraints in all designs
5. **Experimental Nature**: Account for DuckDB VSS experimental status and limitations

### Key Implementation Priorities
1. **Systematic Testing**: Follow the 48-combination experimental framework
2. **Accurate Measurement**: Implement precise performance metrics collection
3. **Scalable Design**: Handle data sizes from 10K to 250K vectors
4. **Visualization Ready**: Structure data for easy analysis and plotting
5. **Documentation**: Maintain clear documentation of all findings

### Technical Considerations
- **HNSW Parameters**: Test different configurations for optimization
- **Memory Management**: Monitor and optimize memory usage patterns
- **Query Patterns**: Implement both simple and complex search scenarios
- **Error Handling**: Account for experimental extension limitations
- **Performance Isolation**: Ensure clean measurement environments

This project aims to provide the most comprehensive analysis of DuckDB VSS extension performance characteristics for text search scenarios, with practical recommendations for real-world usage.
