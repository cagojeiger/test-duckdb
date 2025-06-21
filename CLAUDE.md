# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DuckDB Vector Search benchmarking project focused on systematically analyzing the performance characteristics of DuckDB's VSS (Vector Similarity Search) extension for text vector search scenarios. The project follows a **functional programming paradigm** with immutable data structures, pure functions, and explicit effect handling.

## Architecture

### Functional Programming Design
The project uses a layered functional architecture (see plan/ directory for detailed design):
- **Pure Layer**: Business logic with no side effects (data generation, transformations, calculations)
- **Effect Layer**: IO operations wrapped in monadic types (database, file I/O, metrics collection)
- **Pipeline Layer**: Function composition to build complex workflows from simple functions

### Experiment Matrix
48 experimental combinations testing:
- **Data Scales**: 10K, 100K, 250K vectors
- **Vector Dimensions**: 128, 256, 512, 1024
- **Search Types**: Pure vector search, Hybrid (vector + BM25)
- **Filter Conditions**: With/without metadata filtering

### Key Components
- **Database**: DuckDB with VSS extension (HNSW indexing)
- **Text Data**: Korean text generated using Faker library
- **Type System**: Immutable dataclasses with frozen=True
- **Effect Management**: IO monad for side effects, Either for error handling
- **Performance Metrics**: Throughput (QPS), response time, accuracy (Recall@K), resource usage

## Common Commands

### Environment Setup
```bash
# Install Python dependencies
pip install duckdb faker pandas numpy psutil matplotlib seaborn plotly pyrsistent

# Install DuckDB VSS extension
python -c "import duckdb; conn = duckdb.connect(); conn.execute('INSTALL vss'); conn.execute('LOAD vss')"

# Verify VSS installation
python -c "import duckdb; conn = duckdb.connect(); print(conn.execute('SELECT * FROM duckdb_extensions() WHERE extension_name = \'vss\'').fetchall())"
```

### Running Experiments
```bash
# Run single experiment (when implemented)
python -m src.main --config experiments/small_128d_pure.json

# Run all 48 experiment combinations
python -m src.runners.experiment_runner --all

# Run specific experiment matrix
python -m src.runners.experiment_runner --data-scale small --dimensions 128,256

# Resume from checkpoint
python -m src.runners.experiment_runner --resume --checkpoint-dir checkpoints/

# Run tests
python -m pytest tests/ -v
python -m pytest tests/pure/ -v  # Test only pure functions
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

### Functional Programming Patterns
- **Immutability**: Use `@dataclass(frozen=True)` for all data structures
- **Pure Functions**: Separate business logic from I/O operations
- **Effect Handling**: Wrap all side effects in IO monad
- **Error Handling**: Use Either type instead of exceptions in pure code
- **Function Composition**: Build pipelines using `compose` and `pipe`

### Type System Usage
```python
# Example type definitions
Vector = NewType('Vector', List[float])
DocumentId = NewType('DocumentId', str)
ExperimentConfig = @dataclass(frozen=True)
IO[T] = Effect wrapper for side effects
Either[E, T] = Error handling without exceptions
```

### DuckDB VSS Limitations
- VSS extension is experimental, not production-ready
- Requires vectors as FLOAT[n] arrays (32-bit floats only)
- HNSW indexes must fit entirely in RAM
- Updates mark deletions only; periodic compression needed

### Performance Optimization
- Test different HNSW parameters (ef_construction, ef_search, M)
- Monitor memory usage carefully as indexes are RAM-resident
- Use parallel_map for concurrent pure operations
- Use connection pooling wrapped in Reader monad
- Profile queries with EXPLAIN ANALYZE

### Korean Text Data Generation
- Use Faker with Korean locale (`ko_KR`)
- Generate with deterministic seeds for reproducibility
- Text categories: news articles, product reviews, documents, social media
- Include metadata for filtering tests (category, timestamp, user_id)

## Project Structure

```
src/
├── types/              # Type definitions (frozen dataclasses)
├── pure/               # Pure functions (no side effects)
│   ├── generators/     # Data generation
│   ├── transformers/   # Data transformations
│   └── calculators/    # Metrics and analysis
├── effects/            # Side effect management
│   ├── db/            # Database IO operations
│   ├── io/            # File IO operations
│   └── metrics/       # Performance monitoring
├── pipelines/          # Function composition pipelines
└── runners/            # Main entry points

plan/                   # Functional design documentation
├── 01-functional-architecture.md
├── 02-type-definitions.md
├── 03-pure-functions.md
├── 04-effect-management.md
├── 05-pipeline-composition.md
├── 06-experiment-workflow.md
└── 07-implementation-guide.md
```