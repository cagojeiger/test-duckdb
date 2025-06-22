"""Experiment pipeline composition for DuckDB VSS benchmarking."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from datetime import datetime
import time
from src.types.monads import IO
from src.types.core import (
    ExperimentConfig,
    ExperimentResult,
    DataScale,
    Dimension,
    SearchType,
    DatabaseConfig,
    TextContent,
    Vector,
    SearchResult,
    Metrics,
    AccuracyMetrics,
)
from src.effects.db.connection import create_connection, DBConnection
from src.effects.db.tables import (
    TableSchema,
    create_documents_table,
    create_hnsw_index,
    batch_insert_documents,
)
from src.effects.db.search import SearchQuery, vector_similarity_search, hybrid_search
from src.pure.generators.text import generate_korean_texts
from src.pure.generators.vectors import generate_random_vectors
from src.pure.calculators.metrics import calculate_accuracy_metrics


@dataclass(frozen=True)
class ExperimentData:
    """Generated data for a single experiment."""

    documents: List[tuple[str, TextContent, Vector]]
    query_vectors: List[Vector]
    ground_truth: dict[str, List[str]]  # query_id -> relevant_doc_ids
    config: ExperimentConfig


def generate_experiment_matrix() -> List[ExperimentConfig]:
    """Generate all 48 experiment configurations."""
    from src.types.core import FilterConfig, HNSWParams

    configs = []

    data_scales = [DataScale.SMALL, DataScale.MEDIUM, DataScale.LARGE]
    dimensions = [Dimension(128), Dimension(256), Dimension(512), Dimension(1024)]
    search_types = [SearchType.PURE_VECTOR, SearchType.HYBRID]
    filter_conditions = [True, False]  # With/without category filtering

    for data_scale in data_scales:
        for dimension in dimensions:
            for search_type in search_types:
                for has_filter in filter_conditions:
                    filter_config = FilterConfig(
                        enabled=has_filter,
                        category=None,  # Will be set during execution if needed
                    )

                    if data_scale == DataScale.SMALL:
                        hnsw_params = HNSWParams(ef_construction=64, ef_search=32, M=8)
                    elif data_scale == DataScale.MEDIUM:
                        hnsw_params = HNSWParams(
                            ef_construction=128, ef_search=64, M=16
                        )
                    else:  # LARGE
                        hnsw_params = HNSWParams(
                            ef_construction=256, ef_search=128, M=32
                        )

                    config = ExperimentConfig(
                        data_scale=data_scale,
                        dimension=dimension,
                        search_type=search_type,
                        filter_config=filter_config,
                        hnsw_params=hnsw_params,
                        batch_size=1000,
                        num_queries=100,
                    )
                    configs.append(config)

    return configs


def generate_experiment_data(config: ExperimentConfig) -> IO[ExperimentData]:
    """Generate data for a single experiment."""

    def create_data() -> ExperimentData:
        if config.data_scale == DataScale.SMALL:
            num_docs = 10_000
        elif config.data_scale == DataScale.MEDIUM:
            num_docs = 100_000
        else:  # LARGE
            num_docs = 250_000

        config_hash = hash(str(config)) % 2**32
        texts = generate_korean_texts(count=num_docs, seed=config_hash)

        vectors = generate_random_vectors(
            count=num_docs, dimension=config.dimension, seed=config_hash
        )

        documents = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            doc_id = f"doc_{config_hash}_{i:06d}"
            documents.append((doc_id, text, vector))

        # Use some document vectors as queries to create ground truth
        query_step = max(1, num_docs // config.num_queries)
        query_indices = list(range(0, num_docs, query_step))[:config.num_queries]
        
        query_vectors = []
        ground_truth = {}
        
        for i, doc_idx in enumerate(query_indices):
            query_id = f"query_{config_hash}_{i:03d}"
            query_vector = documents[doc_idx][2]  # Use document's vector
            query_vectors.append(query_vector)
            
            doc_id = documents[doc_idx][0]
            ground_truth[query_id] = [doc_id]

        return ExperimentData(
            documents=documents, 
            query_vectors=query_vectors, 
            ground_truth=ground_truth,
            config=config
        )

    class GenerateDataIO(IO[ExperimentData]):
        def run(self) -> ExperimentData:
            return create_data()

    return GenerateDataIO()


def setup_database_workflow(
    config: ExperimentConfig, data: ExperimentData
) -> IO[tuple[DBConnection, TableSchema]]:
    """Set up database connection and table for experiment."""

    def setup() -> tuple[DBConnection, TableSchema]:
        db_config = DatabaseConfig(
            database_path=":memory:",  # Use in-memory database for experiments
            memory_limit_mb=8192,  # 8GB memory limit
            threads=4,
        )

        conn_io = create_connection(db_config)
        conn = conn_io.run()

        config_hash = hash(str(config)) % 2**32
        table_name = f"documents_{config_hash}"
        schema = TableSchema(
            table_name=table_name, dimension=config.dimension, has_metadata=True
        )

        create_table_io = create_documents_table(conn, schema)
        create_table_io.run()

        return conn, schema

    class SetupDatabaseIO(IO[tuple[DBConnection, TableSchema]]):
        def run(self) -> tuple[DBConnection, TableSchema]:
            return setup()

    return SetupDatabaseIO()


def insert_data_workflow(
    conn: DBConnection, schema: TableSchema, data: ExperimentData
) -> IO[Metrics]:
    """Insert experiment data into database."""

    def insert_data() -> Metrics:
        start_time = time.time()

        batch_size = 1000
        documents = data.documents

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_insert_io = batch_insert_documents(conn, schema, batch)
            batch_insert_io.run()

        insert_time = time.time() - start_time

        return Metrics(
            query_time_ms=insert_time * 1000,
            throughput_qps=len(documents) / insert_time if insert_time > 0 else 0.0,
            memory_usage_mb=0.0,  # Will be measured separately
            index_size_mb=0.0,
        )

    class InsertDataIO(IO[Metrics]):
        def run(self) -> Metrics:
            return insert_data()

    return InsertDataIO()


def build_index_workflow(
    conn: DBConnection, schema: TableSchema, config: ExperimentConfig
) -> IO[Metrics]:
    """Build HNSW index for experiment."""

    def build_index() -> Metrics:
        start_time = time.time()

        if config.data_scale == DataScale.SMALL:
            ef_construction, ef_search, M = 64, 32, 8
        elif config.data_scale == DataScale.MEDIUM:
            ef_construction, ef_search, M = 128, 64, 16
        else:  # LARGE
            ef_construction, ef_search, M = 256, 128, 32

        create_index_io = create_hnsw_index(
            conn, schema, ef_construction, ef_search, M, "cosine"
        )
        create_index_io.run()

        index_time = time.time() - start_time

        return Metrics(
            query_time_ms=index_time * 1000,
            throughput_qps=0.0,
            memory_usage_mb=0.0,  # Will be measured separately
            index_size_mb=0.0,
        )

    class BuildIndexIO(IO[Metrics]):
        def run(self) -> Metrics:
            return build_index()

    return BuildIndexIO()


def execute_search_workflow(
    conn: DBConnection, schema: TableSchema, data: ExperimentData
) -> IO[List[SearchResult]]:
    """Execute search queries for experiment."""

    def execute_searches() -> List[SearchResult]:
        results = []
        config = data.config

        for i, query_vector in enumerate(data.query_vectors):
            config_hash = hash(str(config)) % 2**32
            query_id = f"query_{config_hash}_{i:03d}"
            
            search_query = SearchQuery(
                query_vector=query_vector,
                k=10,
                filter_category=config.filter_config.category
                if config.filter_config.enabled
                else None,
                include_distances=True,
            )

            if config.search_type == SearchType.PURE_VECTOR:
                search_io = vector_similarity_search(
                    conn, schema.table_name, search_query
                )
            else:  # HYBRID
                search_io = hybrid_search(
                    conn,
                    schema.table_name,
                    search_query,
                    text_query="테스트",
                    vector_weight=0.7,
                    text_weight=0.3,
                )

            result = search_io.run()
            result = SearchResult(
                query_id=query_id,
                retrieved_ids=result.retrieved_ids,
                distances=result.distances,
                metrics=result.metrics
            )
            results.append(result)

        return results

    class ExecuteSearchesIO(IO[List[SearchResult]]):
        def run(self) -> List[SearchResult]:
            return execute_searches()

    return ExecuteSearchesIO()


def single_experiment_pipeline(config: ExperimentConfig) -> IO[ExperimentResult]:
    """Complete pipeline for a single experiment."""

    def run_experiment() -> ExperimentResult:
        data_io = generate_experiment_data(config)
        data = data_io.run()

        setup_io = setup_database_workflow(config, data)
        conn, schema = setup_io.run()

        try:
            insert_io = insert_data_workflow(conn, schema, data)
            insertion_metrics = insert_io.run()

            index_io = build_index_workflow(conn, schema, config)
            index_metrics = index_io.run()

            search_io = execute_search_workflow(conn, schema, data)
            search_results = search_io.run()

            # Calculate actual accuracy metrics using ground truth
            accuracy_metrics = calculate_accuracy_metrics(search_results, data.ground_truth)

            return ExperimentResult(
                config=config,
                insert_metrics=insertion_metrics,
                index_metrics=index_metrics,
                search_results=search_results,
                accuracy=accuracy_metrics,
                timestamp=datetime.now(),
            )

        finally:
            conn.close()

    class SingleExperimentIO(IO[ExperimentResult]):
        def run(self) -> ExperimentResult:
            return run_experiment()

    return SingleExperimentIO()


def batch_experiment_pipeline(
    configs: List[ExperimentConfig], batch_size: int = 4
) -> IO[List[ExperimentResult]]:
    """Execute multiple experiments in batches."""

    def run_batch() -> List[ExperimentResult]:
        all_results = []

        for i in range(0, len(configs), batch_size):
            batch_configs = configs[i : i + batch_size]

            batch_results = []
            for config in batch_configs:
                experiment_io = single_experiment_pipeline(config)
                result = experiment_io.run()
                batch_results.append(result)

                config_hash = hash(str(config)) % 2**32
                print(f"Completed experiment {config_hash}")

            all_results.extend(batch_results)

            import gc

            gc.collect()

        return all_results

    class BatchExperimentIO(IO[List[ExperimentResult]]):
        def run(self) -> List[ExperimentResult]:
            return run_batch()

    return BatchExperimentIO()
