#!/usr/bin/env python3
"""
Integration test for DuckDB VSS pipeline
Tests the complete end-to-end workflow before merging
"""

import sys

sys.path.insert(0, "src")

from src.types.core import (
    DataScale,
    Dimension,
    SearchType,
    FilterConfig,
    HNSWParams,
    ExperimentConfig,
)
from src.pipelines.experiments import (
    generate_experiment_matrix,
    generate_experiment_data,
)


def test_experiment_matrix_generation() -> bool:
    """Test experiment matrix generation"""
    print("\n📊 Testing experiment matrix generation...")

    try:
        configs = generate_experiment_matrix()

        if len(configs) == 48:
            print(f"✅ Generated {len(configs)} experiment configurations")

            data_scales = set(config.data_scale for config in configs)
            dimensions = set(config.dimension for config in configs)
            search_types = set(config.search_type for config in configs)

            print(f"   Data scales: {len(data_scales)} ({list(data_scales)})")
            print(f"   Dimensions: {len(dimensions)} ({[int(d) for d in dimensions]})")
            print(f"   Search types: {len(search_types)} ({list(search_types)})")

            return True
        else:
            print(f"❌ Expected 48 configurations, got {len(configs)}")
            return False

    except Exception as e:
        print(f"❌ Failed to generate experiment matrix: {e}")
        return False


def test_data_generation() -> bool:
    """Test data generation for small experiment"""
    print("\n📝 Testing data generation...")

    try:
        config = ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

        data_io = generate_experiment_data(config)
        data = data_io.run()

        if len(data.documents) == 10_000:  # SMALL scale = 10K documents
            print(f"✅ Generated {len(data.documents)} documents")
        else:
            print(f"❌ Expected 10,000 documents, got {len(data.documents)}")
            return False

        if len(data.query_vectors) == 10:
            print(f"✅ Generated {len(data.query_vectors)} query vectors")
        else:
            print(f"❌ Expected 10 query vectors, got {len(data.query_vectors)}")
            return False

        doc_id, text_content, vector = data.documents[0]

        if vector.dimension == Dimension(128):
            print(f"✅ Vector dimension correct: {vector.dimension}")
        else:
            print(f"❌ Expected dimension 128, got {vector.dimension}")
            return False

        if len(vector.data) == 128:
            print(f"✅ Vector data length correct: {len(vector.data)}")
        else:
            print(f"❌ Expected vector length 128, got {len(vector.data)}")
            return False

        if text_content.title and text_content.text:
            print("✅ Text content generated successfully")
        else:
            print("❌ Text content missing title or content")
            return False

        return True

    except Exception as e:
        print(f"❌ Failed to generate data: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_integration_tests() -> bool:
    """Run all integration tests"""
    print("🧪 Starting DuckDB VSS Integration Tests")
    print("=" * 50)

    tests = [
        ("Experiment Matrix Generation", test_experiment_matrix_generation),
        ("Data Generation", test_data_generation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - Ready for merge!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Fix issues before merge")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
