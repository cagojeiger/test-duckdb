#!/usr/bin/env python3
"""
Test script to validate the recall calculation fix
"""

from src.types.core import ExperimentConfig, DataScale, Dimension, SearchType, FilterConfig, HNSWParams
from src.pipelines.experiments import single_experiment_pipeline

def test_recall_fix():
    """Test that recall values are now calculated correctly"""
    
    print("🧪 Testing recall calculation fix...")
    
    config = ExperimentConfig(
        data_scale=DataScale.SMALL,
        dimension=Dimension(128),
        search_type=SearchType.PURE_VECTOR,
        filter_config=FilterConfig(enabled=False, category=None),
        hnsw_params=HNSWParams(ef_construction=64, ef_search=32, M=8),
        batch_size=1000,
        num_queries=10  # Small number for quick test
    )
    
    print(f"📋 Test config: {config.data_scale.name}-{config.dimension}d-{config.search_type.name}")
    print(f"📊 Number of queries: {config.num_queries}")
    
    try:
        experiment_io = single_experiment_pipeline(config)
        result = experiment_io.run()
        
        print(f"✅ Experiment completed successfully!")
        print(f"📈 Results:")
        print(f"   - Search results: {len(result.search_results)}")
        print(f"   - Recall@1: {result.accuracy.recall_at_1:.3f}")
        print(f"   - Recall@5: {result.accuracy.recall_at_5:.3f}")
        print(f"   - Recall@10: {result.accuracy.recall_at_10:.3f}")
        print(f"   - MRR: {result.accuracy.mean_reciprocal_rank:.3f}")
        
        if result.accuracy.recall_at_10 > 0:
            print(f"🎉 SUCCESS: Recall@10 is now {result.accuracy.recall_at_10:.3f} (was 0.000)")
            return True
        else:
            print(f"❌ FAILURE: Recall@10 is still 0.000")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recall_fix()
    if success:
        print("\n✅ Recall fix validation PASSED")
    else:
        print("\n❌ Recall fix validation FAILED")
