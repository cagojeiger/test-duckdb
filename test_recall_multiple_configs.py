#!/usr/bin/env python3
"""
Test script to validate recall fix with multiple configurations
"""

from src.types.core import ExperimentConfig, DataScale, Dimension, SearchType, FilterConfig, HNSWParams
from src.pipelines.experiments import single_experiment_pipeline

def test_multiple_configurations():
    """Test recall fix with different configurations"""
    
    print("🧪 Testing recall calculation fix with multiple configurations...")
    
    test_configs = [
        {
            'name': 'SMALL-128d-PURE_VECTOR',
            'config': ExperimentConfig(
                data_scale=DataScale.SMALL,
                dimension=Dimension(128),
                search_type=SearchType.PURE_VECTOR,
                filter_config=FilterConfig(enabled=False, category=None),
                hnsw_params=HNSWParams(ef_construction=64, ef_search=32, M=8),
                batch_size=1000,
                num_queries=5
            )
        },
        {
            'name': 'SMALL-256d-PURE_VECTOR',
            'config': ExperimentConfig(
                data_scale=DataScale.SMALL,
                dimension=Dimension(256),
                search_type=SearchType.PURE_VECTOR,
                filter_config=FilterConfig(enabled=False, category=None),
                hnsw_params=HNSWParams(ef_construction=64, ef_search=32, M=8),
                batch_size=1000,
                num_queries=5
            )
        },
        {
            'name': 'SMALL-128d-HYBRID',
            'config': ExperimentConfig(
                data_scale=DataScale.SMALL,
                dimension=Dimension(128),
                search_type=SearchType.HYBRID,
                filter_config=FilterConfig(enabled=False, category=None),
                hnsw_params=HNSWParams(ef_construction=64, ef_search=32, M=8),
                batch_size=1000,
                num_queries=5
            )
        }
    ]
    
    all_passed = True
    
    for test_case in test_configs:
        print(f"\n📋 Testing: {test_case['name']}")
        
        try:
            experiment_io = single_experiment_pipeline(test_case['config'])
            result = experiment_io.run()
            
            print(f"   ✅ Completed successfully")
            print(f"   📈 Recall@1: {result.accuracy.recall_at_1:.3f}")
            print(f"   📈 Recall@5: {result.accuracy.recall_at_5:.3f}")
            print(f"   📈 Recall@10: {result.accuracy.recall_at_10:.3f}")
            print(f"   📈 MRR: {result.accuracy.mean_reciprocal_rank:.3f}")
            
            if result.accuracy.recall_at_10 > 0:
                print(f"   🎉 PASS: Non-zero recall values")
            else:
                print(f"   ❌ FAIL: Recall still zero")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_multiple_configurations()
    if success:
        print("\n✅ All configuration tests PASSED")
    else:
        print("\n❌ Some configuration tests FAILED")
