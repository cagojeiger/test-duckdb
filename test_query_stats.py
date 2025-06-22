#!/usr/bin/env python3
"""Test script to verify query time statistics implementation"""

from src.pure.analyzers.performance_analyzer import calculate_statistical_summary

def test_statistical_summary():
    """Test the statistical summary calculation with sample data"""
    test_values = [10.5, 12.3, 8.7, 15.2, 9.8, 11.1, 13.4, 7.9, 14.6, 10.2]
    stats = calculate_statistical_summary(test_values)
    
    if stats is None:
        print("ERROR: calculate_statistical_summary returned None")
        return
    
    print("Query Time Statistics Test:")
    print(f"Mean: {stats.mean:.2f} ms")
    print(f"Median: {stats.median:.2f} ms")
    print(f"Min: {stats.min_value:.2f} ms")
    print(f"Max: {stats.max_value:.2f} ms")
    print(f"Std Dev: {stats.std_dev:.2f} ms")
    print(f"P5: {stats.percentile_5:.2f} ms")
    print(f"P10: {stats.percentile_10:.2f} ms")
    print(f"P90: {stats.percentile_90:.2f} ms")
    print(f"P95: {stats.percentile_95:.2f} ms")
    
    assert stats.percentile_5 <= stats.percentile_10
    assert stats.percentile_10 <= stats.median
    assert stats.median <= stats.percentile_90
    assert stats.percentile_90 <= stats.percentile_95
    print("✅ Percentiles are correctly ordered")

if __name__ == "__main__":
    test_statistical_summary()
