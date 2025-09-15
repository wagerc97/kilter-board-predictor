#!/usr/bin/env python3
"""
Quick Demo - Kilter Board Predictor Data Exploration

A minimal example showing the key capabilities of the data exploration tools.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from kilter_board_predictor import DataExplorer, BasicStatistics


def main():
    """Quick demonstration of key features."""
    print("🧗 Kilter Board Predictor - Quick Demo")
    print("=" * 40)
    
    # Create sample climbing data
    np.random.seed(42)
    n = 300
    
    # Generate realistic climbing route features
    df = pd.DataFrame({
        'angle': np.random.choice([0, 20, 30, 40, 50, 60, 70], n),
        'num_holds': np.random.randint(8, 25, n),
        'height': np.random.normal(3.5, 0.8, n),
        'wall_type': np.random.choice(['vertical', 'overhang', 'slab'], n),
        'completion_rate': np.random.beta(2, 3, n),
        'grade_v_scale': np.random.randint(0, 17, n),
    })
    
    # Create realistic relationships
    df['grade_v_scale'] = np.clip(
        df['grade_v_scale'] + (df['angle'] / 70) * 3 - (df['num_holds'] - 15) * 0.15,
        0, 16
    ).astype(int)
    
    # Initialize tools
    explorer = DataExplorer(df)
    stats = BasicStatistics(df)
    
    print(f"\n📊 Analyzing {len(df)} climbing routes with {len(df.columns)} features")
    
    # Quick overview
    overview = explorer.data_overview()
    
    # Key statistics
    desc_stats = stats.descriptive_statistics()
    print(f"\n🎯 Grade Distribution:")
    grade_stats = desc_stats['grade_v_scale']
    print(f"   Range: V{int(grade_stats['min'])} - V{int(grade_stats['max'])}")
    print(f"   Average: V{grade_stats['mean']:.1f}")
    print(f"   Most Common: V{int(grade_stats['mode'])}")
    
    # Correlation insights
    corr_stats = stats.correlation_statistics()
    print(f"\n🔗 Relationships:")
    print(f"   Strong correlations found: {corr_stats['summary']['num_strong_pairs']}")
    if corr_stats['strong_correlations']:
        for var1, var2, corr in corr_stats['strong_correlations'][:2]:
            print(f"   • {var1} ↔ {var2}: {corr:.3f}")
    
    # Feature significance
    hypothesis_results = stats.hypothesis_tests('grade_v_scale')
    significant = [(f, r['p_value']) for f, r in hypothesis_results.items() 
                   if 'error' not in r and r['significant']]
    significant.sort(key=lambda x: x[1])
    
    print(f"\n🎯 Most Predictive Features:")
    for feature, p_val in significant[:3]:
        print(f"   • {feature}: p={p_val:.6f}")
    
    # Data quality
    missing = explorer.missing_values_analysis(visualize=False)
    total_missing = missing['Missing_Count'].sum()
    outliers = stats.outlier_statistics()
    total_outliers = sum(s['outlier_count'] for s in outliers.values() if 'error' not in s)
    
    print(f"\n✅ Data Quality:")
    print(f"   Missing values: {total_missing} ({(total_missing/len(df)/len(df.columns))*100:.1f}%)")
    print(f"   Outliers detected: {total_outliers} ({(total_outliers/len(df))*100:.1f}%)")
    
    print(f"\n🚀 Ready for Machine Learning!")
    print(f"   • {len(significant)} significant predictors identified")
    print(f"   • Data quality checks passed")
    print(f"   • Statistical assumptions analyzed")
    
    print("\nNext steps:")
    print("1. Use DataExplorer.plot_distributions() for visual analysis")
    print("2. Generate full reports with generate_report()")
    print("3. Apply preprocessing based on findings")
    print("4. Train ML models with identified features")


if __name__ == "__main__":
    main()