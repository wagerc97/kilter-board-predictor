#!/usr/bin/env python3
"""
Basic Data Exploration Example for Kilter Board Predictor

This script demonstrates how to use the DataExplorer and BasicStatistics classes
to analyze climbing route data for machine learning preparation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kilter_board_predictor import DataExplorer, BasicStatistics


def create_sample_data(n_routes=500):
    """Create sample climbing route data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic climbing route data
    data = {
        'route_id': range(1, n_routes + 1),
        'angle': np.random.choice([0, 20, 30, 40, 50, 60, 70], n_routes, p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]),
        'num_holds': np.random.randint(8, 25, n_routes),
        'height': np.random.normal(3.5, 0.8, n_routes),
        'wall_type': np.random.choice(['vertical', 'overhang', 'slab'], n_routes, p=[0.4, 0.5, 0.1]),
        'hold_type': np.random.choice(['crimps', 'jugs', 'slopers', 'pinches', 'mixed'], n_routes, p=[0.3, 0.2, 0.2, 0.1, 0.2]),
        'setter_experience': np.random.randint(1, 10, n_routes),
        'completion_rate': np.random.beta(2, 3, n_routes),
        'average_attempts': np.random.poisson(3, n_routes) + 1,
        'grade_v_scale': np.random.randint(0, 17, n_routes),
    }
    
    # Create realistic correlations
    angle_bonus = (data['angle'] / 70) * 3
    data['grade_v_scale'] = np.clip(data['grade_v_scale'] + angle_bonus, 0, 16).astype(int)
    
    hold_penalty = (data['num_holds'] - 15) * -0.2
    data['grade_v_scale'] = np.clip(data['grade_v_scale'] + hold_penalty, 0, 16).astype(int)
    
    grade_difficulty = data['grade_v_scale'] / 16
    data['completion_rate'] = np.clip(1 - grade_difficulty * 0.7 + np.random.normal(0, 0.1, n_routes), 0.05, 0.95)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'completion_rate'] = np.nan
    
    return df


def main():
    """Main function to demonstrate data exploration capabilities."""
    print("=" * 60)
    print("KILTER BOARD PREDICTOR - DATA EXPLORATION DEMO")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample climbing route data...")
    df = create_sample_data(500)
    print(f"   Created dataset with {len(df)} routes and {len(df.columns)} features")
    
    # Initialize explorers
    print("\n2. Initializing data exploration tools...")
    explorer = DataExplorer(df)
    stats_analyzer = BasicStatistics(df)
    
    # Basic data overview
    print("\n3. Data Overview:")
    print("-" * 20)
    overview = explorer.data_overview()
    
    # Missing values analysis
    print("\n4. Missing Values Analysis:")
    print("-" * 30)
    missing_stats = explorer.missing_values_analysis(visualize=False)
    missing_cols = missing_stats[missing_stats['Missing_Count'] > 0]
    if not missing_cols.empty:
        print(missing_cols.to_string(index=False))
    else:
        print("   No missing values detected")
    
    # Descriptive statistics
    print("\n5. Descriptive Statistics:")
    print("-" * 27)
    desc_stats = stats_analyzer.descriptive_statistics()
    
    for col, stats in desc_stats.items():
        if 'error' not in stats:
            print(f"\n   {col}:")
            print(f"     Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
            print(f"     Std: {stats['std']:.3f}, Range: {stats['range']:.3f}")
            print(f"     Skewness: {stats['skewness']:.3f}")
    
    # Correlation analysis
    print("\n6. Correlation Analysis:")
    print("-" * 23)
    corr_stats = stats_analyzer.correlation_statistics()
    print(f"   Mean absolute correlation: {corr_stats['summary']['mean_absolute_correlation']:.4f}")
    print(f"   Strong correlations (|r| >= 0.7): {corr_stats['summary']['num_strong_pairs']}")
    
    if corr_stats['strong_correlations']:
        print("   Strong correlation pairs:")
        for var1, var2, corr in corr_stats['strong_correlations']:
            print(f"     - {var1} ↔ {var2}: {corr:.4f}")
    
    # Categorical analysis
    print("\n7. Categorical Variables:")
    print("-" * 24)
    cat_stats = stats_analyzer.categorical_statistics()
    
    for col, stats in cat_stats.items():
        if 'error' not in stats:
            print(f"\n   {col}:")
            print(f"     Unique values: {stats['unique_values']}")
            print(f"     Mode: {stats['mode']} ({stats['mode_percentage']:.1f}%)")
    
    # Hypothesis testing
    print("\n8. Feature Significance (vs. grade_v_scale):")
    print("-" * 44)
    hypothesis_results = stats_analyzer.hypothesis_tests('grade_v_scale')
    
    significant_features = []
    for feature, result in hypothesis_results.items():
        if 'error' not in result:
            significant_features.append((feature, result['p_value'], result['significant']))
    
    significant_features.sort(key=lambda x: x[1])
    
    for feature, p_value, significant in significant_features:
        status = "***" if significant else "   "
        print(f"   {status} {feature}: p={p_value:.6f}")
    
    print("\n   *** indicates p < 0.05 (statistically significant)")
    
    # Outlier detection
    print("\n9. Outlier Detection:")
    print("-" * 19)
    outlier_stats = stats_analyzer.outlier_statistics(method='iqr')
    
    for col, stats in outlier_stats.items():
        if 'error' not in stats:
            print(f"   {col}: {stats['outlier_count']} outliers ({stats['outlier_percentage']:.1f}%)")
    
    # Generate comprehensive reports
    print("\n10. Generating Reports...")
    print("-" * 23)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save exploration report
    exploration_report = explorer.generate_report()
    exploration_file = os.path.join(data_dir, 'exploration_report.txt')
    with open(exploration_file, 'w') as f:
        f.write(exploration_report)
    print(f"   Data exploration report saved to: {exploration_file}")
    
    # Save statistics report
    stats_report = stats_analyzer.generate_statistics_report(target_column='grade_v_scale')
    stats_file = os.path.join(data_dir, 'statistics_report.txt')
    with open(stats_file, 'w') as f:
        f.write(stats_report)
    print(f"   Statistical analysis report saved to: {stats_file}")
    
    # Save sample data
    data_file = os.path.join(data_dir, 'sample_climbing_data.csv')
    df.to_csv(data_file, index=False)
    print(f"   Sample dataset saved to: {data_file}")
    
    print("\n" + "=" * 60)
    print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nKey Findings:")
    print("- Dataset contains realistic climbing route features")
    print(f"- {len([f for f, p, s in significant_features if s])} features show significant relationship with grade")
    print(f"- {corr_stats['summary']['num_strong_pairs']} strong correlations detected")
    print("- Data is ready for machine learning preprocessing")
    
    print("\nNext Steps:")
    print("1. Load real climbing route data using the same analysis framework")
    print("2. Perform feature engineering based on insights")
    print("3. Prepare data for machine learning model training")
    print("4. Consider the relationships found for model selection")


if __name__ == "__main__":
    main()