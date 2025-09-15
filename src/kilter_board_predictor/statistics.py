"""
Basic statistics module for Kilter Board Predictor.

This module provides statistical analysis functions to understand data characteristics
and prepare for machine learning model training.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class BasicStatistics:
    """
    A class for computing basic statistical measures and tests on climbing route data.
    
    This class provides methods for descriptive statistics, hypothesis testing,
    and statistical analysis to support machine learning preparation.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the BasicStatistics analyzer.
        
        Args:
            data: Optional pandas DataFrame containing the data to analyze
        """
        self.data = data
        self.numeric_columns = []
        self.categorical_columns = []
        
        if data is not None:
            self._analyze_columns()
    
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data into the statistics analyzer.
        
        Args:
            data: pandas DataFrame containing the data to analyze
        """
        self.data = data
        self._analyze_columns()
    
    def _analyze_columns(self) -> None:
        """Analyze and categorize columns by data type."""
        if self.data is None:
            return
            
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def descriptive_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive descriptive statistics for numeric columns.
        
        Args:
            columns: List of columns to analyze. If None, analyze all numeric columns
            
        Returns:
            Dictionary with descriptive statistics for each column
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            raise ValueError("No numeric columns found for analysis.")
        
        statistics = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            series = self.data[col].dropna()
            
            if len(series) == 0:
                statistics[col] = {'error': 'No valid data points'}
                continue
            
            stats_dict = {
                'count': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'mode': series.mode().iloc[0] if not series.mode().empty else np.nan,
                'std': series.std(),
                'variance': series.var(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'coefficient_of_variation': (series.std() / series.mean()) * 100 if series.mean() != 0 else np.nan
            }
            
            statistics[col] = stats_dict
        
        return statistics
    
    def normality_tests(self, columns: Optional[List[str]] = None, 
                       alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Perform normality tests on numeric columns.
        
        Args:
            columns: List of columns to test. If None, test all numeric columns
            alpha: Significance level for tests
            
        Returns:
            Dictionary with normality test results for each column
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            raise ValueError("No numeric columns found for testing.")
        
        results = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            series = self.data[col].dropna()
            
            if len(series) < 3:
                results[col] = {'error': 'Not enough data points for testing'}
                continue
            
            # Shapiro-Wilk test (best for small samples)
            if len(series) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                shapiro_normal = shapiro_p > alpha
            else:
                shapiro_stat, shapiro_p, shapiro_normal = np.nan, np.nan, 'Sample too large'
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
            ks_normal = ks_p > alpha
            
            # D'Agostino-Pearson test
            if len(series) >= 20:
                dagostino_stat, dagostino_p = stats.normaltest(series)
                dagostino_normal = dagostino_p > alpha
            else:
                dagostino_stat, dagostino_p, dagostino_normal = np.nan, np.nan, 'Sample too small'
            
            results[col] = {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_normal
                },
                'kolmogorov_smirnov': {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'is_normal': ks_normal
                },
                'dagostino_pearson': {
                    'statistic': dagostino_stat,
                    'p_value': dagostino_p,
                    'is_normal': dagostino_normal
                }
            }
        
        return results
    
    def outlier_statistics(self, columns: Optional[List[str]] = None,
                          method: str = 'iqr') -> Dict[str, Dict[str, Any]]:
        """
        Calculate outlier statistics for numeric columns.
        
        Args:
            columns: List of columns to analyze. If None, analyze all numeric columns
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dictionary with outlier statistics for each column
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            raise ValueError("No numeric columns found for analysis.")
        
        outlier_stats = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            series = self.data[col].dropna()
            
            if len(series) == 0:
                outlier_stats[col] = {'error': 'No valid data points'}
                continue
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = series[z_scores > 3]
                lower_bound = series.mean() - 3 * series.std()
                upper_bound = series.mean() + 3 * series.std()
                
            elif method == 'modified_zscore':
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                outliers = series[np.abs(modified_z_scores) > 3.5]
                lower_bound = median - 3.5 * mad / 0.6745
                upper_bound = median + 3.5 * mad / 0.6745
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_stats[col] = {
                'method': method,
                'total_count': len(series),
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist()
            }
        
        return outlier_stats
    
    def correlation_statistics(self, columns: Optional[List[str]] = None,
                             method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation statistics and identify strong correlations.
        
        Args:
            columns: List of columns to analyze. If None, analyze all numeric columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns
        
        if len(columns) < 2:
            raise ValueError("At least 2 numeric columns required for correlation analysis.")
        
        # Calculate correlation matrix
        corr_matrix = self.data[columns].corr(method=method)
        
        # Find strong correlations (excluding self-correlations)
        strong_correlations = []
        high_correlations = []
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    pair = (columns[i], columns[j], corr_value)
                    
                    if abs(corr_value) >= 0.7:
                        strong_correlations.append(pair)
                    elif abs(corr_value) >= 0.5:
                        high_correlations.append(pair)
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Calculate summary statistics
        corr_values = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    corr_values.append(abs(corr_value))
        
        return {
            'method': method,
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,  # |r| >= 0.7
            'high_correlations': high_correlations,      # 0.5 <= |r| < 0.7
            'summary': {
                'mean_absolute_correlation': np.mean(corr_values) if corr_values else 0,
                'max_correlation': max(corr_values) if corr_values else 0,
                'min_correlation': min(corr_values) if corr_values else 0,
                'num_strong_pairs': len(strong_correlations),
                'num_high_pairs': len(high_correlations)
            }
        }
    
    def categorical_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for categorical columns.
        
        Args:
            columns: List of columns to analyze. If None, analyze all categorical columns
            
        Returns:
            Dictionary with categorical statistics for each column
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.categorical_columns
        
        if not columns:
            raise ValueError("No categorical columns found for analysis.")
        
        cat_stats = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            series = self.data[col].dropna()
            
            if len(series) == 0:
                cat_stats[col] = {'error': 'No valid data points'}
                continue
            
            value_counts = series.value_counts()
            mode_value = value_counts.index[0] if len(value_counts) > 0 else None
            
            cat_stats[col] = {
                'count': len(series),
                'unique_values': series.nunique(),
                'mode': mode_value,
                'mode_frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'mode_percentage': (value_counts.iloc[0] / len(series)) * 100 if len(value_counts) > 0 else 0,
                'entropy': stats.entropy(value_counts.values),
                'value_counts': value_counts.to_dict(),
                'missing_values': self.data[col].isnull().sum(),
                'missing_percentage': (self.data[col].isnull().sum() / len(self.data)) * 100
            }
        
        return cat_stats
    
    def hypothesis_tests(self, target_column: str, feature_columns: Optional[List[str]] = None,
                        test_type: str = 'auto') -> Dict[str, Dict[str, Any]]:
        """
        Perform hypothesis tests between features and target variable.
        
        Args:
            target_column: Name of the target variable column
            feature_columns: List of feature columns to test. If None, test all other columns
            test_type: Type of test ('auto', 't_test', 'chi_square', 'anova', 'mann_whitney')
            
        Returns:
            Dictionary with hypothesis test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != target_column]
        
        results = {}
        target_is_numeric = target_column in self.numeric_columns
        
        for feature_col in feature_columns:
            if feature_col not in self.data.columns or feature_col == target_column:
                continue
            
            feature_is_numeric = feature_col in self.numeric_columns
            
            # Remove rows with missing values in either column
            clean_data = self.data[[target_column, feature_col]].dropna()
            
            if len(clean_data) < 3:
                results[feature_col] = {'error': 'Not enough data points for testing'}
                continue
            
            target_data = clean_data[target_column]
            feature_data = clean_data[feature_col]
            
            if test_type == 'auto':
                # Automatically choose appropriate test
                if target_is_numeric and feature_is_numeric:
                    # Correlation test (Pearson)
                    statistic, p_value = stats.pearsonr(target_data, feature_data)
                    test_name = 'Pearson Correlation'
                elif target_is_numeric and not feature_is_numeric:
                    # ANOVA or t-test depending on number of groups
                    groups = [group[target_column].values for name, group in clean_data.groupby(feature_col)]
                    if len(groups) == 2:
                        statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                        test_name = 'Independent t-test'
                    else:
                        statistic, p_value = stats.f_oneway(*groups)
                        test_name = 'One-way ANOVA'
                elif not target_is_numeric and not feature_is_numeric:
                    # Chi-square test
                    contingency_table = pd.crosstab(target_data, feature_data)
                    statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    test_name = 'Chi-square test'
                else:  # not target_is_numeric and feature_is_numeric
                    # ANOVA (treating numeric feature as continuous)
                    groups = [group[feature_col].values for name, group in clean_data.groupby(target_column)]
                    if len(groups) >= 2:
                        statistic, p_value = stats.f_oneway(*groups)
                        test_name = 'One-way ANOVA'
                    else:
                        results[feature_col] = {'error': 'Not enough groups for testing'}
                        continue
            
            else:
                # Use specified test type
                if test_type == 't_test':
                    if feature_col in self.categorical_columns:
                        groups = [group[target_column].values for name, group in clean_data.groupby(feature_col)]
                        if len(groups) == 2:
                            statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                            test_name = 'Independent t-test'
                        else:
                            results[feature_col] = {'error': 'T-test requires exactly 2 groups'}
                            continue
                    else:
                        results[feature_col] = {'error': 'T-test requires categorical feature'}
                        continue
                        
                elif test_type == 'chi_square':
                    if not target_is_numeric and not feature_is_numeric:
                        contingency_table = pd.crosstab(target_data, feature_data)
                        statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
                        test_name = 'Chi-square test'
                    else:
                        results[feature_col] = {'error': 'Chi-square test requires categorical variables'}
                        continue
                        
                elif test_type == 'anova':
                    if feature_col in self.categorical_columns:
                        groups = [group[target_column].values for name, group in clean_data.groupby(feature_col)]
                        statistic, p_value = stats.f_oneway(*groups)
                        test_name = 'One-way ANOVA'
                    else:
                        results[feature_col] = {'error': 'ANOVA requires categorical feature'}
                        continue
                        
                elif test_type == 'mann_whitney':
                    if feature_col in self.categorical_columns:
                        groups = [group[target_column].values for name, group in clean_data.groupby(feature_col)]
                        if len(groups) == 2:
                            statistic, p_value = stats.mannwhitneyu(groups[0], groups[1])
                            test_name = 'Mann-Whitney U test'
                        else:
                            results[feature_col] = {'error': 'Mann-Whitney test requires exactly 2 groups'}
                            continue
                    else:
                        results[feature_col] = {'error': 'Mann-Whitney test requires categorical feature'}
                        continue
            
            results[feature_col] = {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_size': len(clean_data)
            }
        
        return results
    
    def generate_statistics_report(self, target_column: Optional[str] = None) -> str:
        """
        Generate a comprehensive statistical analysis report.
        
        Args:
            target_column: Optional target variable for hypothesis testing
            
        Returns:
            String containing the full statistical report
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        report = []
        report.append("KILTER BOARD PREDICTOR - STATISTICAL ANALYSIS REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Descriptive statistics for numeric columns
        if self.numeric_columns:
            report.append("DESCRIPTIVE STATISTICS (Numeric Columns):")
            report.append("-" * 45)
            desc_stats = self.descriptive_statistics()
            
            for col, stats in desc_stats.items():
                if 'error' in stats:
                    report.append(f"{col}: {stats['error']}")
                    continue
                    
                report.append(f"\n{col}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Median: {stats['median']:.4f}")
                report.append(f"  Std Dev: {stats['std']:.4f}")
                report.append(f"  Min: {stats['min']:.4f}")
                report.append(f"  Max: {stats['max']:.4f}")
                report.append(f"  Skewness: {stats['skewness']:.4f}")
                report.append(f"  Kurtosis: {stats['kurtosis']:.4f}")
            
            report.append("")
        
        # Categorical statistics
        if self.categorical_columns:
            report.append("CATEGORICAL STATISTICS:")
            report.append("-" * 25)
            cat_stats = self.categorical_statistics()
            
            for col, stats in cat_stats.items():
                if 'error' in stats:
                    report.append(f"{col}: {stats['error']}")
                    continue
                    
                report.append(f"\n{col}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Unique Values: {stats['unique_values']}")
                report.append(f"  Mode: {stats['mode']}")
                report.append(f"  Mode Frequency: {stats['mode_frequency']} ({stats['mode_percentage']:.2f}%)")
                report.append(f"  Entropy: {stats['entropy']:.4f}")
            
            report.append("")
        
        # Correlation analysis
        if len(self.numeric_columns) >= 2:
            report.append("CORRELATION ANALYSIS:")
            report.append("-" * 20)
            corr_stats = self.correlation_statistics()
            
            report.append(f"Method: {corr_stats['method']}")
            report.append(f"Mean Absolute Correlation: {corr_stats['summary']['mean_absolute_correlation']:.4f}")
            report.append(f"Maximum Correlation: {corr_stats['summary']['max_correlation']:.4f}")
            
            if corr_stats['strong_correlations']:
                report.append("\nStrong Correlations (|r| >= 0.7):")
                for var1, var2, corr in corr_stats['strong_correlations'][:5]:  # Top 5
                    report.append(f"  {var1} - {var2}: {corr:.4f}")
            
            report.append("")
        
        # Normality tests
        if self.numeric_columns:
            report.append("NORMALITY TESTS:")
            report.append("-" * 16)
            normality_results = self.normality_tests()
            
            for col, results in normality_results.items():
                if 'error' in results:
                    report.append(f"{col}: {results['error']}")
                    continue
                    
                report.append(f"\n{col}:")
                if isinstance(results['shapiro_wilk']['is_normal'], bool):
                    report.append(f"  Shapiro-Wilk: p={results['shapiro_wilk']['p_value']:.4f}, Normal: {results['shapiro_wilk']['is_normal']}")
                report.append(f"  Kolmogorov-Smirnov: p={results['kolmogorov_smirnov']['p_value']:.4f}, Normal: {results['kolmogorov_smirnov']['is_normal']}")
            
            report.append("")
        
        # Hypothesis tests with target variable
        if target_column and target_column in self.data.columns:
            report.append(f"HYPOTHESIS TESTS (Target: {target_column}):")
            report.append("-" * 35)
            test_results = self.hypothesis_tests(target_column)
            
            significant_features = []
            for feature, result in test_results.items():
                if 'error' not in result and result['significant']:
                    significant_features.append((feature, result['p_value'], result['test_name']))
            
            significant_features.sort(key=lambda x: x[1])  # Sort by p-value
            
            if significant_features:
                report.append("Significant Associations (p < 0.05):")
                for feature, p_value, test_name in significant_features:
                    report.append(f"  {feature}: p={p_value:.6f} ({test_name})")
            else:
                report.append("No significant associations found (p < 0.05)")
            
            report.append("")
        
        return "\n".join(report)