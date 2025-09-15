"""
Data exploration module for Kilter Board Predictor.

This module provides comprehensive data exploration capabilities including
visualization and basic statistical analysis to prepare data for machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class DataExplorer:
    """
    A comprehensive data exploration class for analyzing climbing route data.
    
    This class provides methods for data visualization, statistical analysis,
    and data quality assessment to prepare for machine learning model training.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataExplorer.
        
        Args:
            data: Optional pandas DataFrame containing the data to explore
        """
        self.data = data
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        if data is not None:
            self._analyze_columns()
    
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data into the explorer.
        
        Args:
            data: pandas DataFrame containing the data to explore
        """
        self.data = data
        self._analyze_columns()
    
    def _analyze_columns(self) -> None:
        """Analyze and categorize columns by data type."""
        if self.data is None:
            return
            
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self.data.select_dtypes(include=['datetime64']).columns.tolist()
    
    def data_overview(self) -> Dict[str, Any]:
        """
        Provide a comprehensive overview of the dataset.
        
        Returns:
            Dictionary containing dataset overview information
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        overview = {
            'shape': self.data.shape,
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'column_types': {
                'numeric': len(self.numeric_columns),
                'categorical': len(self.categorical_columns),
                'datetime': len(self.datetime_columns)
            },
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum(),
            'columns': {
                'numeric': self.numeric_columns,
                'categorical': self.categorical_columns,
                'datetime': self.datetime_columns
            }
        }
        
        print("=== DATASET OVERVIEW ===")
        print(f"Dataset Shape: {overview['shape']}")
        print(f"Memory Usage: {overview['memory_usage']}")
        print(f"Total Missing Values: {overview['missing_values']}")
        print(f"Duplicate Rows: {overview['duplicate_rows']}")
        print("\nColumn Types:")
        print(f"  - Numeric: {overview['column_types']['numeric']}")
        print(f"  - Categorical: {overview['column_types']['categorical']}")
        print(f"  - Datetime: {overview['column_types']['datetime']}")
        
        return overview
    
    def missing_values_analysis(self, visualize: bool = True) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            visualize: Whether to create visualizations
            
        Returns:
            DataFrame with missing value statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        missing_stats = pd.DataFrame({
            'Column': self.data.columns,
            'Missing_Count': self.data.isnull().sum(),
            'Missing_Percentage': (self.data.isnull().sum() / len(self.data)) * 100
        })
        missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
        
        if visualize and missing_stats['Missing_Count'].sum() > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot of missing values
            missing_cols = missing_stats[missing_stats['Missing_Count'] > 0]
            if not missing_cols.empty:
                ax1.bar(missing_cols['Column'], missing_cols['Missing_Count'])
                ax1.set_title('Missing Values Count by Column')
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Missing Count')
                ax1.tick_params(axis='x', rotation=45)
                
                # Heatmap of missing values pattern
                if len(missing_cols) > 1:
                    missing_matrix = self.data[missing_cols['Column']].isnull()
                    sns.heatmap(missing_matrix, yticklabels=False, cbar=True, ax=ax2)
                    ax2.set_title('Missing Values Pattern')
            
            plt.tight_layout()
            plt.show()
        
        return missing_stats
    
    def plot_distributions(self, columns: Optional[List[str]] = None, 
                          plot_type: str = 'histogram') -> None:
        """
        Plot distributions of numeric columns.
        
        Args:
            columns: List of columns to plot. If None, plot all numeric columns
            plot_type: Type of plot ('histogram', 'box', 'violin')
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("No numeric columns found to plot.")
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if len(columns) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if plot_type == 'histogram':
                self.data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
            elif plot_type == 'box':
                self.data.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Box Plot of {col}')
            elif plot_type == 'violin':
                data_clean = self.data[col].dropna()
                axes[i].violinplot([data_clean])
                axes[i].set_title(f'Violin Plot of {col}')
                axes[i].set_xticks([1])
                axes[i].set_xticklabels([col])
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, method: str = 'pearson', 
                           visualize: bool = True) -> pd.DataFrame:
        """
        Analyze correlations between numeric variables.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            visualize: Whether to create correlation heatmap
            
        Returns:
            Correlation matrix
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if not self.numeric_columns:
            print("No numeric columns found for correlation analysis.")
            return pd.DataFrame()
        
        corr_matrix = self.data[self.numeric_columns].corr(method=method)
        
        if visualize:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title(f'{method.capitalize()} Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        return corr_matrix
    
    def plot_categorical_distributions(self, columns: Optional[List[str]] = None,
                                     max_categories: int = 20) -> None:
        """
        Plot distributions of categorical columns.
        
        Args:
            columns: List of columns to plot. If None, plot all categorical columns
            max_categories: Maximum number of categories to show per plot
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.categorical_columns
        
        if not columns:
            print("No categorical columns found to plot.")
            return
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        if len(columns) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            value_counts = self.data[col].value_counts().head(max_categories)
            value_counts.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_scatter_matrix(self, columns: Optional[List[str]] = None) -> None:
        """
        Create an interactive scatter plot matrix using Plotly.
        
        Args:
            columns: List of numeric columns to include. If None, use all numeric columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if columns is None:
            columns = self.numeric_columns[:6]  # Limit to 6 columns for readability
        
        if not columns:
            print("No numeric columns found for scatter matrix.")
            return
        
        fig = px.scatter_matrix(self.data, dimensions=columns,
                               title="Interactive Scatter Plot Matrix")
        fig.update_layout(width=800, height=800)
        fig.show()
    
    def outlier_detection(self, method: str = 'iqr', visualize: bool = True) -> Dict[str, pd.Series]:
        """
        Detect outliers in numeric columns.
        
        Args:
            method: Method for outlier detection ('iqr', 'zscore')
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with outlier information for each column
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = self.data[(self.data[col] < lower_bound) | 
                                        (self.data[col] > upper_bound)][col]
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers[col] = self.data[z_scores > 3][col]
        
        if visualize:
            n_cols = min(3, len(self.numeric_columns))
            n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if len(self.numeric_columns) == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.numeric_columns):
                self.data.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Outliers in {col} ({len(outliers[col])} detected)')
            
            # Hide empty subplots
            for i in range(len(self.numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        return outliers
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive data exploration report.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            String containing the full report
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        report = []
        report.append("KILTER BOARD PREDICTOR - DATA EXPLORATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Dataset overview
        overview = self.data_overview()
        report.append("DATASET OVERVIEW:")
        report.append(f"Shape: {overview['shape']}")
        report.append(f"Memory Usage: {overview['memory_usage']}")
        report.append(f"Missing Values: {overview['missing_values']}")
        report.append(f"Duplicate Rows: {overview['duplicate_rows']}")
        report.append("")
        
        # Column information
        report.append("COLUMN INFORMATION:")
        report.append(f"Numeric Columns ({len(self.numeric_columns)}): {', '.join(self.numeric_columns)}")
        report.append(f"Categorical Columns ({len(self.categorical_columns)}): {', '.join(self.categorical_columns)}")
        report.append(f"Datetime Columns ({len(self.datetime_columns)}): {', '.join(self.datetime_columns)}")
        report.append("")
        
        # Basic statistics for numeric columns
        if self.numeric_columns:
            report.append("NUMERIC COLUMNS STATISTICS:")
            stats = self.data[self.numeric_columns].describe()
            report.append(stats.to_string())
            report.append("")
        
        # Categorical columns summary
        if self.categorical_columns:
            report.append("CATEGORICAL COLUMNS SUMMARY:")
            for col in self.categorical_columns:
                unique_count = self.data[col].nunique()
                most_common = self.data[col].value_counts().head(3)
                report.append(f"{col}: {unique_count} unique values")
                report.append(f"  Most common: {most_common.to_dict()}")
            report.append("")
        
        # Missing values summary
        missing_stats = self.missing_values_analysis(visualize=False)
        missing_cols = missing_stats[missing_stats['Missing_Count'] > 0]
        if not missing_cols.empty:
            report.append("MISSING VALUES SUMMARY:")
            report.append(missing_cols.to_string(index=False))
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text