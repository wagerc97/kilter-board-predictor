# Kilter Board Predictor

A comprehensive machine learning project for predicting climbing route grades on Kilter Board climbing walls. This project provides robust data exploration and statistical analysis tools to prepare climbing route data for machine learning model training.

## Features

### 🔍 Data Exploration
- **Comprehensive Data Overview**: Dataset shape, memory usage, column types, missing values, and duplicate detection
- **Missing Values Analysis**: Statistical analysis and visualization of missing data patterns
- **Distribution Analysis**: Histograms, box plots, and violin plots for numeric variables
- **Categorical Analysis**: Frequency distributions and entropy calculations for categorical variables
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlation matrices with visualization
- **Outlier Detection**: IQR, Z-score, and Modified Z-score methods with visualization
- **Interactive Visualizations**: Plotly-based scatter plot matrices and interactive charts

### 📊 Statistical Analysis
- **Descriptive Statistics**: Mean, median, mode, standard deviation, skewness, kurtosis, and more
- **Normality Testing**: Shapiro-Wilk, Kolmogorov-Smirnov, and D'Agostino-Pearson tests
- **Hypothesis Testing**: Automatic test selection (t-tests, ANOVA, Chi-square) based on data types
- **Correlation Statistics**: Identification of strong and moderate correlations with significance testing
- **Categorical Statistics**: Mode analysis, entropy calculations, and frequency distributions
- **Comprehensive Reporting**: Automated generation of detailed statistical reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wagerc97/kilter-board-predictor.git
cd kilter-board-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using Python Scripts

Run the basic exploration example:
```bash
python examples/basic_exploration.py
```

### Using in Your Code

```python
import pandas as pd
from kilter_board_predictor import DataExplorer, BasicStatistics

# Load your climbing route data
df = pd.read_csv('your_climbing_data.csv')

# Initialize explorers
explorer = DataExplorer(df)
stats_analyzer = BasicStatistics(df)

# Get data overview
overview = explorer.data_overview()

# Analyze missing values with visualization
missing_stats = explorer.missing_values_analysis(visualize=True)

# Plot distributions
explorer.plot_distributions(plot_type='histogram')
explorer.plot_categorical_distributions()

# Correlation analysis
corr_matrix = explorer.correlation_analysis(method='pearson', visualize=True)

# Detect outliers
outliers = explorer.outlier_detection(method='iqr', visualize=True)

# Statistical analysis
desc_stats = stats_analyzer.descriptive_statistics()
normality_results = stats_analyzer.normality_tests()
hypothesis_results = stats_analyzer.hypothesis_tests('target_column')

# Generate comprehensive reports
exploration_report = explorer.generate_report()
statistics_report = stats_analyzer.generate_statistics_report('target_column')
```

### Using Jupyter Notebooks

Open and run the example notebook:
```bash
jupyter notebook notebooks/data_exploration_example.ipynb
```

## Project Structure

```
kilter-board-predictor/
│
├── src/kilter_board_predictor/
│   ├── __init__.py              # Package initialization
│   ├── data_exploration.py      # DataExplorer class
│   └── statistics.py            # BasicStatistics class
│
├── notebooks/
│   └── data_exploration_example.ipynb  # Comprehensive example notebook
│
├── examples/
│   └── basic_exploration.py     # Simple Python script example
│
├── data/                        # Generated data and reports
│   ├── sample_climbing_data.csv # Sample dataset
│   ├── exploration_report.txt   # Data exploration report
│   └── statistics_report.txt    # Statistical analysis report
│
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Key Classes

### DataExplorer

The `DataExplorer` class provides comprehensive data exploration capabilities:

- `data_overview()`: Get dataset shape, memory usage, and column information
- `missing_values_analysis()`: Analyze and visualize missing data patterns
- `plot_distributions()`: Create histograms, box plots, or violin plots
- `correlation_analysis()`: Compute and visualize correlation matrices
- `plot_categorical_distributions()`: Analyze categorical variable distributions
- `outlier_detection()`: Detect outliers using various methods
- `generate_report()`: Create comprehensive exploration reports

### BasicStatistics

The `BasicStatistics` class provides statistical analysis tools:

- `descriptive_statistics()`: Comprehensive descriptive statistics
- `normality_tests()`: Test distributions for normality
- `correlation_statistics()`: Statistical correlation analysis
- `categorical_statistics()`: Analyze categorical variables
- `hypothesis_tests()`: Perform hypothesis tests between features and targets
- `outlier_statistics()`: Statistical outlier detection and analysis
- `generate_statistics_report()`: Create detailed statistical reports

## Data Requirements

The tools work with pandas DataFrames and automatically handle:
- **Numeric columns**: Integer and float data types
- **Categorical columns**: String and category data types
- **Datetime columns**: Datetime data types
- **Missing values**: NaN and null values
- **Mixed data types**: Automatic type detection and appropriate analysis

## Example Output

The analysis provides insights such as:
- Dataset characteristics (shape: 1000 × 10, memory usage: 0.08 MB)
- Missing value patterns and percentages
- Statistical distributions and normality tests
- Correlation strengths and multicollinearity warnings
- Significant feature relationships for machine learning
- Outlier detection and data quality assessment

## Machine Learning Preparation

The exploration results help with:
1. **Feature Selection**: Identify significant predictors
2. **Data Preprocessing**: Handle missing values and outliers
3. **Feature Engineering**: Understand relationships between variables
4. **Model Selection**: Choose appropriate algorithms based on data characteristics
5. **Validation Strategy**: Design appropriate cross-validation based on data structure

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Static plotting
- `seaborn`: Statistical visualization
- `scipy`: Scientific computing and statistics
- `scikit-learn`: Machine learning utilities
- `plotly`: Interactive visualizations
- `jupyter`: Notebook environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] Time series analysis for climbing route trends
- [ ] Advanced feature engineering tools
- [ ] Automated machine learning pipeline integration
- [ ] Real-time data streaming capabilities
- [ ] Web-based dashboard for interactive exploration