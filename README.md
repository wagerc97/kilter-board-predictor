# Kilter Board Predictor

A comprehensive data exploration and analysis pipeline for climbing route data from Kilter boards.

## 🚀 Features

- **SQLite Database Integration**: Seamless connection and querying of climbing route databases
- **Comprehensive Data Exploration**: Complete pipeline for analyzing holds, routes, and user attempts
- **Interactive Visualizations**: Dynamic charts and plots using Plotly and Matplotlib
- **Statistical Analysis**: Correlation analysis, performance metrics, and trend identification
- **Data Quality Assessment**: Automated checks for missing values, duplicates, and data consistency
- **Export Capabilities**: Export processed data and analysis results to CSV and JSON formats
- **Interactive Jupyter Notebook**: User-friendly interface for data exploration

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/wagerc97/kilter-board-predictor.git
cd kilter-board-predictor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a sample database (optional, for testing):
```bash
python create_sample_db.py
```

## 🚀 Quick Start

### Using the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `data_exploration_pipeline.ipynb` in your browser

3. Run all cells to execute the complete data exploration pipeline

### Using Your Own Database

To use your own SQLite database:

1. Update the `DATABASE_PATH` variable in the notebook to point to your database file
2. Modify the SQL queries in the notebook to match your database schema
3. Run the notebook cells to explore your data

## 📊 What the Pipeline Includes

### 1. Database Connection & Schema Exploration
- Establishes secure SQLite connections
- Automatically discovers and documents database schema
- Provides table structure and relationship insights

### 2. Data Quality Assessment
- Identifies missing values and duplicates
- Analyzes data types and memory usage
- Generates comprehensive quality reports

### 3. Exploratory Data Analysis
- Descriptive statistics for all numerical columns
- Categorical data distribution analysis
- Advanced SQL queries for business insights

### 4. Statistical Analysis
- Correlation analysis between variables
- User performance metrics and rankings
- Success rate analysis by difficulty grades

### 5. Visualization Pipeline
- Hold position mapping on climbing boards
- Success rate heatmaps by user and grade
- Time series analysis of climbing activity
- Interactive dashboards with filtering capabilities

### 6. Export & Reporting
- Export processed data to CSV files
- Generate comprehensive JSON reports
- Create summary statistics and insights

## 📁 Project Structure

```
kilter-board-predictor/
│
├── data_exploration_pipeline.ipynb    # Main Jupyter notebook
├── create_sample_db.py                # Sample database generator
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── kilter_board_data.db              # Sample SQLite database (created after running script)
└── exports/                          # Generated reports and data exports
    ├── analysis_report.json
    ├── holds_data.csv
    ├── routes_data.csv
    ├── attempts_data.csv
    └── ...
```

## 🎯 Sample Database Schema

The sample database includes the following tables:

- **holds**: Physical holds on the climbing board (position, type, difficulty)
- **routes**: Climbing routes (name, grade, setter, description)
- **route_holds**: Relationship between routes and holds (role: start, hand, foot, finish)
- **user_attempts**: User climbing attempts (result, rating, attempts count)

## 📈 Key Insights Generated

- **Route Difficulty Analysis**: Average difficulty by route setter
- **Success Rate Metrics**: Performance analysis by grade level
- **Popular Routes**: Most attempted routes with user ratings
- **Hold Usage Statistics**: Usage patterns by hold type
- **User Performance Rankings**: Success rates and climbing patterns
- **Temporal Analysis**: Activity trends over time

## 🔧 Customization

### Adding New Analysis

To add new analysis sections:

1. Create new cells in the Jupyter notebook
2. Use the `execute_query()` function to run custom SQL queries
3. Leverage pandas and plotly for data manipulation and visualization

### Modifying Database Schema

To work with different database schemas:

1. Update the table names in SQL queries
2. Modify column references to match your schema
3. Adjust data types and analysis methods as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include your Python version and any error messages

## 🙏 Acknowledgments

- Built for the climbing community to analyze route data
- Inspired by data-driven approaches to climbing performance
- Uses modern Python data science stack for robust analysis