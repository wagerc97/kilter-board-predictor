"""
Configuration file for the Kilter Board Data Exploration Pipeline
"""

# Database configuration
DATABASE_CONFIG = {
    'default_path': 'kilter_board_data.db',
    'connection_timeout': 30,
    'max_rows_display': 100
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'color_palette': 'husl',
    'plot_style': 'seaborn-v0_8',
    'dpi': 100
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'min_attempts_for_user_stats': 10,
    'confidence_level': 0.95,
    'correlation_threshold': 0.3
}

# Export configuration
EXPORT_CONFIG = {
    'output_directory': 'exports',
    'csv_format': {
        'index': False,
        'encoding': 'utf-8'
    },
    'json_format': {
        'indent': 2,
        'ensure_ascii': False
    }
}

# Grade mappings for climbing routes
GRADE_MAPPINGS = {
    'V0': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 
    'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10,
    'V11': 11, 'V12': 12, 'V13': 13, 'V14': 14, 'V15': 15, 'V16': 16, 'V17': 17
}

# Hold type categories
HOLD_TYPES = [
    'jug', 'crimp', 'pinch', 'sloper', 'pocket', 
    'edge', 'volume', 'undercling', 'sidepull'
]

# Attempt result categories
ATTEMPT_RESULTS = [
    'flash',      # Completed on first try
    'send',       # Completed after multiple tries
    'attempt',    # Attempted but not completed
    'fail'        # Failed attempt
]