"""
Data processing module for Bot Monitoring Dashboard
Contains functions to process and validate data
"""

from data_processing.processors import process_data_for_dashboard, extract_project_name, create_hourly_matrix
from data_processing.validators import validate_raw_data, validate_processed_data, validate_matrix_data

