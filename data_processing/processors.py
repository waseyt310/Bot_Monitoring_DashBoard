"""
Data processing module for Bot Monitoring Dashboard
Contains functions to process and transform data for display
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re
import gc
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union
from data_processing.validators import validate_processed_data, validate_matrix_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_processor')

# Constants
STATUS_PRIORITY = {
    "Failed": 100,      # Highest priority
    "Error": 100,
    "TimedOut": 100,
    "Running": 80,      # Medium priority 
    "InProgress": 80,
    "Started": 80,
    "Succeeded": 60,    # Success statuses
    "Completed": 60,
    "Done": 60,
    "Skipped": 40,      # Less important statuses
    "Canceled": 30,
    "Suspended": 20,
    "Paused": 20,
    "No Run": 0         # Lowest priority
}

# Regex patterns (compiled for performance)
CAMEL_CASE_PATTERN = re.compile(r'^([A-Z][a-z]+)')
ALPHA_SEQUENCE_PATTERN = re.compile(r'[A-Za-z]{3,}')
SPLIT_PATTERN = re.compile(r'[_\s-]')

# Common project identifiers
COMMON_IDENTIFIERS = frozenset(["AMZ", "AWS", "C2D", "AZ", "WF", "PS", "VP", "BI"])

# Required columns for different operations
MATRIX_COLUMNS = {'display_name', 'automation_project', 'taskstatus', 'hour'}
PROCESS_COLUMNS = {'datetimestarted', 'flowname', 'taskstatus', 'flowowner', 'wassuccessful', 'triggertype'}

@lru_cache(maxsize=1000)
def extract_project_name(flow_name: str) -> str:
    """
    Extract project name from flow name using pattern matching.
    
    Args:
        flow_name: Flow name to extract project from
        
    Returns:
        Extracted project name or 'Unknown' if not found
    """

    try:
        if not isinstance(flow_name, str) or not flow_name.strip():
            return 'Unknown'
            
        flow_name = flow_name.strip()
        
        # Pattern 1: Text before hyphen
        if ' - ' in flow_name:
            return flow_name.split(' - ')[0].strip()
            
        # Pattern 2: Text before underscore
        if '_' in flow_name:
            return flow_name.split('_')[0].strip()
            
        # Pattern 3: First CamelCase word
        camel_match = CAMEL_CASE_PATTERN.match(flow_name)
        if camel_match:
            return camel_match.group(1)
            
        # Pattern 4: First word if capitalized
        words = flow_name.split()
        if words and len(words[0]) > 2 and words[0][0].isupper():
            return words[0]
            
        # Pattern 5: Common project identifiers
        flow_upper = flow_name.upper()
        for identifier in COMMON_IDENTIFIERS:
            if identifier in flow_upper:
                parts = SPLIT_PATTERN.split(flow_name)
                for part in parts:
                    if identifier in part.upper():
                        return part
        
        # Pattern 6: First alphabetic sequence
        alpha_match = ALPHA_SEQUENCE_PATTERN.search(flow_name)
        if alpha_match:
            return alpha_match.group(0)
            
        return 'Unknown'
        
    except Exception as e:
        logger.error(f"Error extracting project from {flow_name}: {e}")
        return 'Unknown'

def process_data_for_dashboard(df: pd.DataFrame, day_filter: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
    """
    Process data with dynamic project mapping and optimized performance.
    
    Args:
        df (pd.DataFrame): Raw DataFrame with bot data
        day_filter (Optional[Union[str, datetime]]): Optional date filter
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for dashboard display
    """

    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to process_data_for_dashboard")
            return pd.DataFrame()
        
        # Create copy with needed columns
        processed_df = df.copy()
        
        # Ensure all needed columns exist
        for col in PROCESS_COLUMNS:
            if col not in processed_df.columns:
                if col == 'wassuccessful':
                    processed_df[col] = np.where(
                        processed_df['taskstatus'] == 'Succeeded', 1, 0
                    ) if 'taskstatus' in processed_df.columns else 0
                elif col == 'triggertype':
                    processed_df[col] = 'unknown'
                else:
                    processed_df[col] = None
        
        # Filter by day if specified
        if day_filter:
            try:
                filter_date = pd.to_datetime(day_filter).date()
                logger.info(f"Applying day filter for date: {filter_date}")
                
                # Ensure datetimestarted is datetime type
                processed_df['datetimestarted'] = pd.to_datetime(processed_df['datetimestarted'], errors='coerce')
                
                # Handle NaT values
                if processed_df['datetimestarted'].isna().any():
                    logger.warning(f"Found {processed_df['datetimestarted'].isna().sum()} rows with invalid datetime values")
                    processed_df = processed_df[~processed_df['datetimestarted'].isna()]
                
                # Apply date filter
                processed_df = processed_df[processed_df['datetimestarted'].dt.date == filter_date]
                logger.info(f"After date filtering: {len(processed_df)} records")
            except Exception as e:
                logger.error(f"Error during date filtering: {e}")
                # Continue with unfiltered data
        
        # Add derived columns
        processed_df['hour'] = pd.to_datetime(processed_df['datetimestarted']).dt.hour
        processed_df['owner'] = processed_df['flowowner'].str.replace(' serviceaccount', '').str.title()
        processed_df['automation_project'] = processed_df['flowname'].apply(extract_project_name)
        
        # Add trigger type grouping
        if 'triggertype' in processed_df.columns:
            conditions = [
                processed_df['triggertype'] == 'manual',
                processed_df['triggertype'] == 'Recurrence'
            ]
            choices = ['Manual', 'Recurrence']
            processed_df['trigger_group'] = np.select(conditions, choices, default='OtherTrigger')
        
        # Create display name
        processed_df['display_name'] = (
            processed_df['owner'] + ' | ' + 
            processed_df['automation_project'] + ' | ' + 
            processed_df['flowname']
        )
        
        # Add success rate calculation
        if 'wassuccessful' in processed_df.columns:
            processed_df['success_rate'] = processed_df.groupby('flowname')['wassuccessful'].transform('mean') * 100
        
        # Cleanup to free memory
        gc.collect()
        
        logger.info(f"Data processing completed with {len(processed_df)} records")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in process_data_for_dashboard: {e}")
        return pd.DataFrame()

def create_hourly_matrix(
    df: pd.DataFrame, 
    selected_project: str = 'All Projects', 
    selected_status: str = 'All Statuses', 
    max_rows: int = 300
) -> Tuple[Dict[str, Dict[int, str]], List[str], List[int]]:
    """
    Create hourly matrix for dashboard display.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with bot data
        selected_project (str): Project filter (or 'All Projects')
        selected_status (str): Status filter (or 'All Statuses')
        max_rows (int): Maximum number of rows to display
    
    Returns:
        tuple: A tuple containing:
            - bot_hour_status (Dict[str, Dict[int, str]]): Dictionary of display_name to hour to status
            - display_names (List[str]): List of display names to show
            - hours (List[int]): List of hours (0-23)
    """

    try:
        logger.info("Creating hourly matrix")
        
        # Generate list of all hours - constant regardless of data
        hours = list(range(24))  # 0-23 hours
        
        # Handle empty dataframe early
        if df is None or df.empty:
            logger.warning("No data available for matrix creation")
            return {}, [], hours
        
        # Validate processed data
        is_valid, message, validated_df = validate_processed_data(df)
        if not is_valid:
            logger.warning(f"Processed data validation warning: {message}")
            if validated_df is not None:
                df = validated_df
            else:
                return {}, [], hours
        
        # Optimize memory usage by selecting only needed columns
        try:
            # Extract only needed columns to reduce memory footprint
            matrix_df = df[list(MATRIX_COLUMNS)].copy()
        except KeyError as e:
            logger.error(f"Missing required columns for matrix creation: {e}")
            return {}, [], hours
        
        # Apply filters with vectorized operations for performance
        filter_mask = pd.Series(True, index=matrix_df.index)
        orig_count = len(matrix_df)
        
        # Apply project filter if specified (vectorized operation)
        if selected_project != 'All Projects':
            project_mask = (matrix_df['automation_project'] == selected_project)
            filter_mask &= project_mask
            logger.info(f"Project filter '{selected_project}' matched {project_mask.sum()} of {orig_count} records")
            
        # Apply status filter if specified (vectorized operation)
        if selected_status != 'All Statuses':
            status_mask = (matrix_df['taskstatus'] == selected_status)
            filter_mask &= status_mask
            logger.info(f"Status filter '{selected_status}' matched {status_mask.sum()} of {orig_count} records")
        
        # Apply combined filter in one operation (more efficient)
        filtered_df = matrix_df.loc[filter_mask]
        logger.info(f"Filtered from {len(matrix_df)} to {len(filtered_df)} records")
        
        # Clean up intermediate objects to free memory
        del matrix_df
        gc.collect()
        
        # Check if we have data after filtering
        if filtered_df.empty:
            logger.warning("No data after filtering")
            return {}, [], hours
            
        # Validate that display_name column has correct format
        invalid_display_names = filtered_df['display_name'].isna() | (filtered_df['display_name'] == '')
        if invalid_display_names.any():
            logger.warning(f"Found {invalid_display_names.sum()} rows with invalid display names")
            filtered_df = filtered_df[~invalid_display_names]
            
            # If all display names were invalid, try to recreate them
            if filtered_df.empty:
                logger.warning("All display names were invalid, attempting to recreate")
                try:
                    # Get the original filtered dataframe again
                    filtered_df = matrix_df.loc[filter_mask].copy()
                    
                    # Recreate display names
                    filtered_df['owner'] = filtered_df.get('owner', 'Unknown')
                    filtered_df['automation_project'] = filtered_df.get('automation_project', 'Unknown')
                    filtered_df['flowname'] = filtered_df.get('flowname', 'Unknown')
                    
                    filtered_df['display_name'] = (
                        filtered_df['owner'].fillna('Unknown') + ' | ' + 
                        filtered_df['automation_project'].fillna('Unknown') + ' | ' + 
                        filtered_df['flowname'].fillna('Unknown')
                    )
                    
                    # Filter out any remaining invalid display names
                    filtered_df = filtered_df[filtered_df['display_name'] != 'Unknown | Unknown | Unknown']
                    
                    if filtered_df.empty:
                        logger.warning("Still no valid data after display name repair")
                        return {}, [], hours
                except Exception as e:
                    logger.error(f"Error recreating display names: {e}")
                    return {}, [], hours
            
        # Get display names with optimized approach
        # Use np.array for memory efficiency over list
        display_names_array = filtered_df['display_name'].unique()
        
        # Smart selection of rows based on activity if too many rows
        if len(display_names_array) > max_rows:
            logger.warning(f"Too many display names ({len(display_names_array)}), using intelligent selection")
            
            # Use a smarter approach to select the most interesting bots
            # Count statuses by display_name, prioritizing failures and actives
            status_counts = (
                filtered_df.groupby('display_name')['taskstatus']
                .apply(lambda x: pd.Series({
                    'failed': (x == 'Failed').sum(),
                    'running': (x == 'Running').sum(),
                    'total': len(x)
                }))
            )
            
            # Create score based on status counts (prioritize failures, then running, then total)
            status_counts['score'] = (
                status_counts['failed'] * 100 + 
                status_counts['running'] * 10 + 
                status_counts['total']
            )
            
            # Select top bots by score
            selected_names = status_counts.sort_values('score', ascending=False).head(max_rows).index.tolist()
            display_names = selected_names
            filtered_df = filtered_df[filtered_df['display_name'].isin(display_names)]
        else:
            display_names = display_names_array.tolist()
        
        # More efficient matrix creation using dict comprehension
        bot_hour_status = {name: {hour: "No Run" for hour in hours} for name in display_names}
        
        try:
            # Use optimized groupby approach
            # Group once by both dimensions for better performance
            grouped = filtered_df.groupby(['display_name', 'hour'])
            
            # Process groups for matrix creation
            for (name, hour), group in grouped:
                if name in bot_hour_status and 0 <= hour <= 23:
                    # Find highest priority status in this group efficiently
                    # Use numpy operations for performance
                    if len(group) > 0:
                        status_values = group['taskstatus'].values
                        
                        # Find the status with highest priority using max with key function
                        highest_status = max(
                            np.unique(status_values),
                            key=lambda x: STATUS_PRIORITY.get(x, 0)
                        )
                        
                        bot_hour_status[name][hour] = highest_status
            
        except Exception as e:
            logger.error(f"Error creating status matrix: {e}")
            # Continue with what we have so far
        
        # Debug logs for matrix data
        logger.info(f"Matrix pre-validation: {len(bot_hour_status)} bots, {len(display_names)} display names")
        if not display_names:
            logger.warning("Empty display_names list before validation")
            # Try to recover by using bot_hour_status keys
            if bot_hour_status:
                display_names = list(bot_hour_status.keys())
                logger.info(f"Recovered {len(display_names)} display names from bot_hour_status keys")
        
        # Validate the matrix data before returning
        is_valid, message, validated_data = validate_matrix_data(bot_hour_status, display_names, hours)
        if not is_valid:
            logger.warning(f"Matrix validation warning: {message}")
            # Always attempt to return usable data even with validation issues
            if validated_data and isinstance(validated_data, tuple) and len(validated_data) == 3:
                logger.info(f"Using validated data with {len(validated_data[1])} display names")
                return validated_data
            else:
                logger.warning("Falling back to pre-validation data")
                return bot_hour_status, display_names, hours
        
        logger.info(f"Matrix creation completed with {len(display_names)} rows")
        return validated_data
        
    except Exception as e:
        logger.error(f"Error creating hourly matrix: {e}")
        return {}, [], hours

