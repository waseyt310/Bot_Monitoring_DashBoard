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
import json
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
    "Cancelled": 30,    # Matching actual status in data
    "Canceled": 30,     # Alternative spelling
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

def process_data_for_dashboard(df):
    """Process data for dashboard display with enhanced flow mapping"""
    try:
        if df is None or df.empty:
            logger.warning("Empty dataframe passed to process_data_for_dashboard")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Load flow mapping
        try:
            with open('flow_mapping.json', 'r') as f:
                flow_mapping = json.load(f)
            logger.info(f"Loaded {len(flow_mapping)} flow mappings")
        except Exception as e:
            logger.warning(f"Could not load flow mapping, using fallback: {e}")
            flow_mapping = {}
        
        # Function to get project name from mapping
        def get_project_name(flow_name):
            if pd.isna(flow_name):
                return 'Other Cloud Flow'
            flow_key = str(flow_name).strip()
            return flow_mapping.get(flow_key, {}).get('project', 'Other Cloud Flow')
        
        # Apply flow mapping
        processed_df['automation_project'] = processed_df['flowname'].apply(get_project_name)
        
        # Ensure status values match our priority dictionary
        processed_df['taskstatus'] = processed_df['taskstatus'].map(
            lambda x: x if x in STATUS_PRIORITY else 'No Run'
        )
        
        # Ensure datetime columns are in proper format
        processed_df['datetimestarted'] = pd.to_datetime(processed_df['datetimestarted'])
        if 'datetimecompleted' in processed_df.columns:
            processed_df['datetimecompleted'] = pd.to_datetime(processed_df['datetimecompleted'])
        
        # Add duration if both start and end times exist
        if 'datetimecompleted' in processed_df.columns:
            pass  # Placeholder for duration calculation
        
        # Add derived columns
        processed_df['hour'] = pd.to_datetime(processed_df['datetimestarted']).dt.hour
        processed_df['owner'] = processed_df['flowowner'].str.replace(' serviceaccount', '').str.title()
            
        # Create display name for matrix - combining owner, project and flow
        processed_df['display_name'] = processed_df.apply(
            lambda row: f"{row['owner']} | {row['automation_project']} | {row['flowname']}", 
            axis=1
        )
        
        # Add trigger type grouping
        if 'triggertype' in processed_df.columns:
            conditions = [
                processed_df['triggertype'] == 'manual',
                processed_df['triggertype'] == 'Recurrence'
            ]
            choices = ['Manual', 'Recurrence']
            processed_df['trigger_group'] = np.select(conditions, choices, default='OtherTrigger')
        
        # Ensure boolean columns are properly typed
        if 'wassuccessful' in processed_df.columns:
            processed_df['wassuccessful'] = pd.to_numeric(processed_df['wassuccessful'], errors='coerce').fillna(0)
            
        # Calculate success rate
        # Calculate success rate
        processed_df['success_rate'] = processed_df['wassuccessful'] * 100
        
        # Add status priority for sorting
        processed_df['status_priority'] = processed_df['taskstatus'].map(STATUS_PRIORITY).fillna(0)
        
        # Log processing results
        logger.info(f"Processed {len(processed_df)} records")
        logger.info(f"Unique projects: {processed_df['automation_project'].nunique()}")
        logger.info(f"Unique display names: {processed_df['display_name'].unique().size} bots")
        
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

        # Create hour column if not exists
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['datetimestarted']).dt.hour

        # Check required columns
        required_columns = {'display_name', 'automation_project', 'taskstatus', 'hour', 'datetimestarted'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns for matrix creation: {missing_columns}")
            return {}, [], hours
            
        # Apply filters
        mask = pd.Series(True, index=df.index)
        orig_count = len(df)
        
        if selected_project != 'All Projects':
            mask &= (df['automation_project'] == selected_project)
            logger.info(f"Project filter applied: {selected_project}")
            
        if selected_status != 'All Statuses':
            mask &= (df['taskstatus'] == selected_status)
            logger.info(f"Status filter applied: {selected_status}")
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered from {orig_count} to {len(filtered_df)} records")
        
        # Check if we have data after filtering
        if filtered_df.empty:
            logger.warning("No data after filtering")
            return {}, [], hours
        
        # Smart selection of display names
        display_names = (filtered_df.groupby('display_name')
            .agg({
                'taskstatus': lambda x: (x == 'Failed').sum() * 100 + 
                                      (x == 'Running').sum() * 10 + 
                                      len(x),
                'datetimestarted': 'max'
            })
            .sort_values(['taskstatus', 'datetimestarted'], ascending=[False, False])
            .head(max_rows)
            .index.tolist())

        # Create matrix dictionary
        bot_hour_status = {name: {hour: "No Run" for hour in hours} for name in display_names}
        
        # Fill matrix efficiently
        status_data = (filtered_df[filtered_df['display_name'].isin(display_names)]
            .groupby(['display_name', 'hour'])['taskstatus']
            .agg(lambda x: max(x, key=lambda s: STATUS_PRIORITY.get(s, 0)))
            .to_dict())
            
        for (name, hour), status in status_data.items():
            if name in bot_hour_status and 0 <= hour <= 23:
                bot_hour_status[name][hour] = status

        try:
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
            logger.error(f"Error in matrix validation: {e}")
            return bot_hour_status, display_names, hours
            
    except Exception as e:
        logger.error(f"Error creating hourly matrix: {e}")
        return {}, [], hours

