"""
Data validation module for the Bot Monitoring Dashboard
Contains functions to validate data at different stages of processing
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_validator')

def validate_raw_data(df):
    """
    Validate raw data from database or CSV
    
    Args:
        df (pd.DataFrame): Raw data to validate
        
    Returns:
        tuple: (is_valid, error_message, validated_df)
    """
    try:
        if df is None or df.empty:
            return False, "No data available", None
            
        # Required columns
        required_columns = [
            'flowname', 'flowowner', 'datetimestarted',
            'taskstatus'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", None
            
        # Create a copy for validation
        validated_df = df.copy()
        
        # Validate and convert datetime
        if 'datetimestarted' in validated_df.columns:
            validated_df['datetimestarted'] = pd.to_datetime(
                validated_df['datetimestarted'],
                errors='coerce'
            )
            
            # Remove rows with invalid dates
            invalid_dates = validated_df['datetimestarted'].isna()
            if invalid_dates.any():
                logger.warning(f"Removed {invalid_dates.sum()} rows with invalid dates")
                validated_df = validated_df[~invalid_dates]
        
        # Validate status values and fill missing values
        if 'taskstatus' in validated_df.columns:
            validated_df['taskstatus'] = validated_df['taskstatus'].fillna('No Run')
            
        # Add wassuccessful column if not present
        if 'wassuccessful' not in validated_df.columns and 'taskstatus' in validated_df.columns:
            validated_df['wassuccessful'] = np.where(
                validated_df['taskstatus'].str.lower().isin(['succeeded', 'completed']),
                1, 0
            )
            
        # Validate flowowner (not empty)
        if 'flowowner' in validated_df.columns:
            validated_df['flowowner'] = validated_df['flowowner'].fillna('Unknown')
        
        # Validate triggertype
        if 'triggertype' in validated_df.columns:
            validated_df['triggertype'] = validated_df['triggertype'].fillna('unknown')
        else:
            validated_df['triggertype'] = 'unknown'
        
        return True, "Validation successful", validated_df
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False, f"Validation error: {str(e)}", None

def validate_processed_data(df):
    """
    Validate processed data before dashboard display
    
    Args:
        df (pd.DataFrame): Processed data to validate
        
    Returns:
        tuple: (is_valid, error_message, validated_df)
    """
    try:
        if df is None or df.empty:
            return False, "No processed data available", None
            
        # Create a copy for validation
        validated_df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['owner', 'automation_project', 'flowname', 'taskstatus', 'datetimestarted']
        for column in required_columns:
            if column not in validated_df.columns:
                if column == 'owner' and 'flowowner' in validated_df.columns:
                    validated_df['owner'] = validated_df['flowowner']
                elif column == 'automation_project':
                    validated_df['automation_project'] = 'Unknown'
                else:
                    validated_df[column] = 'Unknown'
                    
        # Ensure hour column exists
        if 'hour' not in validated_df.columns and 'datetimestarted' in validated_df.columns:
            validated_df['hour'] = pd.to_datetime(validated_df['datetimestarted']).dt.hour
        
        # Ensure display_name exists
        if 'display_name' not in validated_df.columns:
            validated_df['display_name'] = (
                validated_df['owner'] + ' | ' + 
                validated_df['automation_project'] + ' | ' + 
                validated_df['flowname']
            )
            
        # Validate success rate calculation
        if 'wassuccessful' in validated_df.columns and 'success_rate' not in validated_df.columns:
            try:
                validated_df['success_rate'] = validated_df.groupby('flowname')['wassuccessful'].transform('mean') * 100
            except Exception as e:
                logger.warning(f"Could not calculate success rate: {e}")
                validated_df['success_rate'] = 0
                
        return True, "Validation successful", validated_df
        
    except Exception as e:
        logger.error(f"Error validating processed data: {e}")
        return False, f"Validation error: {str(e)}", None

def validate_matrix_data(bot_hour_status, display_names, hours):
    """
    Validate matrix data before display
    
    Args:
        bot_hour_status (dict): Dictionary of display_name to hour to status
        display_names (list): List of display names to show
        hours (list): List of hours (0-23)
        
    Returns:
        tuple: (is_valid, error_message, validated_data)
    """
    try:
        if not isinstance(bot_hour_status, dict):
            logger.warning("Invalid bot_hour_status type - creating empty dictionary")
            return False, "Invalid bot_hour_status type", (dict(), [], list(range(24)))
            
        if not isinstance(display_names, list):
            logger.warning(f"Invalid display_names type: {type(display_names)}")
            display_names = []
            
        # Handle empty display_names by attempting to extract them from bot_hour_status
        if not display_names and bot_hour_status:
            logger.info("No display names provided but bot_hour_status exists - extracting keys")
            display_names = list(bot_hour_status.keys())
            if display_names:
                logger.info(f"Extracted {len(display_names)} display names from bot_hour_status")
            
        if not display_names:
            logger.warning("Empty display_names after extraction attempts")
            return False, "Invalid or empty display_names", (bot_hour_status, [], list(range(24)))
            
        if not isinstance(hours, list) or not hours:
            logger.info("Invalid hours - defaulting to 0-23")
            hours = list(range(24))
            
        # Ensure hours are within valid range
        valid_hours = [h for h in hours if isinstance(h, int) and 0 <= h <= 23]
        if len(valid_hours) != len(hours):
            logger.warning(f"Filtered {len(hours) - len(valid_hours)} invalid hours")
            hours = valid_hours if valid_hours else list(range(24))
            
        # Validate bot_hour_status structure
        valid_bot_hour_status = {}
        valid_display_names = []
        
        for name in display_names:
            # Skip invalid names
            if not isinstance(name, str) or not name:
                continue
                
            # Get or create hour status dictionary for this name
            hour_status = bot_hour_status.get(name, {})
            
            # Create a valid hour status dictionary
            valid_hour_status = {}
            for hour in hours:
                # Get or default the status
                status = hour_status.get(hour, "No Run")
                if not isinstance(status, str) or not status:
                    status = "No Run"
                valid_hour_status[hour] = status
                
            # Add to valid dictionaries
            valid_bot_hour_status[name] = valid_hour_status
            valid_display_names.append(name)
            
        if not valid_display_names:
            logger.warning("No valid display names found")
            return False, "No valid display names", (dict(), [], hours)
            
        return True, "Validation successful", (valid_bot_hour_status, valid_display_names, hours)
        
    except Exception as e:
        logger.error(f"Error validating matrix data: {e}")
        return False, f"Validation error: {str(e)}", (dict(), [], list(range(24)))

