# -*- coding: utf-8 -*-
"""
Bot Monitoring Dashboard with Streamlit
Features:
- Emoji status indicators for better visibility
- Matrix view with hours as columns and bots as rows
- Filtering by project and status
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
import time
import logging
import gc
import traceback
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Any
from data_processing.processors import process_data_for_dashboard, create_hourly_matrix
from secure_db_connection import get_flow_data, test_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('bot_monitor_dashboard')

# Set page config at the very beginning
try:
    st.set_page_config(
        page_title="Bot Monitoring Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.warning(f"Page config warning: {e}")
    pass

# Status emojis for better visibility
STATUS_EMOJIS = {
    "Succeeded": "🟢",  # Green circle for success
    "Failed": "🔴",     # Red circle for failure
    "Running": "🟡",    # Yellow circle for running
    "No Run": "⚪",     # White circle for no run
    "Completed": "🟢",  # Same as Succeeded
    "Canceled": "⚫",   # Black circle for canceled
    "Suspended": "🔵",  # Blue circle for suspended
    "Skipped": "⚪",    # White circle for skipped (like No Run)
    "Error": "🔴",      # Same as Failed
    "TimedOut": "🔴"    # Same as Failed
}

# Function to get emoji for status with case-insensitive matching
def get_status_emoji(status):
    """Get emoji for a status value with fallback and case-insensitive matching"""
    if not status:
        return STATUS_EMOJIS.get("No Run", "⚪")
        
    # Direct match first
    if status in STATUS_EMOJIS:
        return STATUS_EMOJIS[status]
        
    # Case-insensitive match
    status_lower = status.lower()
    for key, emoji in STATUS_EMOJIS.items():
        if key.lower() == status_lower:
            return emoji
            
    # Default fallback
    return STATUS_EMOJIS.get("No Run", "⚪")

def safe_dashboard_reload() -> None:
    """
    Safely reload the dashboard data with proper error handling
    
    This function:
    1. Implements debounce to prevent rapid successive reloads
    2. Shows a loading spinner during reload
    3. Updates the refresh timestamp
    4. Attempts to reload using the best available method
    5. Provides appropriate fallbacks if the preferred method fails
    
    Fallback Methods (in order):
    1. st.rerun() - Modern Streamlit API (v1.18.0+)
    2. st.experimental_rerun() - Legacy Streamlit API
    3. JavaScript window.location.reload() - Browser-level fallback
    
    Returns:
        None
    """
    # Implement debounce to prevent rapid successive reloads
    current_time = time.time()
    try:
        # Check if a refresh is already in progress
        if st.session_state.get('refresh_in_progress', False):
            logger.info("Refresh already in progress, skipping")
            return
            
        # Check if we're within the debounce period
        last_refresh_time = getattr(st.session_state, 'last_refresh_time', 0)
        debounce_time = getattr(st.session_state, 'refresh_debounce_time', 2.0)
        
        if current_time - last_refresh_time < debounce_time:
            logger.info(f"Refresh requested too soon (within {debounce_time}s), skipping")
            return
            
        # Set refresh in progress flag
        st.session_state.refresh_in_progress = True
        st.session_state.last_refresh_time = current_time
        
        # Show loading state
        with st.spinner("Reloading dashboard data..."):
            # Update refresh timestamp
            try:
                if 'last_refresh' in st.session_state:
                    st.session_state.last_refresh = datetime.now()
            except Exception as ts_error:
                logger.warning(f"Failed to update refresh timestamp: {ts_error}")
            
            # Try modern rerun method first (Streamlit >= 1.18.0)
            try:
                if hasattr(st, 'rerun'):
                    logger.info("Using modern st.rerun() method")
                    # Cleanup before rerun
                    st.session_state.refresh_in_progress = False
                    st.rerun()
                    return  # Exit if successful
                else:
                    # Fall back to older experimental API
                    logger.info("Falling back to st.experimental_rerun() method")
                    # Cleanup before rerun
                    st.session_state.refresh_in_progress = False
                    st.experimental_rerun()
                    return  # Exit if successful
            except RuntimeError as rt_error:
                logger.warning(f"Runtime error during rerun: {rt_error}")
            except ImportError as imp_error:
                logger.warning(f"Import error during rerun (possible version mismatch): {imp_error}")
            except Exception as rerun_error:
                logger.warning(f"Streamlit rerun methods failed: {rerun_error}")
            
            # JavaScript fallback as last resort
            try:
                logger.info("Attempting JavaScript fallback refresh")
                html = """
                <script>
                    setTimeout(function() {
                        window.location.reload();
                    }, 2000);
                </script>
                """
                st.components.v1.html(html)
                st.info("Page will refresh momentarily...")
            except Exception as js_error:
                logger.error(f"JavaScript refresh failed: {js_error}")
                st.warning("Unable to auto-refresh. Please refresh the page manually.")
    except ValueError as val_error:
        logger.error(f"Value error during dashboard reload: {val_error}")
        st.error("A data validation error occurred. Please refresh manually.")
    except TypeError as type_error:
        logger.error(f"Type error during dashboard reload: {type_error}")
        st.error("A type error occurred. Please refresh manually.")
    except Exception as e:
        logger.error(f"Error during dashboard reload: {e}", exc_info=True)
        st.error("Failed to reload. Please refresh manually.")
    finally:
        # Always reset the refresh in progress flag
        try:
            st.session_state.refresh_in_progress = False
        except:
            pass
        
def display_matrix(bot_hour_status, display_names, hours, enable_grouping=True):
    """
    Display the matrix as a styled table in Streamlit
    
    Parameters:
    - bot_hour_status: Dictionary mapping display_name to a dictionary mapping hour to status
                      Format: {display_name: {hour: status}}
    - display_names: List of display names (flow identifiers) to show in the matrix
    - hours: List of hours (0-23) to display as columns in the matrix
    - enable_grouping: Whether to enable project grouping for visual organization (default: True)
    
    Returns:
        None - Displays the matrix directly in the Streamlit interface
    """
    try:
        # Handle empty data
        if not display_names:
            st.warning("No data available to display in matrix. Try adjusting filters.")
            return

        # Create header row with hour labels
        header_row = ["Owner", "Automation Project", "Cloud Flow"] + [f"{hour:02d}:00" for hour in hours]
        
        # Create the data rows with emojis - using list comprehension for better performance
        data_rows = []
        for display_name in sorted(display_names):
            try:
                # Split the display name into its components
                name_parts = display_name.split(" | ", 2)
                # Handle case where display name doesn't have expected format
                if len(name_parts) >= 3:
                    owner, project, flow = name_parts
                elif len(name_parts) == 2:
                    owner, project = name_parts
                    flow = "Unknown"
                else:
                    owner = name_parts[0]
                    project = "Unknown"
                    flow = "Unknown"
                
                # Create row with base columns
                row = [owner, project, flow]
                
                # Add emoji status for each hour
                for hour in hours:
                    status = bot_hour_status[display_name].get(hour, "No Run")
                    emoji = get_status_emoji(status)
                    row.append(emoji)
                
                data_rows.append(row)
            except Exception as row_error:
                logger.error(f"Error processing row {display_name}: {row_error}")
                continue
        
        # Create a DataFrame for easy display
        matrix_df = pd.DataFrame(data_rows, columns=header_row)
        
        # Add hour column configs dynamically with tooltips
        hour_column_config = {
            f"{hour:02d}:00": st.column_config.TextColumn(
                f"{hour:02d}:00",
                width="small",
                help=f"Status at {hour:02d}:00 hour"
            ) for hour in hours
        }
        
        # Combine base columns with hour columns and add tooltips
        column_config = {
            "Owner": st.column_config.TextColumn(
                "Owner",
                width="medium",
                help="Flow owner or service account name"
            ),
            "Automation Project": st.column_config.TextColumn(
                "Automation Project",
                width="medium",
                help="Project name extracted from flow name"
            ),
            "Cloud Flow": st.column_config.TextColumn(
                "Cloud Flow",
                width="large",
                help="Name of the Power Automate flow"
            ),
            **hour_column_config
        }
        
        # Calculate appropriate height for the dataframe
        row_height = 35  # Base height per row
        min_height = 200
        header_footer_space = 100
        calculated_height = max(min_height, (len(data_rows) * row_height) + header_footer_space)
        max_height = 800
        display_height = min(calculated_height, max_height)
        
        # Add CSS styling for better emoji alignment
        st.markdown("""
        <style>
        /* Center align emoji cells */
        [data-testid="stDataFrame"] td:nth-child(n+4) {
            text-align: center !important;
            font-size: 18px !important;
            vertical-align: middle !important;
        }
        /* Improve header style */
        [data-testid="stDataFrame"] th {
            background-color: #f1f3f4 !important;
            font-weight: bold !important;
            text-align: center !important;
        }
        /* Add padding and alignment to all cells for better readability */
        [data-testid="stDataFrame"] td {
            padding: 8px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the dataframe with settings
        st.dataframe(
            matrix_df,
            column_config=column_config,
            height=display_height,
            use_container_width=True,
            hide_index=True
        )
        
        # Removed status legend from here (moved to before the matrix display)
            
    except Exception as e:
        logger.error(f"Error displaying matrix: {e}", exc_info=True)
        st.error("Error displaying the matrix. Please check logs for details.")

def load_data(use_csv=False):
    """Load data with proper error handling and status updates"""
    try:
        # Display loading status
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_placeholder.info("Loading data...")
        
        # Load data from database or CSV
        df = get_flow_data(use_csv=use_csv)
        
        if df is None or df.empty:
            status_placeholder.error("No data available. Please check data source.")
            progress_bar.empty()
            return None, None
            
        # Convert datetimestarted to datetime if it's not already
        df['datetimestarted'] = pd.to_datetime(df['datetimestarted'])
        
        # Get the latest date
        latest_date = df['datetimestarted'].dt.date.max()
        
        # Update progress
        progress_bar.progress(100)
        status_placeholder.empty()
        
        logger.info(f"Data loaded successfully with {len(df)} records. Latest date: {latest_date}")
        return df, latest_date
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Failed to load data: {str(e)}")
        return None, None

def filter_data_by_date(df, selected_date, use_latest=False):
    """Filter data for specific date"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    try:
        # Ensure datetime format
        if 'datetimestarted' not in df.columns:
            logger.error("datetimestarted column not found in DataFrame")
            return pd.DataFrame()
        
        # Get the latest date from the data if use_latest is True
        if use_latest:
            filter_date = pd.to_datetime(df['datetimestarted']).dt.date.max()
        else:
            # Convert selected_date to date object if needed
            if isinstance(selected_date, datetime):
                filter_date = selected_date.date()
            elif isinstance(selected_date, str):
                filter_date = pd.to_datetime(selected_date).date()
            else:
                filter_date = selected_date
            
        logger.info(f"Filtering data for date: {filter_date}")
        
        # Create a date column if it doesn't exist
        if 'date' not in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['datetimestarted']).dt.date
        
        # Apply the filter
        filtered_df = df[df['date'] == filter_date]
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} records")
        
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering data by date: {e}")
        return pd.DataFrame()

def create_flow_mapper():
    """Create a mapping between flows and their projects"""
    try:
        # Try to load from JSON first
        with open('flow_mapping.json', 'r') as f:
            flow_mapping = json.load(f)
            
        # Convert to DataFrame format
        mapping_data = []
        for flow_name, info in flow_mapping.items():
            mapping_data.append({
                'FlowName': flow_name.lower(),
                'Project': info['project'],
                'Type': info['type'],
                'UOW_Type': info['uow_type'],
                'Owner': info.get('owner', 'Unassigned')  # Maintain backwards compatibility
            })
        
        return pd.DataFrame(mapping_data)
        
    except FileNotFoundError:
        logger.warning("flow_mapping.json not found, creating default mapping")
        return pd.DataFrame(columns=['FlowName', 'Project', 'Owner', 'Type', 'UOW_Type'])
    except Exception as e:
        logger.error(f"Error creating flow mapping: {e}")
        return pd.DataFrame(columns=['FlowName', 'Project', 'Owner', 'Type', 'UOW_Type'])

def load_flow_mapping():
    """Load flow mapping from JSON"""
    try:
        # Try loading from the cached CSV first for performance
        if Path('flow_mapping.csv').exists():
            return pd.read_csv('flow_mapping.csv')
        else:
            # Create mapping from JSON source
            mapping_df = create_flow_mapper()
            # Cache it for future use
            mapping_df.to_csv('flow_mapping.csv', index=False)
            return mapping_df
    except Exception as e:
        logger.error(f"Error loading flow mapping: {e}")
        return pd.DataFrame(columns=['FlowName', 'Project', 'Owner', 'Type', 'UOW_Type'])

def get_project_for_flow(flow_name, mapping_df):
    """Get project name for a given flow"""
    try:
        if pd.isna(flow_name):
            return 'Other Cloud Flow'
        flow_name_lower = str(flow_name).strip().lower()
        match = mapping_df[mapping_df['FlowName'].str.lower() == flow_name_lower]
        return match['Project'].iloc[0] if not match.empty else 'Other Cloud Flow'
    except Exception as e:
        logger.error(f"Error getting project for flow {flow_name}: {e}")
        return 'Other Cloud Flow'

def initialize_session_state():
    """
    Initialize all session state variables needed for the dashboard
    
    This function sets up the following session state variables:
    - last_refresh: datetime - When the dashboard was last refreshed
    - refresh_count: int - Number of times dashboard has been refreshed
    - last_error: Optional[str] - Last error message if any
    - refresh_in_progress: bool - Flag to prevent multiple simultaneous refreshes
    - refresh_debounce_time: float - Minimum time between refresh attempts
    """
    try:
        # Initialize refresh timestamp with validation
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        elif not isinstance(st.session_state.last_refresh, datetime):
            logger.warning(f"Invalid last_refresh type: {type(st.session_state.last_refresh)}")
            st.session_state.last_refresh = datetime.now()
        
        # Initialize refresh counter with validation
        if 'refresh_count' not in st.session_state:
            st.session_state.refresh_count = 0
        elif not isinstance(st.session_state.refresh_count, int) or st.session_state.refresh_count < 0:
            logger.warning(f"Invalid refresh_count: {st.session_state.refresh_count}")
            st.session_state.refresh_count = 0
        
        # Initialize error tracking
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None
        
        # Initialize refresh control variables
        if 'refresh_in_progress' not in st.session_state:
            st.session_state.refresh_in_progress = False
            
        # Set minimum time between refresh attempts (2 seconds)
        if 'refresh_debounce_time' not in st.session_state:
            st.session_state.refresh_debounce_time = 2.0
            
    except Exception as e:
        # If session state initialization fails, log but don't crash
        logger.error(f"Failed to initialize session state: {e}", exc_info=True)

def main():
    """Main dashboard application"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Title and description
        st.title("Bot Monitoring Dashboard")
        st.markdown("Monitor Power Automate Cloud Flow execution status by hour")
        
        # Sidebar controls
        with st.sidebar:
            st.title("Dashboard Controls")
            
            # Data source selection
            use_csv = st.checkbox("Use CSV Data", value=False, 
                                 help="Use CSV files instead of database")
            
            # Load data first to get the latest date
            df, latest_date = load_data(use_csv=use_csv)
            
            # Date selection controls
            st.markdown("### Date Selection")
            use_latest = st.checkbox("Show Latest Data", value=True,
                                   help="Automatically show data from the most recent date")
            
            # Date selection
            today = date.today()
            min_date = today - timedelta(days=30)
            
            # Set the default date to latest_date if available, otherwise today
            default_date = latest_date if latest_date else today
            
            # Only show date picker if not using latest date
            if not use_latest:
                selected_date = st.date_input(
                    "Select Date", 
                    value=default_date,
                    min_value=min_date,
                    max_value=today,
                    help="Select date to view"
                )
            else:
                selected_date = latest_date if latest_date else today
                st.info(f"Showing data for: {selected_date}")
            
            # Manual refresh button with counter update
            if st.button("Refresh Data"):
                try:
                    # Update session state variables atomically
                    st.session_state.refresh_count += 1
                    st.session_state.last_refresh = datetime.now()
                    logger.info(f"Manual refresh triggered (refresh #{st.session_state.refresh_count})")
                except ValueError as val_error:
                    logger.warning(f"Value error updating session state: {val_error}")
                except TypeError as type_error:
                    logger.warning(f"Type error updating session state: {type_error}") 
                except Exception as button_error:
                    logger.warning(f"Error updating session state: {button_error}")
                # Call safe reload function
                safe_dashboard_reload()
            st.markdown("### Auto Refresh")
            auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh interval (minutes)",
                    min_value=1,
                    max_value=60,
                    value=5
                )
                
                try:
                    # Calculate time until next refresh for display
                    current_time = datetime.now()
                    next_refresh = st.session_state.last_refresh + timedelta(minutes=refresh_interval)
                    time_to_refresh = next_refresh - current_time
                    
                    # Format time remaining for display
                    seconds_to_refresh = max(0, int(time_to_refresh.total_seconds()))
                    minutes_to_refresh = seconds_to_refresh // 60
                    seconds_remaining = seconds_to_refresh % 60
                    
                    # Show next refresh time with minutes and seconds
                    if minutes_to_refresh > 0:
                        st.info(f"Next refresh in {minutes_to_refresh} min {seconds_remaining} sec")
                    else:
                        st.info(f"Next refresh in {seconds_remaining} seconds")
                    
                    # Show the refresh count
                    st.caption(f"Dashboard refreshed {st.session_state.refresh_count} times since load")
                    
                    # Trigger refresh if it's time
                    if time_to_refresh.total_seconds() <= 0:
                        try:
                            # Update refresh timestamp and count atomically
                            st.session_state.last_refresh = current_time
                            st.session_state.refresh_count += 1
                            logger.info(f"Auto-refresh triggered after {refresh_interval} minutes")
                            
                            # Perform the refresh
                            safe_dashboard_reload()
                        except ValueError as val_error:
                            logger.warning(f"Value error during auto-refresh: {val_error}")
                        except TypeError as type_error:
                            logger.warning(f"Type error during auto-refresh: {type_error}")
                        except Exception as refresh_error:
                            logger.error(f"Error during auto-refresh: {refresh_error}")
                            st.error("Auto-refresh failed. Try refreshing manually.")
                except KeyError as key_error:
                    logger.error(f"Session state key error: {key_error}")
                    st.warning("Auto-refresh configuration error. Try reloading the page.")
                except ValueError as val_error:
                    logger.error(f"Invalid value in refresh calculation: {val_error}")
                    st.warning("Invalid refresh interval value. Please adjust the slider.")
                except Exception as refresh_error:
                    logger.error(f"Auto-refresh calculation error: {refresh_error}")
                    st.warning("Error in refresh calculation. Try refreshing manually.")
        
        # Use the already loaded data from the sidebar
        if df is not None and not df.empty:
            # Add debug logging
            logger.info(f"Data loaded successfully: {len(df)} rows")
            logger.info(f"Columns available: {df.columns.tolist()}")
            
            # Filter data for selected date
            filtered_df = filter_data_by_date(df, selected_date, use_latest)
            logger.info(f"After date filtering: {len(filtered_df)} rows")
            
            if filtered_df.empty:
                st.warning(f"No data available for selected date: {selected_date}")
                return
                
            # Load flow mapping
            flow_mapping = load_flow_mapping()
            
            # Process data for dashboard display
            # Process data for dashboard display
            processed_df = process_data_for_dashboard(filtered_df)
            logger.info(f"After processing: {len(processed_df)} rows")
            
            # Add project information from mapping
            if 'flowname' in processed_df.columns:
                processed_df['automation_project'] = processed_df['flowname'].apply(
                    lambda x: get_project_for_flow(x, flow_mapping)
                )
                logger.info(f"Projects mapped: {processed_df['automation_project'].nunique()} unique projects")
            if processed_df is not None and not processed_df.empty:
                # Filter controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Get all possible projects from flow mapping
                    try:
                        with open('flow_mapping.json', 'r') as f:
                            flow_mapping = json.load(f)
                        mapped_projects = sorted(set(info['project'] for info in flow_mapping.values()))
                    except Exception as e:
                        logger.warning(f"Could not load projects from mapping: {e}")
                        mapped_projects = []
                    
                    # Combine with projects from current data
                    current_projects = sorted(processed_df['automation_project'].unique().tolist())
                    all_projects = sorted(set(mapped_projects + current_projects))
                    
                    # Remove 'Other Cloud Flow' to add it at the end
                    if 'Other Cloud Flow' in all_projects:
                        all_projects.remove('Other Cloud Flow')
                    
                    # Create final project list
                    projects = ['All Projects'] + all_projects + ['Other Cloud Flow']
                    selected_project = st.selectbox("Select Project", projects)

                with col2:
                    statuses = ['All Statuses'] + sorted(processed_df['taskstatus'].unique().tolist())
                    selected_status = st.selectbox("Select Status", statuses)

                with col3:
                    owners = ['All Owners'] + sorted(processed_df['owner'].unique().tolist())
                    selected_owner = st.selectbox("Select Owner", owners)
                    
                # Apply filters
                if selected_project != 'All Projects':
                    processed_df = processed_df[processed_df['automation_project'] == selected_project]
                if selected_status != 'All Statuses':
                    processed_df = processed_df[processed_df['taskstatus'] == selected_status]
                if selected_owner != 'All Owners':
                    processed_df = processed_df[processed_df['owner'] == selected_owner]
                
                # Create matrix data with proper display names
                bot_hour_status, display_names, hours = create_hourly_matrix(
                    processed_df,
                    selected_project=selected_project,
                    selected_status=selected_status
                )
                
                
                logger.info(f"Matrix created with {len(display_names)} display names and {len(hours)} hours")
                
                # Add spacing before Status Legend
                st.markdown("<br>", unsafe_allow_html=True)

                # Enhanced Status Legend with better styling
                st.markdown("""
                    <style>
                    .status-legend {
                        padding: 10px;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                        margin-bottom: 20px;
                    }
                    .legend-item {
                        display: inline-block;
                        margin-right: 20px;
                        padding: 5px 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("### Status Legend")
                st.markdown('<div class="status-legend">', unsafe_allow_html=True)
                legend_cols = st.columns(5)
                with legend_cols[0]:
                    st.markdown(f'<div class="legend-item">{STATUS_EMOJIS["Succeeded"]} Succeeded/Completed</div>', unsafe_allow_html=True)
                with legend_cols[1]:
                    st.markdown(f'<div class="legend-item">{STATUS_EMOJIS["Failed"]} Failed/Error</div>', unsafe_allow_html=True)
                with legend_cols[2]:
                    st.markdown(f'<div class="legend-item">{STATUS_EMOJIS["Running"]} Running/In Progress</div>', unsafe_allow_html=True)
                with legend_cols[3]:
                    st.markdown(f'<div class="legend-item">{STATUS_EMOJIS["No Run"]} No Run/Skipped</div>', unsafe_allow_html=True)
                with legend_cols[4]:
                    st.markdown(f'<div class="legend-item">{STATUS_EMOJIS["Canceled"]} Canceled</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Add spacing after legend
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Then display matrix
                if display_names:  # Check if we have data to display
                    st.markdown("### Bot Activity Matrix")
                    display_matrix(bot_hour_status, display_names, hours)
                else:
                    st.warning("No data to display for the selected filters.")
                # Show summary statistics with project information
                st.markdown("### Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("Status Distribution")
                    status_counts = processed_df['taskstatus'].value_counts()
                    st.bar_chart(status_counts)
                
                with col2:
                    st.subheader("Automation Projects")
                    project_counts = processed_df['automation_project'].value_counts().head(10)
                    st.bar_chart(project_counts)
                
                with col3:
                    st.subheader("Owner Distribution")
                    owner_counts = processed_df['owner'].value_counts()
                    st.bar_chart(owner_counts)
                
                with col4:
                with col4:
                    st.subheader("Success Rate")
                    if 'success_rate' in processed_df.columns:
                        avg_success = processed_df['success_rate'].mean()
                        st.metric("Overall Success Rate", f"{avg_success:.1f}%")
                    else:
                        success_rate = processed_df['wassuccessful'].mean() * 100
                        st.metric("Overall Success Rate", f"{success_rate:.1f}%")
                
                # Success rate by project with enhanced metrics - moved outside of col4
                project_metrics = (processed_df.groupby('automation_project')
                    .agg({
                        'wassuccessful': ['count', 'mean'],
                        'taskstatus': lambda x: (x == 'Failed').mean(),
                        'flowname': 'nunique'
                    })
                    .round(4)
                )

                # Create multi-level column names
                project_metrics.columns = [
                    'Total Executions',
                    'Success Rate',
                    'Failure Rate',
                    'Unique Flows'
                ]

                # Calculate metrics
                project_metrics['Success Rate'] = project_metrics['Success Rate'] * 100
                project_metrics['Failure Rate'] = project_metrics['Failure Rate'] * 100
                project_metrics['Health Score'] = (
                    project_metrics['Success Rate'] - 
                    (project_metrics['Failure Rate'] * 2)
                ).round(1)

                # Create display dataframe with better formatting
                success_display = pd.DataFrame({
                    'Project': project_metrics.index,
                    'Success Rate': project_metrics['Success Rate'].round(1),
                    'Failed Rate': project_metrics['Failure Rate'].round(1),
                    'Total Runs': project_metrics['Total Executions'],
                    'Active Flows': project_metrics['Unique Flows'],
                    'Health Score': project_metrics['Health Score']
                })

                # Display with better formatting
                st.markdown("### Project Performance Metrics")
                st.dataframe(
                    success_display.sort_values('Health Score', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Project': st.column_config.TextColumn(
                            'Project Name',
                            help='Automation project name'
                        ),
                        'Success Rate': st.column_config.NumberColumn(
                            'Success Rate',
                            format="%.1f%%",
                            help="Percentage of successful executions"
                        ),
                        'Failed Rate': st.column_config.NumberColumn(
                            'Failure Rate',
                            format="%.1f%%",
                            help="Percentage of failed executions"
                        ),
                        'Total Runs': st.column_config.NumberColumn(
                            'Total Executions',
                            help="Total number of flow executions"
                        ),
                        'Active Flows': st.column_config.NumberColumn(
                            'Active Flows',
                            help="Number of distinct flows in the project"
                        ),
                        'Health Score': st.column_config.NumberColumn(
                            'Health Score',
                            format="%.1f",
                            help="Project health score (Success Rate - 2 × Failure Rate)"
                        )
                    }
                )
                st.markdown("### Additional Analytics")
                
                # 1. Performance Metrics
                st.subheader("Performance Metrics")
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    try:
                        avg_duration = processed_df['datetimecompleted'].dt.timestamp() - processed_df['datetimestarted'].dt.timestamp()
                        avg_duration_mins = avg_duration.mean() / 60
                        st.metric("Average Duration", f"{avg_duration_mins:.1f} mins")
                    except:
                        st.metric("Average Duration", "N/A")
                
                with metric_cols[1]:
                    failure_rate = (processed_df['taskstatus'] == 'Failed').mean() * 100
                    st.metric("Failure Rate", f"{failure_rate:.1f}%")
                
                with metric_cols[2]:
                    total_runs = len(processed_df)
                    st.metric("Total Executions", f"{total_runs:,}")
                
                with metric_cols[3]:
                    active_flows = processed_df['flowname'].nunique()
                    st.metric("Active Flows", f"{active_flows:,}")

                # 2. Hourly Trends
                st.subheader("Execution Trends")
                trend_cols = st.columns(2)
                
                with trend_cols[0]:
                    st.markdown("#### Hourly Distribution")
                    hourly_dist = processed_df.groupby(processed_df['datetimestarted'].dt.hour)['flowname'].count()
                    st.bar_chart(hourly_dist)
                
                with trend_cols[1]:
                    st.markdown("#### Success Rate by Hour")
                    hourly_success = processed_df.groupby(processed_df['datetimestarted'].dt.hour)['wassuccessful'].mean() * 100
                    st.line_chart(hourly_success)

                # 3. Top Issues Analysis
                st.subheader("Issue Analysis")
                issue_cols = st.columns(2)
                
                with issue_cols[0]:
                    st.markdown("#### Top Failing Flows")
                    failed_df = processed_df[processed_df['taskstatus'] == 'Failed']
                    if not failed_df.empty:
                        failed_flows = (failed_df
                                      .groupby('flowname')
                                      .size()
                                      .sort_values(ascending=False)
                                      .head(5))
                        st.bar_chart(failed_flows)
                    else:
                        st.info("No failed flows in the selected timeframe.")
                
                with issue_cols[1]:
                    st.markdown("#### Project Health Score")
                    project_health = (processed_df.groupby('automation_project')
                                    .agg({
                                        'wassuccessful': 'mean',
                                        'taskstatus': lambda x: (x == 'Failed').mean()
                                    })
                                    .assign(health_score=lambda x: (x['wassuccessful'] * 100 - x['taskstatus'] * 50))
                                    .sort_values('health_score', ascending=False))
                    st.dataframe(project_health.round(2))

                # 4. Execution Timeline
                st.subheader("Execution Timeline")
                try:
                    timeline_data = (processed_df.groupby(pd.Grouper(key='datetimestarted', freq='15T'))
                                   .agg({
                                       'flowname': 'count',
                                       'wassuccessful': 'mean'
                                   }))
                    st.line_chart(timeline_data)
                except Exception as e:
                    logger.warning(f"Could not generate timeline: {e}")
                    st.warning("Could not generate execution timeline. Check data format.")
        else:
            st.error("No data available. Please check data source and try again.")
    
    except Exception as e:
        try:
            st.session_state.last_error = str(e)
        except:
            # If session state cannot be updated, just log the error
            pass
            
        # Log the full error with traceback
        logger.error(f"Dashboard error: {e}", exc_info=True)
        
        # Display friendly error to user
        st.error(f"An error occurred: {str(e)}")
        
        # Show traceback in an expandable section for debugging
        with st.expander("Error Details (for troubleshooting)"):
            st.code(traceback.format_exc())
            
    finally:
        # Cleanup resources and ensure refresh flags are reset
        try:
            # Reset refresh progress flag if set
            if st.session_state.get('refresh_in_progress', False):
                st.session_state.refresh_in_progress = False
                logger.info("Reset refresh_in_progress flag during cleanup")
                
            # Force garbage collection to free memory
            gc.collect()
        except:
            # Don't let cleanup errors affect the application
            pass

if __name__ == "__main__":
    main()

