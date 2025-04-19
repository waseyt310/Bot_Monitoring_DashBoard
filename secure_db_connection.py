import os
import sys
import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('database_connection')

# Try to import optional dependencies with graceful fallback
try:
    import pypyodbc as odbc
    ODBC_AVAILABLE = True
    logger.info("ODBC driver available - database connection enabled")
except ImportError:
    logger.warning("ODBC driver (pypyodbc) not available - will use CSV fallback")
    ODBC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if present
    DOTENV_AVAILABLE = True
except ImportError:
    logger.warning("python-dotenv not available - using OS environment variables only")
    DOTENV_AVAILABLE = False

def load_environment_variables():
    """
    Load environment variables from .env file or environment
    Returns True if all required variables are present
    """
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    required_vars = ['DB_SERVER', 'DB_NAME', 'DB_UID', 'DB_PWD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def get_connection_string():
    """Build connection string from environment variables"""
    try:
        driver_name = '{SQL SERVER}'  # Standard SQL Server driver name
        server = os.getenv('DB_SERVER')
        database = os.getenv('DB_NAME')
        uid = os.getenv('DB_UID')
        pwd = os.getenv('DB_PWD')
        
        if not all([server, database, uid, pwd]):
            raise ValueError("Missing required database configuration variables")
        
        connection_string = (
            f"DRIVER={driver_name};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={uid};"
            f"PWD={pwd};"
        )
        
        # Log connection string with masked credentials
        masked_connection = connection_string.replace(pwd, "******") if pwd else connection_string
        logger.debug(f"Connection string (masked): {masked_connection}")
        
        return connection_string
    
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Falling back to CSV data.")
        df = get_data_from_csv()
        if not df.empty:
            return df
        else:
            logger.info("No CSV data available. Using sample data.")
            return generate_sample_data()

def create_db_connection():
    """Create database connection with error handling"""
    try:
        # Check if ODBC driver is available
        if not ODBC_AVAILABLE:
            raise ImportError("ODBC driver (pypyodbc) not available - cannot create database connection")
            
        # Check environment variables
        if not load_environment_variables():
            raise ValueError("Required environment variables not found")
        
        # Get connection string and connect
        connection_string = get_connection_string()
        connection = odbc.connect(connection_string)
        logger.info("Database connection established successfully")
        return connection
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
    except odbc.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating database connection: {e}")
        raise

def execute_query(connection, query, params=None):
    """Execute SQL query with error handling"""
    try:
        cursor = connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    except odbc.Error as e:
        logger.error(f"Query execution failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query execution: {e}")
        raise

def generate_sample_data():
    """
    Generate sample data for demonstration when no real data is available
    
    Returns:
        pandas.DataFrame: Sample flow data for demonstration
    """
    try:
        logger.info("Generating sample data for demonstration purposes")
        
        # Sample flow names
        flows = [
            "AMZ - Order Processing", 
            "C2D - Data Integration",
            "PS - Report Generation",
            "WF - System Check",
            "BI - Data Analytics"
        ]
        
        # Sample owners
        owners = [
            "powerautomate",
            "powerautomate02 serviceaccount",
            "powerautomate03 serviceaccount",
            "powerautomate04"
        ]
        
        # Status options with probabilities
        statuses = ["Succeeded", "Failed", "Running", "Canceled"]
        status_probs = [0.7, 0.15, 0.1, 0.05]  # 70% success rate
        
        # Generate sample data
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        
        # Create empty list to hold records
        records = []
        
        # Generate records for each flow
        for flow in flows:
            for hour in range(24):
                # Half the flows run every hour, half run every other hour
                if hour % 2 == 0 or flow.startswith(("AMZ", "C2D")):
                    # Random start time within the hour
                    start_min = np.random.randint(0, 50)
                    start_time = datetime.combine(yesterday, datetime.min.time()) + timedelta(hours=hour, minutes=start_min)
                    
                    # Random duration between 1-15 minutes
                    duration = np.random.randint(1, 15)
                    end_time = start_time + timedelta(minutes=duration)
                    
                    # Random status with weighted probability
                    status = np.random.choice(statuses, p=status_probs)
                    
                    # Create record
                    record = {
                        'flowguid': f"sample-{flow}-{hour}",
                        'flowname': flow,
                        'startedon': start_time.isoformat(),
                        'lastmodified': end_time.isoformat(),
                        'state': 'Completed' if status in ['Succeeded', 'Failed'] else 'Active',
                        'flowowner': np.random.choice(owners),
                        'datetimestarted': start_time,
                        'datetimecompleted': end_time if status != 'Running' else None,
                        'taskstatus': status,
                        'triggertype': np.random.choice(['Recurrence', 'manual'], p=[0.8, 0.2]),
                        'wassuccessful': 1 if status == 'Succeeded' else 0,
                        'finalsuccessful': 1 if status == 'Succeeded' else 0
                    }
                    records.append(record)
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(records)
        logger.info(f"Generated {len(sample_df)} sample records for demonstration")
        return sample_df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        # Return minimal sample data on error
        columns = [
            'flowguid', 'flowname', 'startedon', 'lastmodified', 'state', 
            'flowowner', 'datetimestarted', 'datetimecompleted', 'taskstatus', 
            'triggertype', 'wassuccessful', 'finalsuccessful'
        ]
        return pd.DataFrame(columns=columns)

def get_data_from_csv(filepath=None):
    """
    Fallback function to load data from CSV when database connection is not available
    
    Args:
        filepath (str, optional): Path to CSV file. If None, looks for most recent flow_data_*.csv
    
    Returns:
        pandas.DataFrame: Data loaded from CSV file
    """
    try:
        if filepath is None:
            # Look for CSV files in multiple possible locations
            csv_files = []
            search_paths = ['.', './data', 'data']
            
            for path in search_paths:
                if os.path.exists(path):
                    logger.info(f"Searching for CSV files in '{path}'")
                    try:
                        # Try using Path for more robust file finding
                        path_obj = Path(path)
                        csv_files.extend([str(f) for f in path_obj.glob('flow_data_*.csv')])
                    except Exception as e:
                        logger.warning(f"Error searching path '{path}': {e}")
                        
                        # Fallback to os.listdir if Path.glob fails
                        try:
                            matching_files = [
                                os.path.join(path, f) 
                                for f in os.listdir(path) 
                                if f.startswith('flow_data_') and f.endswith('.csv')
                            ]
                            csv_files.extend(matching_files)
                        except Exception as list_err:
                            logger.warning(f"Error listing files in '{path}': {list_err}")
            
            if not csv_files:
                # Generate sample data if no CSV files found
                logger.error("No flow_data_*.csv files found in any search path")
                return generate_sample_data()
            
            # Get the most recent file
            filepath = max(csv_files, key=lambda x: os.path.getmtime(x))
            logger.info(f"Using most recent CSV file: {filepath}")
        
        # Load the CSV file with explicit error handling
        try:
            df = pd.read_csv(filepath)
            
            # Ensure wassuccessful column exists
            if 'wassuccessful' not in df.columns and 'taskstatus' in df.columns:
                df['wassuccessful'] = df['taskstatus'].apply(
                    lambda x: 1 if x == 'Succeeded' else 0
                )
            
            # Ensure datetimestarted is datetime
            if 'datetimestarted' in df.columns:
                df['datetimestarted'] = pd.to_datetime(df['datetimestarted'], errors='coerce')
            
            logger.info(f"Successfully loaded {len(df)} records from CSV")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file '{filepath}' is empty")
            return pd.DataFrame()
            
        except pd.errors.ParserError as parser_err:
            logger.error(f"Error parsing CSV file '{filepath}': {parser_err}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame on error

def get_flow_data(use_csv=False):
    """
    Get flow data from either database, CSV, or generate sample data
    
    Args:
        use_csv (bool): Force using CSV instead of database
    
    Returns:
        pandas.DataFrame: Flow data from one of the available sources
    """
    # If ODBC driver isn't available or CSV is specifically requested, use CSV
    if use_csv or not ODBC_AVAILABLE:
        logger.info("Using CSV data source")
        df = get_data_from_csv()
        if not df.empty:
            return df
        else:
            logger.info("No CSV data available. Using sample data.")
            return generate_sample_data()
    
    try:
        # Try database connection first
        connection = create_db_connection()
        
        query = """
        SELECT
            FlowGUID as flowguid,
            FlowName as flowname,
            CreatedTime as startedon,
            LastModified as lastmodified,
            State as state,
            FlowOwner as flowowner,
            StartTime as datetimestarted,
            EndTime as datetimecompleted,
            TaskStatus as taskstatus,
            TriggerType as triggertype,
            CASE WHEN TaskStatus = 'Succeeded' THEN 1 ELSE 0 END as wassuccessful,
            CASE WHEN TaskStatus = 'Succeeded' THEN 1 ELSE 0 END as finalsuccessful
        FROM BusinessAnalytics.dbo.rpa_FlowRunHistory
        WHERE FlowOwner in (
            'powerautomate', 'powerautomate02 serviceaccount',
            'powerautomate03 serviceaccount', 'powerautomate04',
            'powerautomate05', 'powerautomate06', 'powerautomate07',
            'powerautomate08', 'Ryan Kieselhorst', 'Colin Boyle',
            'Cheddrick Bagunu', 'Edu Cielo', 'Mohammad Asim'
        )
        AND StartTime >= DATEADD(month, -1, GETDATE())
        """
        
        cursor = execute_query(connection, query)
        columns = [column[0] for column in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame.from_records(data, columns=columns)
        
        # Close connection
        connection.close()
        logger.info(f"Successfully retrieved {len(df)} records from database")
        
        return df
        
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Falling back to CSV data.")
        return get_data_from_csv()

def test_connection() -> Tuple[bool, str]:
    """
    Test database connection and environment variables
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # First check if ODBC is available
        if not ODBC_AVAILABLE:
            return False, "ODBC driver not available"
        
        # Check environment variables
        if not load_environment_variables():
            return False, "Required environment variables not found"
        
        # Try connecting
        connection = create_db_connection()
        cursor = execute_query(connection, "SELECT 1")
        result = cursor.fetchone()
        connection.close()
        
        if result and result[0] == 1:
            return True, "Connection test successful"
        else:
            return False, "Connection test failed: unexpected result"
            
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except odbc.Error as e:
        return False, f"Database error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

if __name__ == "__main__":
    # Test the connection when run directly
    success, message = test_connection()
    if success:
        print("✅ Database connection test successful!")
    else:
        print(f"❌ Database connection test failed: {message}")
        
        # Try CSV fallback
        print("\nTesting CSV fallback...")
        df = get_data_from_csv()
        if not df.empty:
            print(f"✅ Successfully loaded {len(df)} records from CSV")
        else:
            print("❌ No data available from CSV fallback")

