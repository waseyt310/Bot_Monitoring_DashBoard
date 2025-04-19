# Bot Monitoring Dashboard

A Streamlit-based dashboard for monitoring Power Automate Cloud Flow execution status. This application provides real-time visualization of flow runs, success rates, and execution patterns.

## Features

- 📊 Real-time monitoring of Power Automate Cloud Flows
- 🟢🔴🟡 Status visualization with emoji indicators
- 🔍 Filtering by project and status
- 📅 Date selection for historical data
- 📈 Success rate analytics
- 🔄 Auto-refresh capability
- 📁 Automatic fallback to CSV data when database is unavailable

## Installation

### Prerequisites

- Python 3.8 or higher
- SQL Server (for database connection) or CSV data files

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/waseyt310/test1.git
   cd test1
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Database Connection

Create a `.env` file in the project root with the following variables:
```
DB_SERVER=your_server_name
DB_NAME=your_database_name
DB_UID=your_username
DB_PWD=your_password
```

### CSV Fallback

If database connection is not available, the application will automatically fall back to using CSV data files. Place your data files in one of these locations:
- Root directory: `flow_data_*.csv`
- `data/` directory: `data/flow_data_*.csv`

The application will automatically find the most recent file.

## Usage

### Running Locally

Start the Streamlit app:
```bash
streamlit run bot_monitor_dashboard.py
```

Access the dashboard at `http://localhost:8501`

### Dashboard Controls

- **Date Selection**: Choose the date to view flow execution data
- **Project Filter**: Filter flows by project
- **Status Filter**: Filter by execution status (Succeeded, Failed, Running, etc.)
- **Auto-Refresh**: Enable automatic data refresh at specified intervals

## Deployment

### Streamlit Cloud

1. Fork this repository to your GitHub account
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your forked repository
4. Set the Main file path to: `bot_monitor_dashboard.py`
5. Configure environment variables in Streamlit Cloud:
   - Go to Advanced Settings > Secrets
   - Add the database credentials as shown below

```toml
[db_credentials]
DB_SERVER = "your_server_name"
DB_NAME = "your_database_name"
DB_UID = "your_username"
DB_PWD = "your_password"
```


## Project Structure

```
.
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── data_processing/
│   ├── __init__.py      # Package initialization
│   ├── processors.py    # Data processing logic
│   └── validators.py    # Data validation functions
├── data/                # Optional directory for CSV files
├── .env                 # Environment variables (local only)
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── bot_monitor_dashboard.py  # Main Streamlit application
├── requirements.txt     # Python dependencies
└── secure_db_connection.py   # Database connectivity module
```

## Error Handling

The application includes comprehensive error handling mechanisms:
- Automatic fallback to CSV if database connection fails
- Graceful handling of missing dependencies
- Memory optimization for large datasets
- Detailed logging for troubleshooting

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   - Verify credentials in `.env` file
   - Check if the database server is accessible from your network
   - Confirm that pypyodbc is properly installed

2. **Missing Data**
   - Ensure CSV files are available if database connection fails
   - Check date selection - no data may be available for selected date

3. **Memory Issues**
   - Application automatically manages memory usage
   - For large datasets, consider filtering by project or date

### Logs

Check the console output for detailed logs. The application uses logging at INFO level by default.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

