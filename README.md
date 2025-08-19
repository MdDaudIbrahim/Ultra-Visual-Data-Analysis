# Excel Data Analyzer

A comprehensive web-based data analysis tool for Excel and CSV files with interactive visualizations and statistical insights.

## Features

### 📊 Data Analysis
- **File Upload**: Support for Excel (.xlsx, .xls) and CSV files
- **Interactive Tables**: Browse, search, and filter your data
- **Statistical Summary**: Comprehensive descriptive statistics
- **Data Quality**: Missing data analysis and visualization

### 📈 Visualizations
- **Distribution Plots**: Histograms and box plots for numeric data
- **Correlation Heatmaps**: Understand relationships between variables
- **Scatter Plots**: Explore relationships with optional color coding
- **Value Counts**: Visualize categorical data distributions
- **Time Series**: Analyze trends over time
- **Custom Charts**: Group-by analysis and aggregations

### 💾 Export Options
- **Download Data**: Export processed data as Excel or CSV
- **Summary Reports**: Generate analysis summaries
- **Numeric Data**: Export only numeric columns for further analysis

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run excel_analyzer.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Usage

1. **Upload Your File**: Use the sidebar to upload Excel or CSV files
2. **Configure Analysis**: Choose which analysis sections to display
3. **Explore Data**: 
   - Browse raw data with search and filtering
   - View statistical summaries
   - Analyze missing data patterns
   - Explore correlations and distributions
4. **Create Custom Charts**: Use the custom charts section for specific visualizations
5. **Export Results**: Download processed data and analysis reports

## Supported File Formats

- **Excel Files**: .xlsx, .xls
- **CSV Files**: .csv
- **Data Types**: Automatic detection of numeric, categorical, and datetime columns

## Key Features Explained

### Data Overview
- **Total Rows/Columns**: Quick dataset size metrics
- **Column Types**: Automatic categorization of data types
- **Missing Data**: Visual and tabular missing data analysis

### Statistical Analysis
- **Descriptive Statistics**: Mean, median, std dev, percentiles
- **Distribution Analysis**: Histograms, box plots, skewness
- **Correlation Analysis**: Heatmaps for numeric relationships

### Interactive Features
- **Real-time Search**: Filter data as you type
- **Dynamic Charts**: Interactive Plotly visualizations
- **Custom Analysis**: Group-by operations with aggregations
- **Export Options**: Multiple download formats

## System Requirements

- **Memory**: Minimum 4GB RAM (8GB+ recommended for large files)
- **Storage**: 100MB free space
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Tips for Best Results

1. **File Size**: Works best with files under 100MB
2. **Data Quality**: Clean column names improve analysis
3. **Date Formats**: Use standard date formats for time series analysis
4. **Missing Data**: Consider cleaning strategies before upload

## Troubleshooting

### Common Issues

**File Upload Fails**:
- Check file format (Excel/CSV only)
- Ensure file isn't corrupted
- Try smaller file size

**Charts Not Displaying**:
- Refresh the page
- Check if data has numeric columns
- Verify data isn't all null

**Performance Issues**:
- Use smaller datasets
- Close unnecessary browser tabs
- Increase system memory

## Advanced Features

### Custom Analysis
- **Group By Operations**: Aggregate data by categories
- **Multiple Aggregations**: Mean, sum, count, min, max
- **Filtering**: Real-time data filtering
- **Sorting**: Interactive data sorting

### Export Capabilities
- **Processed Data**: Download cleaned/filtered data
- **Summary Reports**: Comprehensive analysis reports
- **Chart Images**: Export visualizations (via browser)

## Technical Details

### Libraries Used
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions
- **OpenPyXL**: Excel file handling

### Performance Optimization
- **Lazy Loading**: Charts load on demand
- **Memory Management**: Efficient data handling
- **Caching**: Streamlit caching for better performance

## Future Enhancements

- **Machine Learning**: Automated insights and predictions
- **Database Support**: Connect to SQL databases
- **Advanced Statistics**: Hypothesis testing, regression analysis
- **Collaboration**: Share analysis results
- **Real-time Data**: Live data source connections

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your data format
3. Try with a smaller sample file
4. Restart the application

## License

This project is for educational and personal use. Feel free to modify and enhance as needed.

---

**Happy Data Analyzing! 📊**
