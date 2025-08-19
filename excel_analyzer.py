import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Excel Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e4057;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load data from uploaded Excel file"""
    try:
        # Try to read Excel file
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload an Excel (.xlsx, .xls) or CSV file")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_data_info(df):
    """Get comprehensive data information"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
    }
    return info

def create_summary_stats(df):
    """Create summary statistics"""
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        return numeric_df.describe()
    return None

def create_missing_data_chart(df):
    """Create missing data visualization"""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    
    if not missing_data.empty:
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title="Missing Data by Column",
            labels={'x': 'Number of Missing Values', 'y': 'Columns'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(height=500)
        return fig
    return None

def create_distribution_plots(df, column):
    """Create distribution plots for a column"""
    if column in df.select_dtypes(include=[np.number]).columns:
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            name="Distribution",
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            height=400
        )
        return fig
    return None

def create_box_plot(df, column):
    """Create box plot for a column"""
    if column in df.select_dtypes(include=[np.number]).columns:
        fig = px.box(
            df,
            y=column,
            title=f"Box Plot of {column}",
            height=400
        )
        return fig
    return None

def create_value_counts_chart(df, column, top_n=10):
    """Create value counts chart for categorical columns"""
    if column in df.select_dtypes(include=['object']).columns:
        value_counts = df[column].value_counts().head(top_n)
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Top {top_n} Values in {column}",
            labels={'x': column, 'y': 'Count'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create scatter plot"""
    if x_col in df.columns and y_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col in df.columns else None,
            title=f"Scatter Plot: {x_col} vs {y_col}",
            height=500
        )
        return fig
    return None

def create_time_series_plot(df, date_col, value_col):
    """Create time series plot"""
    if date_col in df.columns and value_col in df.columns:
        # Try to convert to datetime if not already
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            fig = px.line(
                df_copy,
                x=date_col,
                y=value_col,
                title=f"Time Series: {value_col} over {date_col}",
                height=500
            )
            return fig
        except:
            st.warning(f"Could not convert {date_col} to datetime format")
    return None

def download_excel(df, filename="processed_data.xlsx"):
    """Create download link for Excel file"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Data')
    writer.close()
    processed_data = output.getvalue()
    
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Processed Excel File</a>'
    return href

def main():
    # Header
    st.markdown('<p class="main-header">📊 Excel Data Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Upload your Excel file and get comprehensive data analysis with interactive visualizations!")
    
    # Sidebar
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your data file to start analysis"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Get data information
            data_info = get_data_info(df)
            
            # Sidebar options
            st.sidebar.markdown("---")
            st.sidebar.header("🔧 Analysis Options")
            
            show_raw_data = st.sidebar.checkbox("Show Raw Data", value=True)
            show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
            show_missing_data = st.sidebar.checkbox("Show Missing Data Analysis", value=True)
            show_correlations = st.sidebar.checkbox("Show Correlations", value=True)
            show_distributions = st.sidebar.checkbox("Show Distributions", value=True)
            show_custom_charts = st.sidebar.checkbox("Show Custom Charts", value=True)
            
            # Main content
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", data_info['shape'][0])
            with col2:
                st.metric("Total Columns", data_info['shape'][1])
            with col3:
                st.metric("Numeric Columns", len(data_info['numeric_columns']))
            with col4:
                st.metric("Categorical Columns", len(data_info['categorical_columns']))
            
            # Raw Data Section
            if show_raw_data:
                st.markdown('<div class="section-header">📋 Raw Data</div>', unsafe_allow_html=True)
                
                # Search and filter functionality
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_term = st.text_input("🔍 Search in data")
                with col2:
                    rows_to_show = st.selectbox("Rows to display", [10, 25, 50, 100, "All"])
                
                # Apply search filter
                display_df = df.copy()
                if search_term:
                    mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                    display_df = display_df[mask]
                
                # Apply row limit
                if rows_to_show != "All":
                    display_df = display_df.head(rows_to_show)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download option
                st.markdown(download_excel(df, f"processed_{uploaded_file.name}"), unsafe_allow_html=True)
            
            # Summary Statistics
            if show_summary:
                st.markdown('<div class="section-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
                summary_stats = create_summary_stats(df)
                if summary_stats is not None:
                    st.dataframe(summary_stats, use_container_width=True)
                else:
                    st.info("No numeric columns found for summary statistics.")
            
            # Missing Data Analysis
            if show_missing_data:
                st.markdown('<div class="section-header">❓ Missing Data Analysis</div>', unsafe_allow_html=True)
                
                missing_chart = create_missing_data_chart(df)
                if missing_chart:
                    st.plotly_chart(missing_chart, use_container_width=True)
                else:
                    st.success("✅ No missing data found!")
                
                # Missing data table
                missing_info = pd.DataFrame({
                    'Column': data_info['columns'],
                    'Missing Count': [data_info['missing_values'][col] for col in data_info['columns']],
                    'Missing %': [round(data_info['missing_percentage'][col], 2) for col in data_info['columns']]
                })
                missing_info = missing_info[missing_info['Missing Count'] > 0]
                if not missing_info.empty:
                    st.dataframe(missing_info, use_container_width=True)
            
            # Correlation Analysis
            if show_correlations and len(data_info['numeric_columns']) > 1:
                st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
                
                corr_heatmap = create_correlation_heatmap(df)
                if corr_heatmap:
                    st.plotly_chart(corr_heatmap, use_container_width=True)
            
            # Distribution Analysis
            if show_distributions and data_info['numeric_columns']:
                st.markdown('<div class="section-header">📊 Distribution Analysis</div>', unsafe_allow_html=True)
                
                selected_column = st.selectbox(
                    "Select column for distribution analysis",
                    data_info['numeric_columns']
                )
                
                if selected_column:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        dist_plot = create_distribution_plots(df, selected_column)
                        if dist_plot:
                            st.plotly_chart(dist_plot, use_container_width=True)
                    
                    with col2:
                        box_plot = create_box_plot(df, selected_column)
                        if box_plot:
                            st.plotly_chart(box_plot, use_container_width=True)
                    
                    # Statistical tests
                    st.subheader(f"Statistical Analysis for {selected_column}")
                    col_data = df[selected_column].dropna()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{col_data.mean():.2f}")
                    with col2:
                        st.metric("Median", f"{col_data.median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{col_data.std():.2f}")
                    with col4:
                        st.metric("Skewness", f"{stats.skew(col_data):.2f}")
            
            # Custom Charts
            if show_custom_charts:
                st.markdown('<div class="section-header">🎨 Custom Charts</div>', unsafe_allow_html=True)
                
                chart_type = st.selectbox(
                    "Select chart type",
                    ["Scatter Plot", "Value Counts", "Time Series", "Custom Analysis"]
                )
                
                if chart_type == "Scatter Plot" and len(data_info['numeric_columns']) >= 2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_column = st.selectbox("X-axis", data_info['numeric_columns'])
                    with col2:
                        y_column = st.selectbox("Y-axis", data_info['numeric_columns'])
                    with col3:
                        color_column = st.selectbox("Color by (optional)", ["None"] + data_info['categorical_columns'])
                    
                    if x_column and y_column:
                        color_col = color_column if color_column != "None" else None
                        scatter_plot = create_scatter_plot(df, x_column, y_column, color_col)
                        if scatter_plot:
                            st.plotly_chart(scatter_plot, use_container_width=True)
                
                elif chart_type == "Value Counts" and data_info['categorical_columns']:
                    selected_cat_column = st.selectbox(
                        "Select categorical column",
                        data_info['categorical_columns']
                    )
                    top_n = st.slider("Number of top values to show", 5, 20, 10)
                    
                    if selected_cat_column:
                        value_counts_chart = create_value_counts_chart(df, selected_cat_column, top_n)
                        if value_counts_chart:
                            st.plotly_chart(value_counts_chart, use_container_width=True)
                
                elif chart_type == "Time Series":
                    all_columns = data_info['columns']
                    col1, col2 = st.columns(2)
                    with col1:
                        date_column = st.selectbox("Date column", all_columns)
                    with col2:
                        value_column = st.selectbox("Value column", data_info['numeric_columns'])
                    
                    if date_column and value_column:
                        ts_plot = create_time_series_plot(df, date_column, value_column)
                        if ts_plot:
                            st.plotly_chart(ts_plot, use_container_width=True)
                
                elif chart_type == "Custom Analysis":
                    st.subheader("Custom Data Analysis")
                    
                    # Group by analysis
                    if data_info['categorical_columns'] and data_info['numeric_columns']:
                        st.write("**Group By Analysis**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            group_column = st.selectbox("Group by", data_info['categorical_columns'])
                        with col2:
                            agg_column = st.selectbox("Aggregate column", data_info['numeric_columns'])
                        with col3:
                            agg_function = st.selectbox("Aggregation", ["mean", "sum", "count", "min", "max"])
                        
                        if group_column and agg_column:
                            grouped_data = df.groupby(group_column)[agg_column].agg(agg_function).sort_values(ascending=False)
                            
                            fig = px.bar(
                                x=grouped_data.index,
                                y=grouped_data.values,
                                title=f"{agg_function.title()} of {agg_column} by {group_column}",
                                labels={'x': group_column, 'y': f"{agg_function}({agg_column})"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(grouped_data.reset_index(), use_container_width=True)
            
            # Data Export Section
            st.markdown('<div class="section-header">💾 Export Options</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Summary Report"):
                    summary_data = {
                        'File Name': uploaded_file.name,
                        'Upload Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Total Rows': data_info['shape'][0],
                        'Total Columns': data_info['shape'][1],
                        'Numeric Columns': len(data_info['numeric_columns']),
                        'Categorical Columns': len(data_info['categorical_columns']),
                        'Missing Values': sum(data_info['missing_values'].values())
                    }
                    summary_df = pd.DataFrame([summary_data])
                    st.download_button(
                        label="Download Summary",
                        data=summary_df.to_csv(index=False),
                        file_name=f"summary_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"processed_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
            
            with col3:
                if data_info['numeric_columns']:
                    numeric_only_df = df[data_info['numeric_columns']]
                    numeric_csv = numeric_only_df.to_csv(index=False)
                    st.download_button(
                        label="🔢 Download Numeric Data",
                        data=numeric_csv,
                        file_name=f"numeric_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )

    else:
        # Welcome message
        st.markdown("""
        ### 🚀 Welcome to Excel Data Analyzer!
        
        This powerful tool helps you analyze your Excel and CSV files with:
        
        - **📊 Interactive Data Tables** - Browse and search your data
        - **📈 Statistical Analysis** - Get comprehensive statistics
        - **🎯 Missing Data Detection** - Identify and visualize gaps
        - **🔗 Correlation Analysis** - Understand relationships
        - **📊 Distribution Plots** - Visualize data patterns
        - **🎨 Custom Charts** - Create tailored visualizations
        - **💾 Export Options** - Download processed data
        
        **Getting Started:**
        1. Upload your Excel (.xlsx, .xls) or CSV file using the sidebar
        2. Explore your data with interactive tables and filters
        3. Analyze patterns with automatic visualizations
        4. Export your insights and processed data
        
        **Supported Features:**
        - Multiple file formats (Excel, CSV)
        - Real-time data filtering and search
        - Statistical summaries and tests
        - Interactive charts and plots
        - Data quality assessment
        - Custom analysis tools
        """)
        
        # Example data section
        st.markdown("### 📝 Example Data Format")
        example_data = pd.DataFrame({
            'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'Sales': [100, 150, 120],
            'Region': ['North', 'South', 'East'],
            'Product': ['A', 'B', 'A']
        })
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
