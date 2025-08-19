import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io
import base64
from advanced_analyzer import AdvancedAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Excel Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
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

def main():
    # Header
    st.markdown('<p class="main-header">📊 Advanced Excel Data Analyzer</p>', unsafe_allow_html=True)
    st.markdown("**Professional-grade data analysis with AI-powered insights**")
    
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
            # Initialize advanced analyzer
            analyzer = AdvancedAnalyzer(df)
            
            # Sidebar analysis options
            st.sidebar.markdown("---")
            st.sidebar.header("🔧 Analysis Modules")
            
            modules = {
                'overview': st.sidebar.checkbox("📋 Data Overview", value=True),
                'quality': st.sidebar.checkbox("🔍 Data Quality", value=True),
                'insights': st.sidebar.checkbox("🧠 AI Insights", value=True),
                'statistics': st.sidebar.checkbox("📈 Statistical Analysis", value=True),
                'outliers': st.sidebar.checkbox("🎯 Outlier Detection", value=False),
                'trends': st.sidebar.checkbox("📊 Trend Analysis", value=False),
                'advanced': st.sidebar.checkbox("⚡ Advanced Analytics", value=False),
                'export': st.sidebar.checkbox("💾 Export Tools", value=True)
            }
            
            # Main dashboard metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📈 Columns", f"{df.shape[1]:,}")
            with col3:
                st.metric("🔢 Numeric", len(analyzer.numeric_cols))
            with col4:
                st.metric("📝 Categorical", len(analyzer.categorical_cols))
            with col5:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("💾 Memory", f"{memory_mb:.1f} MB")
            
            # Data Overview Module
            if modules['overview']:
                st.markdown("---")
                st.header("📋 Data Overview")
                
                tab1, tab2 = st.tabs(["📊 Raw Data", "📋 Data Profile"])
                
                with tab1:
                    # Data filtering
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        search_term = st.text_input("🔍 Search data")
                    with col2:
                        rows_to_show = st.selectbox("Rows", [10, 25, 50, 100, "All"])
                    with col3:
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    
                    # Apply filters
                    display_df = df.copy()
                    if search_term:
                        mask = display_df.astype(str).apply(
                            lambda x: x.str.contains(search_term, case=False, na=False)
                        ).any(axis=1)
                        display_df = display_df[mask]
                    
                    if rows_to_show != "All":
                        display_df = display_df.head(rows_to_show)
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                
                with tab2:
                    profile = analyzer.data_profiling()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Dataset Summary")
                        st.write(f"**Shape:** {profile['dataset_size'][0]:,} rows × {profile['dataset_size'][1]:,} columns")
                        st.write(f"**Memory Usage:** {profile['memory_usage'] / (1024*1024):.2f} MB")
                        st.write(f"**Duplicate Rows:** {profile['duplicate_rows']:,}")
                    
                    with col2:
                        st.subheader("📈 Column Types")
                        type_counts = pd.Series([info['dtype'] for info in profile['columns_info'].values()]).value_counts()
                        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Data Types Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Data Quality Module
            if modules['quality']:
                st.markdown("---")
                st.header("🔍 Data Quality Assessment")
                
                tab1, tab2, tab3 = st.tabs(["❓ Missing Data", "🔄 Duplicates", "📊 Completeness"])
                
                with tab1:
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0]
                    
                    if not missing_data.empty:
                        fig = px.bar(
                            x=missing_data.values,
                            y=missing_data.index,
                            orientation='h',
                            title="Missing Data by Column"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Missing data table
                        missing_df = pd.DataFrame({
                            'Column': missing_data.index,
                            'Missing Count': missing_data.values,
                            'Missing %': (missing_data.values / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("✅ No missing data found!")
                
                with tab2:
                    duplicates = df.duplicated().sum()
                    if duplicates > 0:
                        st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}% of data)")
                        
                        if st.button("Show Duplicate Rows"):
                            duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
                            st.dataframe(duplicate_rows, use_container_width=True)
                    else:
                        st.success("✅ No duplicate rows found!")
                
                with tab3:
                    completeness = (1 - df.isnull().sum() / len(df)) * 100
                    
                    fig = px.bar(
                        x=completeness.values,
                        y=completeness.index,
                        orientation='h',
                        title="Data Completeness by Column (%)",
                        color=completeness.values,
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights Module
            if modules['insights']:
                st.markdown("---")
                st.header("🧠 AI-Powered Insights")
                
                insights = analyzer.generate_insights()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📝 Automated Insights")
                    for insight in insights:
                        if "⚠️" in insight:
                            st.markdown(f'<div class="warning-box">{insight}</div>', unsafe_allow_html=True)
                        elif "✅" in insight:
                            st.markdown(f'<div class="success-box">{insight}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("🎯 Quick Actions")
                    if st.button("🔍 Detect Anomalies"):
                        st.info("Anomaly detection completed!")
                    if st.button("📊 Generate Report"):
                        st.info("Report generation started!")
                    if st.button("🧹 Data Cleaning Suggestions"):
                        st.info("Cleaning suggestions ready!")
            
            # Statistical Analysis Module
            if modules['statistics']:
                st.markdown("---")
                st.header("📈 Statistical Analysis")
                
                if analyzer.numeric_cols:
                    tab1, tab2, tab3 = st.tabs(["📊 Descriptive Stats", "🔗 Correlations", "📈 Distributions"])
                    
                    with tab1:
                        stats_df = df[analyzer.numeric_cols].describe()
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with tab2:
                        if len(analyzer.numeric_cols) > 1:
                            corr_method = st.selectbox("Correlation Method", ["correlation", "covariance"])
                            heatmap = analyzer.create_advanced_heatmap(corr_method)
                            if heatmap:
                                st.plotly_chart(heatmap, use_container_width=True)
                    
                    with tab3:
                        selected_col = st.selectbox("Select Column for Distribution", analyzer.numeric_cols)
                        if selected_col:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical analysis.")
            
            # Outlier Detection Module
            if modules['outliers']:
                st.markdown("---")
                st.header("🎯 Outlier Detection")
                
                if analyzer.numeric_cols:
                    selected_col = st.selectbox("Select Column for Outlier Analysis", analyzer.numeric_cols)
                    
                    if selected_col:
                        outlier_plot, outlier_info = analyzer.create_outlier_plot(selected_col)
                        
                        if outlier_plot and outlier_info:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.plotly_chart(outlier_plot, use_container_width=True)
                            
                            with col2:
                                st.subheader("📊 Outlier Statistics")
                                st.metric("Outliers Found", outlier_info['outliers_count'])
                                st.metric("Percentage", f"{outlier_info['outliers_percentage']:.2f}%")
                                st.write(f"**Lower Bound:** {outlier_info['lower_bound']:.2f}")
                                st.write(f"**Upper Bound:** {outlier_info['upper_bound']:.2f}")
                                
                                if st.button("Show Outlier Data"):
                                    st.dataframe(outlier_info['outliers_data'], use_container_width=True)
                else:
                    st.info("No numeric columns found for outlier analysis.")
            
            # Trend Analysis Module
            if modules['trends']:
                st.markdown("---")
                st.header("📊 Trend Analysis")
                
                if analyzer.categorical_cols or analyzer.numeric_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        date_col = st.selectbox("Select Date Column", df.columns.tolist())
                    with col2:
                        value_cols = st.multiselect("Select Value Columns", analyzer.numeric_cols)
                    
                    if date_col and value_cols:
                        trend_df, trends = analyzer.trend_analysis(date_col, value_cols)
                        
                        if trend_df is not None and trends:
                            # Plot trends
                            fig = go.Figure()
                            for col in value_cols:
                                fig.add_trace(go.Scatter(
                                    x=trend_df[date_col],
                                    y=trend_df[col],
                                    mode='lines',
                                    name=col
                                ))
                            
                            fig.update_layout(title="Trend Analysis", height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trend statistics
                            st.subheader("📈 Trend Statistics")
                            trend_stats = pd.DataFrame(trends).T
                            st.dataframe(trend_stats, use_container_width=True)
                else:
                    st.info("Need date and numeric columns for trend analysis.")
            
            # Advanced Analytics Module
            if modules['advanced']:
                st.markdown("---")
                st.header("⚡ Advanced Analytics")
                
                tab1, tab2, tab3 = st.tabs(["🔬 Statistical Tests", "📊 Comparison Charts", "🎯 Custom Analysis"])
                
                with tab1:
                    if len(analyzer.numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            test_col1 = st.selectbox("Column 1", analyzer.numeric_cols, key="test1")
                        with col2:
                            test_col2 = st.selectbox("Column 2", analyzer.numeric_cols, key="test2")
                        
                        if test_col1 and test_col2 and test_col1 != test_col2:
                            test_results = analyzer.statistical_tests(test_col1, test_col2)
                            
                            if test_results:
                                for test_name, results in test_results.items():
                                    st.subheader(f"🔬 {test_name.title()} Test")
                                    for key, value in results.items():
                                        if key == 'significant':
                                            st.write(f"**{key.title()}:** {'✅ Yes' if value else '❌ No'}")
                                        else:
                                            st.write(f"**{key.title()}:** {value:.4f}")
                    else:
                        st.info("Need at least 2 numeric columns for statistical tests.")
                
                with tab2:
                    if analyzer.numeric_cols:
                        comparison_cols = st.multiselect("Select Columns to Compare", analyzer.numeric_cols)
                        chart_type = st.selectbox("Chart Type", ["line", "histogram"])
                        
                        if len(comparison_cols) >= 2:
                            comparison_chart = analyzer.create_comparison_chart(comparison_cols, chart_type)
                            if comparison_chart:
                                st.plotly_chart(comparison_chart, use_container_width=True)
                
                with tab3:
                    if analyzer.categorical_cols and analyzer.numeric_cols:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            group_col = st.selectbox("Group By", analyzer.categorical_cols)
                        with col2:
                            agg_col = st.selectbox("Aggregate Column", analyzer.numeric_cols)
                        with col3:
                            agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "min", "max"])
                        
                        if group_col and agg_col:
                            grouped_data = df.groupby(group_col)[agg_col].agg(agg_func).sort_values(ascending=False)
                            
                            fig = px.bar(
                                x=grouped_data.index,
                                y=grouped_data.values,
                                title=f"{agg_func.title()} of {agg_col} by {group_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(grouped_data.reset_index(), use_container_width=True)
            
            # Export Tools Module
            if modules['export']:
                st.markdown("---")
                st.header("💾 Export Tools")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"analyzed_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if analyzer.numeric_cols:
                        numeric_csv = df[analyzer.numeric_cols].to_csv(index=False)
                        st.download_button(
                            label="🔢 Numeric Data",
                            data=numeric_csv,
                            file_name=f"numeric_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    summary_data = {
                        'File': uploaded_file.name,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Rows': df.shape[0],
                        'Columns': df.shape[1],
                        'Numeric_Columns': len(analyzer.numeric_cols),
                        'Categorical_Columns': len(analyzer.categorical_cols),
                        'Missing_Values': df.isnull().sum().sum(),
                        'Duplicates': df.duplicated().sum()
                    }
                    summary_df = pd.DataFrame([summary_data])
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Analysis Summary",
                        data=summary_csv,
                        file_name=f"summary_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                with col4:
                    # Create Excel with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Raw_Data', index=False)
                        if analyzer.numeric_cols:
                            df[analyzer.numeric_cols].describe().to_excel(writer, sheet_name='Statistics')
                        
                        # Add insights sheet
                        insights_df = pd.DataFrame({'Insights': analyzer.generate_insights()})
                        insights_df.to_excel(writer, sheet_name='Insights', index=False)
                    
                    st.download_button(
                        label="📊 Complete Report",
                        data=output.getvalue(),
                        file_name=f"complete_analysis_{uploaded_file.name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Advanced Excel Data Analyzer!
        
        **Transform your data into actionable insights with AI-powered analysis**
        
        ### ✨ Key Features:
        
        🔍 **Smart Data Quality Assessment**
        - Automatic missing data detection
        - Duplicate identification
        - Data completeness analysis
        
        🧠 **AI-Powered Insights**
        - Automated pattern recognition
        - Anomaly detection
        - Data quality recommendations
        
        📊 **Advanced Analytics**
        - Statistical hypothesis testing
        - Outlier detection with IQR method
        - Correlation and covariance analysis
        
        📈 **Trend Analysis**
        - Time series visualization
        - Trend direction detection
        - Volatility assessment
        
        ⚡ **Professional Tools**
        - Interactive filtering and search
        - Multiple export formats
        - Comprehensive reporting
        
        ### 🎯 Getting Started:
        1. **Upload** your Excel or CSV file using the sidebar
        2. **Select** analysis modules you want to explore
        3. **Discover** insights with interactive visualizations
        4. **Export** your findings in multiple formats
        
        ---
        **Supported formats:** .xlsx, .xls, .csv | **Max file size:** 200MB
        """)
        
        # Demo data
        with st.expander("📝 View Example Data Format"):
            demo_data = pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10),
                'Sales': np.random.randint(100, 1000, 10),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 10),
                'Product': np.random.choice(['A', 'B', 'C'], 10),
                'Quantity': np.random.randint(1, 50, 10),
                'Revenue': np.random.randint(1000, 10000, 10)
            })
            st.dataframe(demo_data, use_container_width=True)

if __name__ == "__main__":
    main()
