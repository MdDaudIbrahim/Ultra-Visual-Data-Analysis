import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io
import base64
from datetime import datetime
import warnings
from advanced_analyzer import AdvancedAnalyzer
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Ultra-Visual Excel Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better visuals
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e4057;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .chart-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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

def create_enhanced_metrics_dashboard(df, analyzer):
    """Create enhanced metrics dashboard with visual cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("📊 Total Rows", f"{df.shape[0]:,}", "#667eea"),
        ("📈 Columns", f"{df.shape[1]:,}", "#764ba2"),
        ("🔢 Numeric", len(analyzer.numeric_cols), "#f093fb"),
        ("📝 Categorical", len(analyzer.categorical_cols), "#f5576c"),
        ("💾 Memory", f"{df.memory_usage(deep=True).sum()/(1024*1024):.1f} MB", "#4facfe")
    ]
    
    cols = [col1, col2, col3, col4, col5]
    for i, (label, value, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}, #43cea2); 
                        padding: 1rem; border-radius: 15px; color: white; text-align: center;
                        margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <h3 style="margin: 0; font-size: 1.5rem;">{value}</h3>
                <p style="margin: 0; opacity: 0.9;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

def create_interactive_data_table(df):
    """Create enhanced interactive data table"""
    st.markdown('<div class="section-header">🗃️ Interactive Data Explorer</div>', unsafe_allow_html=True)
    
    # Advanced filtering options
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_term = st.text_input("🔍 Search across all data", key="search_main")
    
    with col2:
        if df.select_dtypes(include=[np.number]).columns.any():
            numeric_filter_col = st.selectbox("🔢 Filter by numeric column", 
                                            ["None"] + df.select_dtypes(include=[np.number]).columns.tolist())
        else:
            numeric_filter_col = "None"
    
    with col3:
        if df.select_dtypes(include=['object']).columns.any():
            category_filter_col = st.selectbox("📂 Filter by category", 
                                             ["None"] + df.select_dtypes(include=['object']).columns.tolist())
        else:
            category_filter_col = "None"
    
    with col4:
        rows_to_show = st.selectbox("📄 Rows to display", [25, 50, 100, 200, "All"])
    
    # Apply filters
    display_df = df.copy()
    
    # Text search
    if search_term:
        mask = display_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
    
    # Numeric filter
    if numeric_filter_col != "None":
        col1_filter, col2_filter = st.columns(2)
        with col1_filter:
            min_val = st.number_input(f"Min {numeric_filter_col}", 
                                    value=float(df[numeric_filter_col].min()))
        with col2_filter:
            max_val = st.number_input(f"Max {numeric_filter_col}", 
                                    value=float(df[numeric_filter_col].max()))
        
        display_df = display_df[
            (display_df[numeric_filter_col] >= min_val) & 
            (display_df[numeric_filter_col] <= max_val)
        ]
    
    # Category filter
    if category_filter_col != "None":
        unique_values = df[category_filter_col].unique()
        selected_values = st.multiselect(f"Select {category_filter_col} values", 
                                       unique_values, default=unique_values)
        display_df = display_df[display_df[category_filter_col].isin(selected_values)]
    
    # Apply row limit
    if rows_to_show != "All":
        display_df = display_df.head(rows_to_show)
    
    # Show filtered results count
    st.info(f"📊 Showing {len(display_df):,} rows out of {len(df):,} total rows")
    
    # Display enhanced table
    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    st.dataframe(display_df, use_container_width=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return display_df

def create_comprehensive_charts_gallery(df, analyzer):
    """Create a comprehensive gallery of charts"""
    st.markdown('<div class="section-header">📊 Visual Analytics Gallery</div>', unsafe_allow_html=True)
    
    # Chart selection tabs
    chart_tabs = st.tabs([
        "📈 Distribution Gallery", "🔗 Relationship Charts", "📊 Category Analysis", 
        "📅 Time Series", "🎯 Advanced Visuals", "🌟 3D & Special Charts", "🎨 Creative Charts"
    ])
    
    # Distribution Gallery
    with chart_tabs[0]:
        if analyzer.numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_cols = st.multiselect("Select columns for distribution analysis", 
                                             analyzer.numeric_cols, 
                                             default=analyzer.numeric_cols[:2])
            
            with col2:
                chart_types = st.multiselect("Chart types", 
                                           ["Histogram", "Box Plot", "Violin Plot", "Density Plot", "Ridge Plot"],
                                           default=["Histogram", "Box Plot"])
            
            for col in selected_cols:
                st.subheader(f"📊 Distribution Analysis: {col}")
                
                chart_cols = st.columns(min(len(chart_types), 4))
                
                for i, chart_type in enumerate(chart_types):
                    with chart_cols[i % len(chart_cols)]:
                        try:
                            if chart_type == "Histogram":
                                fig = px.histogram(df, x=col, title=f"Histogram: {col}", 
                                                 color_discrete_sequence=['#1f77b4'],
                                                 marginal="rug")
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Box Plot":
                                fig = px.box(df, y=col, title=f"Box Plot: {col}",
                                           color_discrete_sequence=['#ff7f0e'],
                                           points="outliers")
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Violin Plot":
                                fig = px.violin(df, y=col, title=f"Violin Plot: {col}",
                                              color_discrete_sequence=['#2ca02c'],
                                              box=True)
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Density Plot":
                                fig = ff.create_distplot([df[col].dropna()], [col], 
                                                       colors=['#d62728'], show_hist=False)
                                fig.update_layout(title=f"Density Plot: {col}", height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif chart_type == "Ridge Plot":
                                # Create a simple ridge-like plot
                                fig = px.histogram(df, x=col, title=f"Ridge Plot: {col}",
                                                 nbins=30, color_discrete_sequence=['#9467bd'])
                                fig.update_traces(opacity=0.7)
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error creating {chart_type}: {str(e)}")
                
                # Statistical summary for each column
                if col in df.columns:
                    col_data = df[col].dropna()
                    stats_cols = st.columns(4)
                    
                    with stats_cols[0]:
                        st.metric("📊 Mean", f"{col_data.mean():.2f}")
                    with stats_cols[1]:
                        st.metric("📊 Median", f"{col_data.median():.2f}")
                    with stats_cols[2]:
                        st.metric("📊 Std Dev", f"{col_data.std():.2f}")
                    with stats_cols[3]:
                        st.metric("📊 Range", f"{col_data.max() - col_data.min():.2f}")
    
    # Relationship Charts
    with chart_tabs[1]:
        if len(analyzer.numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis", analyzer.numeric_cols, key="rel_x")
            with col2:
                y_axis = st.selectbox("Y-axis", analyzer.numeric_cols, key="rel_y")
            with col3:
                color_by = st.selectbox("Color by", ["None"] + analyzer.categorical_cols, key="rel_color")
            
            if x_axis and y_axis and x_axis != y_axis:
                # Create multiple relationship visualizations
                chart_row1 = st.columns(2)
                chart_row2 = st.columns(2)
                
                with chart_row1[0]:
                    # Scatter plot
                    fig = px.scatter(df, x=x_axis, y=y_axis, 
                                   color=color_by if color_by != "None" else None,
                                   title=f"Scatter: {x_axis} vs {y_axis}",
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_row1[1]:
                    # Correlation heatmap
                    corr_data = df[[x_axis, y_axis]].corr()
                    fig = px.imshow(corr_data, title="Correlation Matrix",
                                  color_continuous_scale="RdBu_r", aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_row2[0]:
                    # Joint plot style
                    fig = px.density_heatmap(df, x=x_axis, y=y_axis, 
                                           title=f"Density Heatmap: {x_axis} vs {y_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_row2[1]:
                    # Bubble chart with size
                    if len(analyzer.numeric_cols) >= 3:
                        size_col = st.selectbox("Bubble size", analyzer.numeric_cols, key="bubble_size")
                        fig = px.scatter(df, x=x_axis, y=y_axis, size=size_col,
                                       color=color_by if color_by != "None" else None,
                                       title=f"Bubble Chart: {x_axis} vs {y_axis}")
                        st.plotly_chart(fig, use_container_width=True)
    
    # Category Analysis
    with chart_tabs[2]:
        if analyzer.categorical_cols:
            selected_cat_col = st.selectbox("Select categorical column", analyzer.categorical_cols)
            
            if selected_cat_col:
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    # Value counts bar chart
                    value_counts = df[selected_cat_col].value_counts().head(15)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Value Counts: {selected_cat_col}",
                               color=value_counts.values,
                               color_continuous_scale="viridis")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_cols[1]:
                    # Pie chart
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                               title=f"Distribution: {selected_cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # If we have numeric columns, create grouped analysis
                if analyzer.numeric_cols:
                    numeric_col = st.selectbox("Analyze with numeric column", analyzer.numeric_cols)
                    agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median", "std"])
                    
                    grouped_data = df.groupby(selected_cat_col)[numeric_col].agg(agg_func).sort_values(ascending=False)
                    
                    chart_cols2 = st.columns(2)
                    
                    with chart_cols2[0]:
                        # Grouped bar chart
                        fig = px.bar(x=grouped_data.index, y=grouped_data.values,
                                   title=f"{agg_func.title()} of {numeric_col} by {selected_cat_col}",
                                   color=grouped_data.values,
                                   color_continuous_scale="plasma")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_cols2[1]:
                        # Box plot by category
                        fig = px.box(df, x=selected_cat_col, y=numeric_col,
                                   title=f"Box Plot: {numeric_col} by {selected_cat_col}")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series
    with chart_tabs[3]:
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head())
                    date_cols.append(col)
                except:
                    pass
        
        if date_cols and analyzer.numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                date_col = st.selectbox("Select date column", date_cols)
            with col2:
                value_cols = st.multiselect("Select value columns", analyzer.numeric_cols)
            
            if date_col and value_cols:
                try:
                    df_time = df.copy()
                    df_time[date_col] = pd.to_datetime(df_time[date_col])
                    df_time = df_time.sort_values(date_col)
                    
                    # Multiple time series visualizations
                    chart_cols = st.columns(2)
                    
                    with chart_cols[0]:
                        # Line chart
                        fig = go.Figure()
                        for col in value_cols:
                            fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[col],
                                                   mode='lines+markers', name=col))
                        fig.update_layout(title="Time Series Analysis", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_cols[1]:
                        # Area chart
                        fig = px.area(df_time, x=date_col, y=value_cols[0] if value_cols else None,
                                    title="Area Chart")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional time series analysis
                    if len(value_cols) >= 1:
                        st.subheader("📈 Advanced Time Series Analysis")
                        
                        # Moving averages
                        window_size = st.slider("Moving average window", 3, 30, 7)
                        
                        fig = go.Figure()
                        col = value_cols[0]
                        fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[col],
                                               mode='lines', name=f"Original {col}", opacity=0.6))
                        
                        ma = df_time[col].rolling(window=window_size).mean()
                        fig.add_trace(go.Scatter(x=df_time[date_col], y=ma,
                                               mode='lines', name=f"MA({window_size})", 
                                               line=dict(width=3)))
                        
                        fig.update_layout(title=f"Moving Average Analysis: {col}", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating time series: {str(e)}")
        else:
            st.info("📅 Upload data with date columns for time series analysis")
    
    # Advanced Visuals
    with chart_tabs[4]:
        if analyzer.numeric_cols:
            st.subheader("🎯 Advanced Statistical Visualizations")
            
            # Correlation matrix with enhanced styling
            if len(analyzer.numeric_cols) > 1:
                corr_matrix = df[analyzer.numeric_cols].corr()
                
                # Create enhanced correlation heatmap
                fig = px.imshow(corr_matrix, 
                              title="Enhanced Correlation Matrix",
                              color_continuous_scale="RdBu_r",
                              aspect="auto")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Pairplot-style matrix
                if len(analyzer.numeric_cols) <= 5:  # Limit for performance
                    st.subheader("📊 Scatter Plot Matrix")
                    fig = px.scatter_matrix(df[analyzer.numeric_cols], 
                                          title="Scatter Plot Matrix")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Parallel coordinates plot
            if len(analyzer.numeric_cols) >= 3:
                st.subheader("🎭 Parallel Coordinates Plot")
                
                # Normalize data for better visualization
                normalized_df = df[analyzer.numeric_cols].copy()
                for col in normalized_df.columns:
                    normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
                
                fig = px.parallel_coordinates(normalized_df, 
                                            title="Parallel Coordinates Plot",
                                            color=normalized_df.iloc[:, 0])
                st.plotly_chart(fig, use_container_width=True)
    
    # 3D & Special Charts
    with chart_tabs[5]:
        if len(analyzer.numeric_cols) >= 3:
            st.subheader("🌟 3D Visualizations")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_3d = st.selectbox("X-axis (3D)", analyzer.numeric_cols, key="3d_x")
            with col2:
                y_3d = st.selectbox("Y-axis (3D)", analyzer.numeric_cols, key="3d_y")
            with col3:
                z_3d = st.selectbox("Z-axis (3D)", analyzer.numeric_cols, key="3d_z")
            
            if x_3d and y_3d and z_3d:
                # 3D Scatter plot
                fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                                  color=analyzer.categorical_cols[0] if analyzer.categorical_cols else None,
                                  title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D Surface plot (if data is suitable)
                if len(df) <= 1000:  # Limit for performance
                    try:
                        # Create a grid for surface plot
                        xi = np.linspace(df[x_3d].min(), df[x_3d].max(), 20)
                        yi = np.linspace(df[y_3d].min(), df[y_3d].max(), 20)
                        X, Y = np.meshgrid(xi, yi)
                        
                        # Simple interpolation for Z values
                        from scipy.interpolate import griddata
                        Z = griddata((df[x_3d], df[y_3d]), df[z_3d], (X, Y), method='linear')
                        
                        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
                        fig.update_layout(title=f"3D Surface: {z_3d} over {x_3d} and {y_3d}",
                                        scene=dict(xaxis_title=x_3d, yaxis_title=y_3d, zaxis_title=z_3d),
                                        height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("🔍 3D surface plot requires suitable data distribution")
        
        # Radar chart
        if analyzer.numeric_cols and len(analyzer.numeric_cols) >= 3:
            st.subheader("🎯 Radar Chart")
            
            # Select a few rows for radar chart
            row_indices = st.multiselect("Select rows for radar chart", 
                                       range(min(10, len(df))), 
                                       default=[0] if len(df) > 0 else [])
            
            if row_indices:
                fig = go.Figure()
                
                for idx in row_indices:
                    values = df.iloc[idx][analyzer.numeric_cols].values
                    # Normalize values
                    max_vals = df[analyzer.numeric_cols].max().values
                    min_vals = df[analyzer.numeric_cols].min().values
                    normalized_values = (values - min_vals) / (max_vals - min_vals)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=analyzer.numeric_cols,
                        fill='toself',
                        name=f'Row {idx}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    title="Radar Chart Comparison",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Creative Charts Tab
    with chart_tabs[6]:
        st.subheader("🎨 Creative & Interactive Visualizations")
        
        if analyzer.numeric_cols:
            # Animated charts
            st.markdown("### 🎬 Animated Visualizations")
            
            if len(analyzer.numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Animated scatter plot
                    if analyzer.categorical_cols:
                        animation_col = st.selectbox("Animation column", analyzer.categorical_cols)
                        x_anim = st.selectbox("X-axis (Animated)", analyzer.numeric_cols, key="anim_x")
                        y_anim = st.selectbox("Y-axis (Animated)", analyzer.numeric_cols, key="anim_y")
                        
                        if x_anim and y_anim and animation_col:
                            fig = px.scatter(df, x=x_anim, y=y_anim, color=animation_col,
                                           title="🎬 Animated Scatter Plot",
                                           size_max=20, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sunburst chart
                    if len(analyzer.categorical_cols) >= 2:
                        st.markdown("#### ☀️ Sunburst Chart")
                        cat_cols_for_sunburst = analyzer.categorical_cols[:2]
                        
                        # Create hierarchy for sunburst
                        try:
                            fig = px.sunburst(df, path=cat_cols_for_sunburst,
                                            title="Hierarchical Data View",
                                            height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.info("Sunburst chart requires categorical data with clear hierarchy")
            
            # Gauge Charts
            st.markdown("### 📊 Gauge & KPI Charts")
            gauge_cols = st.columns(min(len(analyzer.numeric_cols), 3))
            
            for i, col in enumerate(analyzer.numeric_cols[:3]):
                with gauge_cols[i]:
                    try:
                        col_mean = df[col].mean()
                        col_max = df[col].max()
                        col_min = df[col].min()
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = col_mean,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Average {col}"},
                            delta = {'reference': col_min},
                            gauge = {
                                'axis': {'range': [col_min, col_max]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [col_min, col_mean], 'color': "lightgray"},
                                    {'range': [col_mean, col_max], 'color': "gray"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': col_max * 0.8}}))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating gauge for {col}: {str(e)}")
            
            # Treemap
            if analyzer.categorical_cols and analyzer.numeric_cols:
                st.markdown("### 🌳 Treemap Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    tree_cat = st.selectbox("Category for Treemap", analyzer.categorical_cols)
                with col2:
                    tree_value = st.selectbox("Value for Treemap", analyzer.numeric_cols)
                
                if tree_cat and tree_value:
                    try:
                        grouped_tree = df.groupby(tree_cat)[tree_value].sum().reset_index()
                        fig = px.treemap(grouped_tree, path=[tree_cat], values=tree_value,
                                       title=f"Treemap: {tree_value} by {tree_cat}")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating treemap: {str(e)}")
            
            # Waterfall Chart (if suitable data)
            if len(analyzer.numeric_cols) >= 1:
                st.markdown("### 💧 Waterfall Chart")
                waterfall_col = st.selectbox("Select column for waterfall", analyzer.numeric_cols)
                
                if waterfall_col:
                    try:
                        # Create a simple waterfall showing cumulative values
                        data_sorted = df[waterfall_col].sort_values().reset_index(drop=True)
                        cumulative = data_sorted.cumsum()
                        
                        fig = go.Figure(go.Waterfall(
                            name="Waterfall",
                            orientation="v",
                            measure=["relative"] * len(data_sorted),
                            x=[f"Step {i+1}" for i in range(len(data_sorted))],
                            y=data_sorted.values,
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                        ))
                        fig.update_layout(title=f"Waterfall Chart: {waterfall_col}", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Waterfall chart works best with incremental data")
            
            # Funnel Chart
            if analyzer.categorical_cols and analyzer.numeric_cols:
                st.markdown("### 🔽 Funnel Chart")
                
                col1, col2 = st.columns(2)
                with col1:
                    funnel_cat = st.selectbox("Category for Funnel", analyzer.categorical_cols, key="funnel_cat")
                with col2:
                    funnel_value = st.selectbox("Value for Funnel", analyzer.numeric_cols, key="funnel_val")
                
                if funnel_cat and funnel_value:
                    try:
                        grouped_funnel = df.groupby(funnel_cat)[funnel_value].sum().sort_values(ascending=False)
                        
                        fig = go.Figure(go.Funnel(
                            y=grouped_funnel.index,
                            x=grouped_funnel.values,
                            textinfo="value+percent initial"
                        ))
                        fig.update_layout(title=f"Funnel: {funnel_value} by {funnel_cat}", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating funnel chart: {str(e)}")
        
        else:
            st.info("📊 Upload data with numeric columns to see creative visualizations!")

def create_enhanced_statistics_section(df, analyzer):
    """Create enhanced statistics with visual elements"""
    st.markdown('<div class="section-header">📈 Enhanced Statistical Dashboard</div>', unsafe_allow_html=True)
    
    if analyzer.numeric_cols:
        # Statistical summary with visual enhancements
        stats_df = df[analyzer.numeric_cols].describe()
        
        # Display as enhanced table
        st.subheader("📊 Comprehensive Statistical Summary")
        st.dataframe(stats_df.style.background_gradient(cmap='viridis').format('{:.2f}'), 
                    use_container_width=True)
        
        # Visual statistics cards
        st.subheader("📋 Key Statistics Cards")
        
        cols = st.columns(len(analyzer.numeric_cols))
        for i, col in enumerate(analyzer.numeric_cols):
            with cols[i % len(cols)]:
                col_data = df[col].dropna()
                
                # Calculate statistics
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                skew_val = stats.skew(col_data)
                
                # Create mini histogram
                fig = px.histogram(df, x=col, nbins=20, 
                                 title=f"{col} Distribution")
                fig.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics metrics
                st.metric(f"Mean", f"{mean_val:.2f}")
                st.metric(f"Median", f"{median_val:.2f}")
                st.metric(f"Std Dev", f"{std_val:.2f}")
                st.metric(f"Skewness", f"{skew_val:.2f}")

def main():
    # Enhanced header
    st.markdown('<p class="main-header">🌟 Ultra-Visual Excel Data Analyzer</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    🚀 <strong>Transform your data into stunning visual insights with advanced analytics</strong> 📊
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h2>📁 Data Upload Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose your Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your data file to start visual analysis"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Initialize analyzer
            analyzer = AdvancedAnalyzer(df)
            
            # Enhanced sidebar options
            st.sidebar.markdown("---")
            st.sidebar.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
                <h3>🎨 Visual Modules</h3>
            </div>
            """, unsafe_allow_html=True)
            
            modules = {
                'dashboard': st.sidebar.checkbox("📊 Enhanced Dashboard", value=True),
                'data_explorer': st.sidebar.checkbox("🗃️ Interactive Data Explorer", value=True),
                'visual_gallery': st.sidebar.checkbox("🎨 Visual Analytics Gallery", value=True),
                'statistics': st.sidebar.checkbox("📈 Enhanced Statistics", value=True),
                'insights': st.sidebar.checkbox("🧠 AI Visual Insights", value=True),
                'export': st.sidebar.checkbox("💾 Export Tools", value=True)
            }
            
            # Enhanced Metrics Dashboard
            if modules['dashboard']:
                create_enhanced_metrics_dashboard(df, analyzer)
            
            # Interactive Data Explorer
            if modules['data_explorer']:
                filtered_df = create_interactive_data_table(df)
            
            # Visual Analytics Gallery
            if modules['visual_gallery']:
                create_comprehensive_charts_gallery(df, analyzer)
            
            # Enhanced Statistics
            if modules['statistics']:
                create_enhanced_statistics_section(df, analyzer)
            
            # AI Visual Insights
            if modules['insights']:
                st.markdown('<div class="section-header">🧠 AI-Powered Visual Insights</div>', unsafe_allow_html=True)
                
                insights = analyzer.generate_insights()
                
                # Display insights as visual cards
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4 style="margin: 0; color: white;">{insight}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate visual recommendations
                st.subheader("🎯 Visualization Recommendations")
                
                recommendations = []
                
                if len(analyzer.numeric_cols) >= 2:
                    recommendations.append("📊 Create scatter plots to explore relationships between numeric variables")
                
                if analyzer.categorical_cols and analyzer.numeric_cols:
                    recommendations.append("📈 Use box plots to compare numeric distributions across categories")
                
                if len(analyzer.numeric_cols) >= 3:
                    recommendations.append("🌟 Try 3D scatter plots for multi-dimensional analysis")
                
                if len(df) > 100:
                    recommendations.append("🔍 Use density plots for large datasets to see patterns")
                
                for rec in recommendations:
                    st.success(rec)
            
            # Export Tools
            if modules['export']:
                st.markdown('<div class="section-header">💾 Enhanced Export Center</div>', unsafe_allow_html=True)
                
                export_cols = st.columns(4)
                
                with export_cols[0]:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"visual_analysis_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                with export_cols[1]:
                    if analyzer.numeric_cols:
                        numeric_csv = df[analyzer.numeric_cols].to_csv(index=False)
                        st.download_button(
                            label="🔢 Numeric Data",
                            data=numeric_csv,
                            file_name=f"numeric_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                
                with export_cols[2]:
                    # Create summary report
                    summary_data = {
                        'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
                        'Value': [df.shape[0], df.shape[1], len(analyzer.numeric_cols), 
                                len(analyzer.categorical_cols), df.isnull().sum().sum()]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Summary Report",
                        data=summary_csv,
                        file_name=f"summary_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                with export_cols[3]:
                    # Create insights report
                    insights_data = pd.DataFrame({'AI_Insights': analyzer.generate_insights()})
                    insights_csv = insights_data.to_csv(index=False)
                    st.download_button(
                        label="🧠 AI Insights",
                        data=insights_csv,
                        file_name=f"insights_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )

    else:
        # Enhanced welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🎨 Visual Data Analysis Features</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        feature_cols = st.columns(3)
        
        features = [
            ("📊 Enhanced Dashboard", "Beautiful metric cards with gradients", "Interactive counters and visual KPIs"),
            ("🗃️ Advanced Data Explorer", "Smart filtering and search", "Real-time data manipulation"),
            ("🎨 Visual Gallery", "20+ chart types", "3D plots, heatmaps, and more"),
            ("📈 Statistical Dashboard", "Visual statistics cards", "Interactive distribution plots"),
            ("🧠 AI Insights", "Smart recommendations", "Automated pattern detection"),
            ("💾 Export Center", "Multiple formats", "Complete analysis packages")
        ]
        
        for i, (title, subtitle, description) in enumerate(features):
            with feature_cols[i % 3]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); 
                            padding: 1.5rem; border-radius: 15px; color: white; 
                            margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <h3 style="margin: 0 0 0.5rem 0;">{title}</h3>
                    <p style="margin: 0 0 0.5rem 0; opacity: 0.9;"><strong>{subtitle}</strong></p>
                    <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Demo data preview
        st.markdown("### 📝 Example of Enhanced Visuals")
        demo_data = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=20),
            'Sales': np.random.randint(100, 1000, 20),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 20),
            'Product': np.random.choice(['A', 'B', 'C'], 20),
            'Revenue': np.random.randint(1000, 10000, 20),
            'Profit': np.random.randint(100, 2000, 20)
        })
        
        # Show a sample visualization
        fig = px.scatter(demo_data, x='Sales', y='Revenue', color='Region', size='Profit',
                        title="🌟 Sample Interactive Visualization")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
