import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import io
import base64

class AdvancedAnalyzer:
    """Advanced data analysis utilities"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        # All non-numeric columns (for cardinality analysis)
        self.non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
    def outlier_analysis(self, column):
        """Detect outliers using IQR method"""
        if column not in self.numeric_cols:
            return None
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return {
            'outliers_count': len(outliers),
            'outliers_percentage': (len(outliers) / len(self.df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_data': outliers
        }
    
    def create_outlier_plot(self, column):
        """Create outlier visualization"""
        outlier_info = self.outlier_analysis(column)
        if not outlier_info:
            return None
            
        fig = go.Figure()
        
        # Box plot
        fig.add_trace(go.Box(
            y=self.df[column],
            name=column,
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title=f"Outlier Analysis for {column}",
            yaxis_title=column,
            height=400
        )
        
        return fig, outlier_info
    
    def data_profiling(self):
        """Comprehensive data profiling"""
        profile = {
            'dataset_size': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'columns_info': {}
        }
        
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'null_count': self.df[col].isnull().sum(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_count': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df)) * 100
            }
            
            if col in self.numeric_cols:
                col_info.update({
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std()
                })
            
            if col in self.categorical_cols:
                col_info.update({
                    'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None,
                    'least_frequent': self.df[col].value_counts().idxmin() if len(self.df[col].value_counts()) > 0 else None
                })
            
            profile['columns_info'][col] = col_info
        
        return profile
    
    def create_comparison_chart(self, columns, chart_type='line'):
        """Create comparison charts for multiple columns"""
        if len(columns) < 2:
            return None
            
        if chart_type == 'line':
            fig = go.Figure()
            for col in columns:
                if col in self.numeric_cols:
                    fig.add_trace(go.Scatter(
                        y=self.df[col],
                        mode='lines',
                        name=col
                    ))
            
            fig.update_layout(
                title=f"Line Comparison: {', '.join(columns)}",
                height=500
            )
            
        elif chart_type == 'histogram':
            fig = make_subplots(
                rows=len(columns), cols=1,
                subplot_titles=columns
            )
            
            for i, col in enumerate(columns):
                if col in self.numeric_cols:
                    fig.add_trace(
                        go.Histogram(x=self.df[col], name=col),
                        row=i+1, col=1
                    )
            
            fig.update_layout(
                title=f"Histogram Comparison: {', '.join(columns)}",
                height=300 * len(columns)
            )
        
        return fig
    
    def trend_analysis(self, date_col, value_cols):
        """Perform trend analysis"""
        try:
            df_trend = self.df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col])
            df_trend = df_trend.sort_values(date_col)
            
            trends = {}
            for col in value_cols:
                if col in self.numeric_cols:
                    # Calculate rolling average
                    df_trend[f'{col}_ma7'] = df_trend[col].rolling(window=7, min_periods=1).mean()
                    
                    # Calculate trend direction
                    trend_slope = np.polyfit(range(len(df_trend)), df_trend[col].fillna(0), 1)[0]
                    
                    trends[col] = {
                        'slope': trend_slope,
                        'direction': 'Increasing' if trend_slope > 0 else 'Decreasing' if trend_slope < 0 else 'Stable',
                        'volatility': df_trend[col].std()
                    }
            
            return df_trend, trends
            
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")
            return None, None
    
    def create_advanced_heatmap(self, method='correlation'):
        """Create advanced heatmaps"""
        if len(self.numeric_cols) < 2:
            return None
            
        numeric_df = self.df[self.numeric_cols]
        
        if method == 'correlation':
            matrix = numeric_df.corr()
            title = "Correlation Heatmap"
        elif method == 'covariance':
            matrix = numeric_df.cov()
            title = "Covariance Heatmap"
        else:
            return None
        
        fig = px.imshow(
            matrix,
            title=title,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(height=500)
        return fig
    
    def statistical_tests(self, col1, col2):
        """Perform statistical tests"""
        from scipy import stats
        
        results = {}
        
        if col1 in self.numeric_cols and col2 in self.numeric_cols:
            # Correlation test
            corr_coef, p_value = stats.pearsonr(
                self.df[col1].dropna(), 
                self.df[col2].dropna()
            )
            
            results['correlation'] = {
                'coefficient': corr_coef,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # T-test (if applicable)
            try:
                t_stat, t_p_value = stats.ttest_ind(
                    self.df[col1].dropna(),
                    self.df[col2].dropna()
                )
                
                results['t_test'] = {
                    'statistic': t_stat,
                    'p_value': t_p_value,
                    'significant': t_p_value < 0.05
                }
            except:
                pass
        
        return results
    
    def generate_insights(self):
        """Generate automated insights"""
        insights = []
        
        # Data quality insights
        missing_cols = [col for col in self.df.columns if self.df[col].isnull().sum() > 0]
        if missing_cols:
            insights.append(f"⚠️ Found missing data in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}")
        
        # Duplicate insights
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            insights.append(f"🔄 Found {duplicates} duplicate rows ({(duplicates/len(self.df)*100):.1f}% of data)")
        
        # High cardinality insights - check all non-numeric columns
        high_cardinality_cols = []
        for col in self.non_numeric_cols:
            try:
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio > 0.5:
                    high_cardinality_cols.append(str(col))  # Convert to string to ensure join works
            except Exception:
                # Skip columns that cause issues
                continue
        
        if high_cardinality_cols:
            insights.append(f"🎯 High cardinality detected in: {', '.join(high_cardinality_cols)}")
        
        # Skewness insights
        for col in self.numeric_cols:
            try:
                skewness = self.df[col].skew()
                if pd.notna(skewness) and abs(skewness) > 2:
                    insights.append(f"📊 {col} is highly skewed (skewness: {skewness:.2f})")
            except Exception:
                # Skip columns that cause issues with skewness calculation
                continue
        
        # Outlier insights
        for col in self.numeric_cols[:3]:  # Check first 3 numeric columns
            try:
                outlier_info = self.outlier_analysis(col)
                if outlier_info and outlier_info['outliers_percentage'] > 5:
                    insights.append(f"🎯 {col} has {outlier_info['outliers_count']} outliers ({outlier_info['outliers_percentage']:.1f}%)")
            except Exception:
                # Skip columns that cause issues with outlier analysis
                continue
        
        return insights
