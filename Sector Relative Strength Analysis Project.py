import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV', 
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}

BENCHMARK = 'SPY'  # S&P500

# DATA FETCHING MODULE 
class DataFetcher:
    def __init__(self, period='1y'):
        self.period = period
        
    def fetch_data(self, tickers):
        """Fetch historical data for given tickers"""
        try:
            data = yf.download(tickers, period=self.period, group_by='ticker')
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def get_sector_data(self):
        """Fetch data for all sectors and benchmark"""
        all_tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
        return self.fetch_data(all_tickers)

# RELATIVE STRENGTH ANALYSIS
class RelativeStrengthAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def calculate_returns(self, ticker, periods=[1, 5, 10, 20, 60, 120, 252]):
        """Calculate returns for various periods"""
        if ticker in self.data.columns.levels[0]:
            prices = self.data[ticker]['Close'].dropna()
        else:
            prices = self.data['Close'][ticker].dropna() if 'Close' in self.data.columns else self.data[ticker].dropna()
            
        returns = {}
        for period in periods:
            if len(prices) >= period:
                returns[f'{period}d'] = ((prices.iloc[-1] / prices.iloc[-period]) - 1) * 100
            else:
                returns[f'{period}d'] = np.nan
        return returns
    
    def calculate_relative_strength(self, sector_ticker, benchmark_ticker=BENCHMARK):
        """Calculate relative strength vs benchmark"""
        try:
            if sector_ticker in self.data.columns.levels[0]:
                sector_prices = self.data[sector_ticker]['Close'].dropna()
            else:
                sector_prices = self.data['Close'][sector_ticker].dropna()
                
            if benchmark_ticker in self.data.columns.levels[0]:
                benchmark_prices = self.data[benchmark_ticker]['Close'].dropna()
            else:
                benchmark_prices = self.data['Close'][benchmark_ticker].dropna()
            
            # Align dates
            common_dates = sector_prices.index.intersection(benchmark_prices.index)
            sector_aligned = sector_prices.loc[common_dates]
            benchmark_aligned = benchmark_prices.loc[common_dates]
            
            # Calculate relative strength ratio
            rs_ratio = sector_aligned / benchmark_aligned
            
            # Normalize to 100 at start
            rs_normalized = (rs_ratio / rs_ratio.iloc[0]) * 100
            
            return rs_normalized
        except Exception as e:
            st.error(f"Error calculating relative strength for {sector_ticker}: {e}")
            return pd.Series()
    
    def get_sector_rankings(self):
        
        rankings = []
        
        for sector, ticker in SECTOR_ETFS.items():
            sector_returns = self.calculate_returns(ticker)
            benchmark_returns = self.calculate_returns(BENCHMARK)
            
            relative_performance = {}
            for period in sector_returns.keys():
                if not np.isnan(sector_returns[period]) and not np.isnan(benchmark_returns[period]):
                    relative_performance[period] = sector_returns[period] - benchmark_returns[period]
                else:
                    relative_performance[period] = np.nan
            
            ranking_data = {
                'Sector': sector,
                'Ticker': ticker,
                **sector_returns,
                **{f'vs_SPY_{k}': v for k, v in relative_performance.items()}
            }
            rankings.append(ranking_data)
        
        return pd.DataFrame(rankings)

# VISUALIZATION 
class Visualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def plot_relative_strength_chart(self, sectors_to_plot=None):
        """Create relative strength chart"""
        if sectors_to_plot is None:
            sectors_to_plot = list(SECTOR_ETFS.keys())[:6]  # Top 6 by default
            
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, sector in enumerate(sectors_to_plot):
            if sector in SECTOR_ETFS:
                ticker = SECTOR_ETFS[sector]
                rs_data = self.analyzer.calculate_relative_strength(ticker)
                
                if not rs_data.empty:
                    fig.add_trace(go.Scatter(
                        x=rs_data.index,
                        y=rs_data.values,
                        mode='lines',
                        name=sector,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{sector}</b><br>Date: %{{x}}<br>RS Ratio: %{{y:.2f}}<extra></extra>'
                    ))
        
        # Add benchmark line at 100
        fig.add_hline(y=100, line_dash="dash", line_color="black", 
                     annotation_text="Benchmark (SPY)", annotation_position="top right")
        
        fig.update_layout(
            title='Sector Relative Strength vs S&P 500',
            xaxis_title='Date',
            yaxis_title='Relative Strength (Normalized to 100)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_performance_heatmap(self, rankings_df):
        
        # Select relative performance columns
        rel_cols = [col for col in rankings_df.columns if 'vs_SPY' in col]
        heatmap_data = rankings_df[['Sector'] + rel_cols].set_index('Sector')
        
        # Clean column names
        heatmap_data.columns = [col.replace('vs_SPY_', '').replace('d', ' days') for col in heatmap_data.columns]
        
        fig = px.imshow(
            heatmap_data.T,
            aspect='auto',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Sector Relative Performance Heatmap (vs S&P 500)'
        )
        
        fig.update_layout(
            xaxis_title='Sectors',
            yaxis_title='Time Period',
            height=400
        )
        
        return fig
    
    def plot_sector_comparison(self, selected_sectors):
        
        fig = go.Figure()
        
        # Add S&P 500 as baseline
        spy_data = self.analyzer.data[BENCHMARK]['Close'].dropna()
        spy_normalized = (spy_data / spy_data.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=spy_normalized.index,
            y=spy_normalized.values,
            mode='lines',
            name='S&P 500 (SPY)',
            line=dict(color='black', width=3, dash='dash')
        ))
        
        colors = px.colors.qualitative.Set1
        for i, sector in enumerate(selected_sectors):
            if sector in SECTOR_ETFS:
                ticker = SECTOR_ETFS[sector]
                try:
                    sector_data = self.analyzer.data[ticker]['Close'].dropna()
                    sector_normalized = (sector_data / sector_data.iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=sector_normalized.index,
                        y=sector_normalized.values,
                        mode='lines',
                        name=sector,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                except Exception as e:
                    st.warning(f"Could not plot {sector}: {e}")
        
        fig.update_layout(
            title='Normalized Price Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Starting at 100)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig

# STREAMLIT DASHBOARD 
def main():
    st.set_page_config(
        page_title="Sector Relative Strength Analysis",
        layout="wide"
    )
    
    st.title("Sector Relative Strength Analysis Dashboard")
    st.markdown("Analyze US sector performance vs S&P 500 with interactive visualizations")
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")
    
    # Time period selection
    period_options = {
        '3 Months': '3mo',
        '6 Months': '6mo', 
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y'
    }
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        options=list(period_options.keys()),
        index=2  # Default to 1 year
    )
    
    # Sector selection for detailed analysis
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors for Detailed Analysis",
        options=list(SECTOR_ETFS.keys()),
        default=list(SECTOR_ETFS.keys())[:4]
    )
    
    # Load data button
    if st.sidebar.button("Load/Refresh Data"):
        with st.spinner("Fetching market data..."):
            # Initialize components
            fetcher = DataFetcher(period=period_options[selected_period])
            data = fetcher.get_sector_data()
            
            if data is not None:
                analyzer = RelativeStrengthAnalyzer(data)
                visualizer = Visualizer(analyzer)
                
                # Store in session state
                st.session_state.analyzer = analyzer
                st.session_state.visualizer = visualizer
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
    
    # Check if data is loaded
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        visualizer = st.session_state.visualizer
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Relative Strength Analysis")
            
            # Relative strength chart
            rs_chart = visualizer.plot_relative_strength_chart(selected_sectors)
            st.plotly_chart(rs_chart, use_container_width=True)
            
            # Price comparison chart
            st.subheader("Price Performance Comparison")
            price_chart = visualizer.plot_sector_comparison(selected_sectors)
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            st.subheader("Sector Rankings")
            
            # Get rankings
            rankings_df = analyzer.get_sector_rankings()
            
            # Display top performers
            st.write("**Top Performers (1 Month vs SPY)**")
            top_performers = rankings_df.nlargest(5, 'vs_SPY_20d')[['Sector', 'vs_SPY_20d']]
            top_performers['vs_SPY_20d'] = top_performers['vs_SPY_20d'].round(2)
            st.dataframe(top_performers, hide_index=True)
            
            # Display underperformers
            st.write("**Underperformers (1 Month vs SPY)**")
            underperformers = rankings_df.nsmallest(5, 'vs_SPY_20d')[['Sector', 'vs_SPY_20d']]
            underperformers['vs_SPY_20d'] = underperformers['vs_SPY_20d'].round(2)
            st.dataframe(underperformers, hide_index=True)
            
            # Recent performance metrics
            st.write("**Quick Stats**")
            total_sectors = len(SECTOR_ETFS)
            outperforming = len(rankings_df[rankings_df['vs_SPY_20d'] > 0])
            underperforming = total_sectors - outperforming
            
            st.metric("Sectors Outperforming", outperforming, f"{outperforming}/{total_sectors}")
            st.metric("Sectors Underperforming", underperforming, f"{underperforming}/{total_sectors}")
        
        # Full width heatmap
        st.subheader("Performance Heatmap")
        heatmap = visualizer.plot_performance_heatmap(rankings_df)
        st.plotly_chart(heatmap, use_container_width=True)
        
        # Detailed rankings table
        st.subheader("Detailed Sector Performance Table")
        
        # Format the dataframe for display
        display_df = rankings_df.copy()
        numeric_cols = [col for col in display_df.columns if col not in ['Sector', 'Ticker']]
        for col in numeric_cols:
            display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
    else:
        # Initial state
        st.info("Please click 'Load/Refresh Data' in the sidebar to begin analysis")
        
        # Show sector information
        st.subheader("Sector ETFs Tracked")
        sector_info = pd.DataFrame(list(SECTOR_ETFS.items()), columns=['Sector', 'ETF Ticker'])
        st.dataframe(sector_info, hide_index=True)
        

if __name__ == "__main__":
    main()

# ===== REQUIREMENTS.TXT =====
"""
Create a requirements.txt file with:

yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
streamlit>=1.28.0
"""

# ===== USAGE INSTRUCTIONS =====
"""
To run this project:

1. Install requirements:
   pip install -r requirements.txt

2. Run the dashboard:
   streamlit run sector_analysis.py

3. Open browser to http://localhost:8501

The tool analyzes 11 major US sector ETFs:
- XLK (Technology)
- XLV (Healthcare) 
- XLF (Financials)
- XLY (Consumer Discretionary)
- XLC (Communication Services)
- XLI (Industrials)
- XLP (Consumer Staples)
- XLE (Energy)
- XLU (Utilities)
- XLRE (Real Estate)
- XLB (Materials)

All compared against SPY (S&P 500 ETF) as the benchmark.
"""