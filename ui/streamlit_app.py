import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import BreakoutScanner
from src.chart_analysis.analyzer import ChartAnalyzer
from src.models.llm_setup import ChartAnalysisLLM
from src.data_ingestion.tc2000_importer import TC2000Importer
from src.chart_analysis.rule_based_analyzer import BreakoutAnalyzer

st.set_page_config(page_title="Chart Analysis Assistant", layout="wide")

@st.cache_resource
def get_scanner():
    account_size = st.session_state.get('account_size', 10000)
    return BreakoutScanner(account_size=account_size)

def main():
    st.title("Chart Analysis Assistant")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        account_size = st.number_input("Account Size ($)", min_value=1000, value=10000, step=1000)
        st.session_state['account_size'] = account_size
        
        st.subheader("Market Condition")
        spy_above = st.checkbox("SPY 10MA > 20MA", value=True)
        qqq_above = st.checkbox("QQQ 10MA > 20MA", value=True)
        
        if not (spy_above or qqq_above):
            st.warning("⚠️ Market conditions not ideal for breakouts")
    
    # Main area tabs
    tab1, tab2, tab3 = st.tabs(["Scan", "Watchlist", "Analysis"])
    
    with tab1:
        st.header("Scan for Breakout Candidates")
        
        col1, col2 = st.columns(2)
        with col1:
            upload_method = st.radio("Input Method", [
                "Upload Multiple CSVs", 
                "Upload Single CSV", 
                "Import from Directory",
                "Enter Symbols Manually"
            ])
        
        importer = TC2000Importer()
        
        if upload_method == "Upload Multiple CSVs":
            uploaded_files = st.file_uploader("Upload multiple chart CSV files", type=["csv"], accept_multiple_files=True)
            
            if uploaded_files:
                st.success(f"Uploaded {len(uploaded_files)} files")
                
                # Button to analyze all files
                if st.button("Analyze All Files"):
                    data_dict = importer.process_multiple_files(uploaded_files)
                    
                    if data_dict:
                        st.success(f"Processed {len(data_dict)} files successfully")
                        analyze_multiple_stocks(data_dict)
                    else:
                        st.error("Error processing files")
        
        elif upload_method == "Upload Single CSV":
            uploaded_file = st.file_uploader("Upload TC2000 CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Process the file
                    df = importer.process_uploaded_file(uploaded_file)
                    
                    # Get symbol from the dataframe
                    symbol = df['Symbol'].iloc[0]
                    
                    # Button to analyze this file
                    if st.button(f"Analyze {symbol}"):
                        analyze_single_stock(symbol, df)
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.exception(e)
        
        elif upload_method == "Import from Directory":
            directory_path = st.text_input("Enter directory path containing CSV files")
            file_pattern = st.text_input("File pattern (e.g. *.csv or *_6MG_*.csv)", value="*.csv")
            
            if directory_path and os.path.isdir(directory_path):
                matching_files = glob.glob(os.path.join(directory_path, file_pattern))
                st.success(f"Found {len(matching_files)} matching files")
                
                if st.button("Analyze All Files in Directory"):
                    data_dict = importer.import_directory(directory_path, file_pattern)
                    
                    if data_dict:
                        st.success(f"Processed {len(data_dict)} files successfully")
                        analyze_multiple_stocks(data_dict)
                    else:
                        st.error("Error processing files")
            elif directory_path:
                st.error(f"Directory not found: {directory_path}")
        
        else:  # Enter Symbols Manually
            symbols_input = st.text_area("Enter symbols (comma or newline separated)")
            if symbols_input:
                symbols = [s.strip() for s in symbols_input.replace(",", "\n").split("\n") if s.strip()]
                st.write(f"Found {len(symbols)} symbols")
                
                if st.button("Scan for Breakouts"):
                    if symbols:
                        with st.spinner("Analyzing charts..."):
                            # Download data for each symbol (this would be your real implementation)
                            data_dict = {}
                            for symbol in symbols:
                                try:
                                    # This is a placeholder - you would fetch real data here
                                    # In a real implementation, you'd use yfinance or your TC2000 importer
                                    pass
                                except Exception as e:
                                    st.error(f"Error fetching data for {symbol}: {str(e)}")
            
                            # Use rule-based analyzer directly
                            analyzer = BreakoutAnalyzer()
            
                            # Placeholder for results - in a real implementation, you'd analyze real data
                            results = []
                            for symbol in symbols:
                                # This is where you'd analyze real data
                                # For now, dummy BreakoutCandidate
                                from src.chart_analysis.analyzer import BreakoutCandidate
                                candidate = BreakoutCandidate(
                                    symbol=symbol,
                                    score=0.8,
                                    entry_price=100.0,
                                    stop_price=95.0,
                                    volume_ratio=1.5,
                                    consolidation_days=5,
                                    ma_surf_quality=0.9,
                                    prior_move_pct=35.0
                                )
                                results.append(candidate)
            
                            if results:
                                display_candidates(results, account_size)
                            else:
                                st.info("No high-quality breakout candidates found")
                    else:
                        st.error("No symbols provided for analysis")
    
    with tab2:
        st.header("Breakout Watchlist")
        
        if 'watchlist' in st.session_state and st.session_state.watchlist:
            watchlist_df = pd.DataFrame.from_dict(st.session_state.watchlist, orient='index')
            watchlist_df['symbol'] = watchlist_df.index
            watchlist_df = watchlist_df[['symbol', 'entry', 'stop', 'score', 'added_on']]
            watchlist_df.sort_values('score', ascending=False, inplace=True)
            
            st.dataframe(watchlist_df)
            
            if st.button("Clear Watchlist"):
                st.session_state.watchlist = {}
                st.experimental_rerun()
        else:
            st.info("No stocks in watchlist. Scan for breakouts first.")
    
    with tab3:
        st.header("Custom Chart Analysis")
        
        symbol = st.text_input("Enter symbol for custom analysis")
        
        if symbol and st.button("Analyze"):
            with st.spinner(f"Analyzing {symbol}..."):
                # In a real implementation, this would fetch real data and analyze
                st.info("This would analyze the chart using your criteria and LLM insights")
                
                # Placeholder for chart
                st.line_chart(pd.DataFrame({'close': [100, 101, 103, 102, 105, 107, 108, 107, 106, 108, 110]}))
                
                # LLM analysis would be included here
                st.write("## Analysis Results")
                st.write("*LLM chart analysis would appear here*")

def analyze_single_stock(symbol, df):
    """Analyze a single stock and display results"""
    try:
        # Initialize analyzer (no LLM needed)
        analyzer = ChartAnalyzer()
        
        # Use rule-based analysis
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyzer.analyze_stock(symbol, df, use_rules=True)
            
            if result:
                st.success(f"Analysis complete for {symbol}")
                st.markdown(result)
                
                # Add to watchlist button
                if st.button(f"Add {symbol} to Watchlist"):
                    if 'watchlist' not in st.session_state:
                        st.session_state.watchlist = {}
                    
                    st.session_state.watchlist[symbol] = {
                        'entry': df['Close'].iloc[-1],  # Default to last close
                        'stop': df['Close'].iloc[-1] * 0.95,  # Default to 5% below
                        'added_on': datetime.now().strftime('%Y-%m-%d'),
                        'score': 0.7  # Default score
                    }
                    st.success(f"Added {symbol} to watchlist")
            else:
                st.warning("No breakout patterns detected")
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        st.exception(e)

def analyze_multiple_stocks(data_dict):
    """Analyze multiple stocks and display results"""
    if not data_dict:
        st.error("No data to analyze")
        return
    
    # Initialize analyzer (no LLM needed)
    analyzer = ChartAnalyzer()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create container for results
    results_container = st.container()
    
    # Process each stock with progress updates
    results = {}
    for i, (symbol, df) in enumerate(data_dict.items()):
        # Update progress
        progress = (i + 1) / len(data_dict)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {symbol}... ({i+1}/{len(data_dict)})")
        
        try:
            # Analyze the stock
            result = analyzer.analyze_stock(symbol, df, use_rules=True)
            results[symbol] = result
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")
            results[symbol] = f"Error: {str(e)}"
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    with results_container:
        st.success(f"Analysis complete for {len(results)} symbols")
        
        for symbol, result in results.items():
            with st.expander(f"{symbol} Analysis"):
                st.markdown(result)
                
                # Add to watchlist button
                if st.button(f"Add {symbol} to Watchlist", key=f"add_{symbol}"):
                    # Extract entry and stop from result (simplified)
                    entry_price = data_dict[symbol]['Close'].iloc[-1]  # Default to last close
                    stop_price = entry_price * 0.95  # Default to 5% below
                    
                    if 'watchlist' not in st.session_state:
                        st.session_state.watchlist = {}
                    
                    st.session_state.watchlist[symbol] = {
                        'entry': entry_price,
                        'stop': stop_price,
                        'added_on': datetime.now().strftime('%Y-%m-%d'),
                        'score': 0.7  # Default score
                    }
                    st.success(f"Added {symbol} to watchlist")

def display_candidates(candidates, account_size):
    """Display breakout candidates with detailed information"""
    st.success(f"Found {len(candidates)} potential breakout candidates")
    
    for candidate in candidates:
        with st.expander(f"{candidate.symbol} - Score: {candidate.score:.1f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Trade Parameters:**")
                st.write(f"Entry: ${candidate.entry_price:.2f}")
                st.write(f"Stop: ${candidate.stop_price:.2f}")
                risk_per_share = candidate.entry_price - candidate.stop_price
                st.write(f"Risk/share: ${risk_per_share:.2f}")
                
                position_size = int((account_size * 0.02) / risk_per_share)
                position_value = position_size * candidate.entry_price
                st.write(f"Position size: {position_size} shares (${position_value:.2f})")
                
            with col2:
                st.write("**Chart Metrics:**")
                st.write(f"Consolidation: {candidate.consolidation_days} days")
                st.write(f"Prior Move: {candidate.prior_move_pct:.1f}%")
                st.write(f"Volume Ratio: {candidate.volume_ratio:.2f}x")
                st.write(f"MA Quality: {candidate.ma_surf_quality:.2f}")
            
            st.write("**Analysis:**")
            # Instead of LLM insights, generate rule-based insights
            st.write(f"Breakout setup identified with {candidate.score*100:.0f}% quality score")
            st.write(f"Stock showing {candidate.consolidation_days} days of consolidation after {candidate.prior_move_pct:.1f}% move")
            st.write(f"Volume pattern suggests accumulation with {candidate.volume_ratio:.1f}x relative volume")
            
            if st.button(f"Add {candidate.symbol} to Watchlist", key=f"add_{candidate.symbol}"):
                if 'watchlist' not in st.session_state:
                    st.session_state.watchlist = {}
                
                st.session_state.watchlist[candidate.symbol] = {
                    'entry': candidate.entry_price,
                    'stop': candidate.stop_price,
                    'added_on': datetime.now().strftime('%Y-%m-%d'),
                    'score': candidate.score
                }
                st.success(f"Added {candidate.symbol} to watchlist")

if __name__ == "__main__":
    main()