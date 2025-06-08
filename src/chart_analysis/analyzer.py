import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass

@dataclass
class BreakoutCandidate:
    symbol: str
    score: float
    entry_price: float
    stop_price: float
    volume_ratio: float
    consolidation_days: int
    ma_surf_quality: float  # How well it's surfing MAs
    prior_move_pct: float

class ChartAnalyzer:
    def __init__(self, llm_model=None):
        self.llm = llm_model  # Can be None now
    
    def calculate_moving_averages(self, df):
        """Check for existing MAs or calculate if needed"""
        # Map standard column names
        column_mapping = {
            'Moving Average 10': 'MA10',
            'Moving Average 20': 'MA20',
            'Moving Average 50': 'MA50',
            'Moving Average 200': 'MA200'
        }
        
        # Use existing MAs if available, otherwise calculate
        for orig_name, std_name in column_mapping.items():
            if orig_name in df.columns:
                df[std_name] = df[orig_name]
            else:
                # Extract the timeperiod from the standard name
                timeperiod = int(std_name[2:])
                df[std_name] = talib.SMA(df['Close'].values, timeperiod=timeperiod)
        
        return df
    
    def identify_consolidation(self, df, lookback=30, threshold=0.03):
        """Identify consolidation patterns"""
        # Calculate daily ranges as percentage of price
        df['DailyRange'] = (df['High'] - df['Low']) / df['Close']
        
        # Rolling standard deviation to find tight ranges
        df['RangeStdDev'] = df['DailyRange'].rolling(window=5).std()
        
        # Find sequences of narrow range days (below threshold)
        narrow_ranges = df['RangeStdDev'] < threshold
        
        # Count consecutive narrow range days
        consolidation_periods = []
        current_count = 0
        
        for i, is_narrow in enumerate(narrow_ranges.values[-lookback:]):
            if is_narrow:
                current_count += 1
            elif current_count >= 3:  # Minimum 3 days of consolidation
                consolidation_periods.append({
                    'start_idx': i - current_count,
                    'end_idx': i - 1,
                    'days': current_count
                })
                current_count = 0
            else:
                current_count = 0
                
        return consolidation_periods
    
    def check_ma_surf(self, df, lookback=10):
        """Check if price is surfing above moving averages"""
        above_ma10 = sum(df['Close'][-lookback:] >= df['MA10'][-lookback:]) / lookback
        above_ma20 = sum(df['Close'][-lookback:] >= df['MA20'][-lookback:]) / lookback
        
        ma10_ma20_aligned = sum(df['MA10'][-lookback:] >= df['MA20'][-lookback:]) / lookback
        
        # Score from 0-1 how well it's surfing MAs
        surf_score = (above_ma10 * 0.4) + (above_ma20 * 0.3) + (ma10_ma20_aligned * 0.3)
        
        return surf_score
    
    def measure_prior_move(self, df, consolidation_start):
        """Measure percentage move prior to consolidation"""
        # Look back up to 60 days before consolidation
        lookback = min(60, consolidation_start)
        if lookback <= 5:  # Not enough data
            return 0
            
        pre_consolidation = df.iloc[consolidation_start-lookback:consolidation_start]
        if len(pre_consolidation) < 5:
            return 0
            
        lowest = pre_consolidation['Low'].min()
        highest = pre_consolidation['High'].max()
        
        return (highest / lowest - 1) * 100  # Percentage gain
    
    def check_breakout_potential(self, df, consolidation_period):
        """Check for breakout potential from consolidation"""
        if not consolidation_period:
            return None
            
        # Get the last consolidation period
        last_cons = consolidation_period[-1]
        cons_end = last_cons['end_idx']
        
        # Last price
        current_price = df['Close'].iloc[-1]
        
        # High of consolidation
        cons_high = df['High'].iloc[cons_end-last_cons['days']:cons_end+1].max()
        
        # Volume increase
        recent_volume = df['Volume'].iloc[-5:].mean()
        cons_volume = df['Volume'].iloc[cons_end-last_cons['days']:cons_end+1].mean()
        volume_ratio = recent_volume / cons_volume
        
        # Check if we're near breakout level
        proximity_to_breakout = current_price / cons_high
        
        return {
            'consolidation_days': last_cons['days'],
            'cons_high': cons_high,
            'proximity': proximity_to_breakout,
            'volume_ratio': volume_ratio
        }
    
    def analyze_stock(self, symbol, df, use_rules=True):
        """Analyze a single stock for breakout patterns"""
        # Ensure data is properly formatted
        df = self.calculate_moving_averages(df)
        
        # Use either rule-based or LLM analysis
        if use_rules:
            return self.rule_based_analysis(symbol, df)
        else:
            # Create a summary of the data for LLM analysis
            summary = self._create_summary_for_llm(symbol, df)
            
            # Get sector info if available
            sector_info = df.get('Sector', None)
            if sector_info is None:
                sector_info = "Technology"  # Default sector if not provided
            
            # Use LLM for pattern recognition
            llm_analysis = self.llm.analyze_chart(summary, sector_info)
            
            return llm_analysis

    def _create_summary_for_llm(self, symbol, df):
        """Create a summary of stock data for LLM analysis"""
        # Get last N days of data
        recent_data = df.tail(30).copy()
        
        # Calculate recent price changes
        if len(recent_data) > 0:
            last_close = recent_data['Close'].iloc[-1]
            first_close = recent_data['Close'].iloc[0]
            price_change = ((last_close - first_close) / first_close) * 100
            
            # Calculate average volume
            avg_volume = recent_data['Volume'].mean()
            
            # Check if price is above moving averages
            above_ma10 = last_close > recent_data['MA10'].iloc[-1] if 'MA10' in recent_data.columns else "Unknown"
            above_ma20 = last_close > recent_data['MA20'].iloc[-1] if 'MA20' in recent_data.columns else "Unknown"
            above_ma50 = last_close > recent_data['MA50'].iloc[-1] if 'MA50' in recent_data.columns else "Unknown"
            above_ma200 = last_close > recent_data['MA200'].iloc[-1] if 'MA200' in recent_data.columns else "Unknown"
            
            # Create summary text
            summary = f"""
            Symbol: {symbol}
            Latest Price: ${last_close:.2f}
            30-Day Change: {price_change:.2f}%
            Average Volume: {avg_volume:.0f}
            
            Moving Averages:
            - Above 10-day MA: {above_ma10}
            - Above 20-day MA: {above_ma20}
            - Above 50-day MA: {above_ma50}
            - Above 200-day MA: {above_ma200}
            
            Recent Price Action (last 5 days):
            {recent_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string(index=False)}
            """
            
            return summary
        else:
            return f"No data available for {symbol}"
    
    def llm_enhanced_analysis(self, candidate, df, market_condition):
        """Use LLM to enhance the analysis with more nuanced observations"""
        # Prepare chart data for LLM
        recent_data = df.tail(30).to_dict(orient='records')
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_chart(recent_data, market_condition)
        
        return {
            'candidate': candidate,
            'llm_insights': llm_analysis
        }
    
    def batch_analyze_stocks(self, data_dict, max_concurrent=1, use_rules=True):
        """Analyze multiple stocks in batch mode
        
        Args:
            data_dict: Dictionary of {symbol: dataframe} pairs
            max_concurrent: Maximum number of concurrent analyses (for future threading)
            use_rules: Whether to use rule-based analysis instead of LLM
            
        Returns:
            Dictionary of {symbol: analysis_result} pairs
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                # Ensure data is properly formatted
                df = self.calculate_moving_averages(df)
                
                if use_rules:
                    # Use rule-based analysis (no LLM needed)
                    result = self.rule_based_analysis(symbol, df)
                else:
                    # Create a summary of the data for LLM analysis
                    summary = self._create_summary_for_llm(symbol, df)
                    
                    # Get sector info if available
                    sector_info = "Technology"  # Default sector
                    
                    # Use LLM for pattern recognition
                    result = self.llm.analyze_chart(summary, sector_info)
                
                # Store the result
                results[symbol] = result
                
            except Exception as e:
                results[symbol] = f"Error analyzing {symbol}: {str(e)}"
        
        return results
    
    def rule_based_analysis(self, symbol, df):
        """Perform rule-based analysis instead of using LLM"""
        from src.chart_analysis.rule_based_analyzer import BreakoutAnalyzer
        
        # Ensure we have moving averages
        df = self.calculate_moving_averages(df)
        
        # Create the rule-based analyzer
        breakout_analyzer = BreakoutAnalyzer()
        
        # Get the analysis
        analysis = breakout_analyzer.analyze_stock(symbol, df)
        
        return analysis
