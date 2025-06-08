import pandas as pd
import argparse
import os
import time
from src.chart_analysis.analyzer import ChartAnalyzer
from src.alert_system.alerter import BreakoutAlerter
from src.data_ingestion.tc2000_importer import TC2000Importer
from src.chart_analysis.rule_based_analyzer import BreakoutAnalyzer  # Import the rule-based analyzer

class BreakoutScanner:
    def __init__(self, account_size=10000):
        self.account_size = account_size
        # Remove LLM dependency
        self.analyzer = ChartAnalyzer()  # No LLM parameter
        self.rule_analyzer = BreakoutAnalyzer()  # Add rule-based analyzer
        self.alerter = BreakoutAlerter()
        self.importer = TC2000Importer()
        
    def check_market_condition(self, spy_data, qqq_data):
        """Check if market conditions are favorable for breakout trades"""
        # Check if SPY/QQQ 10MA is above 20MA
        spy_df = pd.DataFrame(spy_data)
        qqq_df = pd.DataFrame(qqq_data)
        
        spy_df = self.analyzer.calculate_moving_averages(spy_df)
        qqq_df = self.analyzer.calculate_moving_averages(qqq_df)
        
        spy_ma_aligned = spy_df['MA10'].iloc[-1] > spy_df['MA20'].iloc[-1]
        qqq_ma_aligned = qqq_df['MA10'].iloc[-1] > qqq_df['MA20'].iloc[-1]
        
        return {
            'favorable': spy_ma_aligned or qqq_ma_aligned,
            'spy_ma_aligned': spy_ma_aligned,
            'qqq_ma_aligned': qqq_ma_aligned,
            'note': "Market conditions favorable for breakouts" if (spy_ma_aligned or qqq_ma_aligned) else 
                   "Caution: Both SPY and QQQ have 10MA below 20MA"
        }
    
    def scan_for_setups(self, symbols, market_data):
        """Scan provided symbols for breakout setups"""
        candidates = []
        analysis_results = {}
        
        market_condition = self.check_market_condition(
            market_data.get('SPY', []),
            market_data.get('QQQ', [])
        )
        
        if not market_condition['favorable']:
            print("Warning: Market conditions not ideal for breakout trades")
        
        for symbol in symbols:
            if symbol in market_data:
                df = pd.DataFrame(market_data[symbol])
                
                # Use rule-based analysis instead of LLM
                setup = self.rule_analyzer.analyze_setup(symbol, df)
                analysis_text = self.rule_analyzer.generate_analysis_text(setup)
                
                # Only consider high-quality setups
                if setup.breakout_probability > 0.6 and setup.setup_quality > 0.5:
                    # Convert setup to a format compatible with the alerter
                    from src.chart_analysis.analyzer import BreakoutCandidate
                    
                    candidate = BreakoutCandidate(
                        symbol=setup.symbol,
                        score=setup.breakout_probability * 100,  # Convert to 0-100 scale
                        entry_price=setup.resistance_level,  # Use resistance as entry
                        stop_price=setup.last_price * 0.95,  # Simple default stop
                        volume_ratio=1.5,  # Default
                        consolidation_days=setup.consolidation_days,
                        ma_surf_quality=setup.ma_alignment,
                        prior_move_pct=setup.prior_move_pct
                    )
                    
                    # Store candidate and analysis
                    candidates.append({
                        'candidate': candidate,
                        'analysis_text': analysis_text,
                        'setup': setup
                    })
                    
                    # Add to watchlist for alerts
                    self.alerter.add_to_watchlist(
                        symbol, 
                        setup.resistance_level,  # Entry at resistance/breakout level
                        setup.last_price * 0.95  # Simple default stop
                    )
                
                # Store all analyses for reference
                analysis_results[symbol] = analysis_text
        
        # Sort by breakout probability
        candidates.sort(key=lambda x: x['setup'].breakout_probability, reverse=True)
        return candidates, analysis_results
    
    def generate_report(self, candidates):
        """Generate a readable report of breakout candidates"""
        report = "# Breakout Candidates Report\n\n"
        
        for idx, candidate_data in enumerate(candidates, 1):
            c = candidate_data['candidate']
            setup = candidate_data['setup']
            
            report += f"## {idx}. {c.symbol} (Probability: {setup.breakout_probability:.1%})\n\n"
            report += f"- Current Price: ${setup.last_price:.2f}\n"
            report += f"- Potential Breakout Level: ${setup.resistance_level:.2f}\n"
            report += f"- Suggested Stop: ${c.stop_price:.2f}\n"
            report += f"- Risk Per Share: ${setup.resistance_level - c.stop_price:.2f}\n"
            report += f"- Consolidation: {setup.consolidation_days} days\n"
            report += f"- Prior Move: {setup.prior_move_pct:.1f}%\n"
            report += f"- Volume Trend: {setup.volume_trend}\n"
            report += f"- MA Alignment: {setup.ma_alignment:.1%}\n\n"
            
            # Position sizing
            risk_per_share = setup.resistance_level - c.stop_price
            if risk_per_share > 0:
                position_size = int((self.account_size * 0.02) / risk_per_share)
                position_value = position_size * setup.resistance_level
                
                report += f"### Position Plan\n"
                report += f"- Shares: {position_size}\n"
                report += f"- Position Value: ${position_value:.2f}\n"
                report += f"- Account %: {(position_value/self.account_size)*100:.1f}%\n"
                report += f"- Take Profit 1 (3R): ${setup.resistance_level + (risk_per_share*3):.2f}\n"
                report += f"- Take Profit 2 (5R): ${setup.resistance_level + (risk_per_share*5):.2f}\n\n"
            
            report += f"### Detailed Analysis\n"
            for note in setup.notes:
                report += f"- {note}\n"
            report += "\n---\n\n"
        
        return report
    
    def main(self, scan_file=None, symbols=None):
        """Main execution function"""
        if scan_file:
            scan_results = self.importer.parse_scan_results(scan_file)
            symbols = scan_results['symbols']
        
        if not symbols:
            print("No symbols provided for analysis")
            return
        
        # This would be replaced with actual data fetching
        market_data = {}  # Fetch market data for symbols
        
        candidates, analysis_results = self.scan_for_setups(symbols, market_data)
        report = self.generate_report(candidates)
        
        # Save report
        with open("breakout_report.md", "w") as f:
            f.write(report)
        
        # Save all analyses
        with open("all_analyses.md", "w") as f:
            f.write("# Complete Analysis Results\n\n")
            for symbol, analysis in analysis_results.items():
                f.write(f"## {symbol}\n\n")
                f.write(analysis)
                f.write("\n\n---\n\n")
        
        print(f"Found {len(candidates)} potential breakout candidates")
        print("Report saved to breakout_report.md")
        print("All analyses saved to all_analyses.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breakout Scanner")
    parser.add_argument("--account", type=float, default=10000, help="Account size")
    parser.add_argument("--scan", type=str, help="TC2000 scan results file")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    
    args = parser.parse_args()
    
    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip() for s in args.symbols.split(",")]
    
    scanner = BreakoutScanner(account_size=args.account)
    scanner.main(scan_file=args.scan, symbols=symbols_list)