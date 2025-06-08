import os
import sys
import argparse
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.llm_setup import ChartAnalysisLLM
from src.chart_analysis.analyzer import ChartAnalyzer
from src.data_ingestion.tc2000_importer import TC2000Importer

def main():
    parser = argparse.ArgumentParser(description='Analyze multiple stock charts from a directory')
    parser.add_argument('--dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--pattern', default='*.csv', help='File pattern to match (e.g. *_6MG_*.csv)')
    parser.add_argument('--output', default='analysis_results.md', help='Output file for results')
    args = parser.parse_args()
    
    print(f"Analyzing stock charts in {args.dir} matching pattern {args.pattern}")
    
    # Initialize components
    importer = TC2000Importer()
    llm = ChartAnalysisLLM(local_model=True)
    analyzer = ChartAnalyzer(llm_model=llm)
    
    # Import data
    data_dict = importer.import_directory(args.dir, args.pattern)
    
    if not data_dict:
        print("No files found or error processing files")
        return
    
    print(f"Found {len(data_dict)} files to analyze")
    
    # Analyze each stock
    results = {}
    for i, (symbol, df) in enumerate(data_dict.items()):
        print(f"Analyzing {symbol}... ({i+1}/{len(data_dict)})")
        
        try:
            result = analyzer.analyze_stock(symbol, df)
            results[symbol] = result
            print(f"  ✓ Analysis complete")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results[symbol] = f"Error: {str(e)}"
    
    # Write results to output file
    with open(args.output, 'w') as f:
        f.write("# Stock Chart Analysis Results\n\n")
        f.write(f"Analyzed {len(results)} stocks on {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        
        for symbol, result in results.items():
            f.write(f"## {symbol}\n\n")
            f.write(result)
            f.write("\n\n---\n\n")
    
    print(f"Analysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()