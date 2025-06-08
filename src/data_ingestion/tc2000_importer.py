import pandas as pd
import os
import re
import glob

class TC2000Importer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def import_from_csv(self, file_path):
        """Import exported TC2000 CSV data"""
        df = pd.read_csv(file_path)
        return df
    
    def parse_scan_results(self, scan_file):
        """Parse sector and stock data from TC2000 scan results"""
        # Implementation depends on TC2000 export format
        scan_data = pd.read_csv(scan_file)
        return {
            'symbols': scan_data['Symbol'].tolist(),
            'sectors': scan_data['Sector'].tolist() if 'Sector' in scan_data.columns else []
        }
    
    def get_leading_sectors(self, period='1M'):
        """Extract leading sectors from TC2000 data"""
        # This would need to be customized based on your TC2000 export format
        pass

    def process_uploaded_file(self, uploaded_file):
        """Process an uploaded CSV file and standardize the format"""
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Extract filename without extension as symbol if no Symbol column
        if 'Symbol' not in df.columns:
            symbol = uploaded_file.name.split('.')[0].upper()
            # Extract just the symbol part before underscore if present
            symbol = symbol.split('_')[0]
            # Add Symbol column
            df['Symbol'] = symbol
        
        # Rename columns to standard format if needed
        column_mapping = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close', 
            'Price': 'Close',  # Handle TC2000 "Price" column
            'Volume': 'Volume',
            'Moving Average 10': 'MA10',
            'MA10': 'MA10',
            'Moving Average 20': 'MA20',
            'MA20': 'MA20',
            'Moving Average 50': 'MA50',
            'MA50': 'MA50',
            'Moving Average 200': 'MA200',
            'MA200': 'MA200'
        }
        
        # Apply column mapping for any columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure proper date format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    
    def process_multiple_files(self, uploaded_files):
        """Process multiple uploaded files and return a dictionary of dataframes"""
        data_dict = {}
        
        for uploaded_file in uploaded_files:
            try:
                df = self.process_uploaded_file(uploaded_file)
                symbol = df['Symbol'].iloc[0]
                data_dict[symbol] = df
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return data_dict
    
    def import_directory(self, directory_path, pattern="*.csv"):
        """Import all CSV files from a directory"""
        data_dict = {}
        
        # Get all CSV files in the directory matching the pattern
        csv_files = glob.glob(os.path.join(directory_path, pattern))
        
        for file_path in csv_files:
            try:
                # Read the file
                df = pd.read_csv(file_path)
                
                # Extract symbol from filename
                filename = os.path.basename(file_path)
                symbol = filename.split('_')[0].upper()
                
                # Add Symbol column if not present
                if 'Symbol' not in df.columns:
                    df['Symbol'] = symbol
                
                # Rename columns
                column_mapping = {
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Price': 'Close',  # Handle TC2000 "Price" column
                    'Volume': 'Volume',
                    'Moving Average 10': 'MA10',
                    'MA10': 'MA10',
                    'Moving Average 20': 'MA20',
                    'MA20': 'MA20',
                    'Moving Average 50': 'MA50',
                    'MA50': 'MA50',
                    'Moving Average 200': 'MA200',
                    'MA200': 'MA200'
                }
                
                # Apply column mapping for any columns that exist
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Add to dictionary
                data_dict[symbol] = df
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return data_dict