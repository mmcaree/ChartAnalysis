import yfinance as yf
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
from datetime import datetime
import threading

class StockDataExporterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Data Exporter")
        self.root.geometry("1200x1000")
        self.root.minsize(600, 500)
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a more modern theme
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Symbol input section
        ttk.Label(main_frame, text="Enter Stock Symbols (one per line, paste from TC2000):", 
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Symbol text area with scrollbar
        self.symbols_text = scrolledtext.ScrolledText(main_frame, height=10)
        self.symbols_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Export Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Time period selection
        ttk.Label(options_frame, text="Time Period:").grid(row=0, column=0, sticky="w", pady=5)
        self.period_var = tk.StringVar(value="6mo")
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        period_dropdown = ttk.Combobox(options_frame, textvariable=self.period_var, values=period_options, width=10)
        period_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Interval selection
        ttk.Label(options_frame, text="Interval:").grid(row=0, column=2, sticky="w", pady=5)
        self.interval_var = tk.StringVar(value="1d")
        interval_options = ["1d", "5d", "1wk", "1mo", "3mo"]
        interval_dropdown = ttk.Combobox(options_frame, textvariable=self.interval_var, values=interval_options, width=10)
        interval_dropdown.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # TC2000 List Tag
        ttk.Label(options_frame, text="TC2000 List Tag:").grid(row=1, column=0, sticky="w", pady=5)
        self.tag_var = tk.StringVar(value="")
        self.tag_options = ["", "1MG", "3MG", "6MG", "CUSTOM"]
        tag_dropdown = ttk.Combobox(options_frame, textvariable=self.tag_var, values=self.tag_options, width=10)
        tag_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        tag_dropdown.bind("<<ComboboxSelected>>", self.on_tag_selected)
        
        # Custom tag entry (initially hidden)
        self.custom_tag_var = tk.StringVar()
        self.custom_tag_entry = ttk.Entry(options_frame, textvariable=self.custom_tag_var, width=10)
        self.custom_tag_entry.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.custom_tag_entry.grid_remove()  # Hide initially
        
        # Filename suffix options
        ttk.Label(options_frame, text="Filename Format:").grid(row=2, column=0, sticky="w", pady=5)
        suffix_frame = ttk.Frame(options_frame)
        suffix_frame.grid(row=2, column=1, columnspan=3, sticky="w", pady=5)
        
        self.suffix_var = tk.StringVar(value="full_format")
        ttk.Radiobutton(suffix_frame, text="Full Format (e.g. AAPL_6MG_6MO_060725)", 
                        variable=self.suffix_var, value="full_format").pack(anchor="w")
        ttk.Radiobutton(suffix_frame, text="Period + Date (e.g. AAPL_6MO_060725)", 
                        variable=self.suffix_var, value="period_date").pack(anchor="w")
        ttk.Radiobutton(suffix_frame, text="Tag + Date (e.g. AAPL_6MG_060725)", 
                        variable=self.suffix_var, value="tag_date").pack(anchor="w")
        ttk.Radiobutton(suffix_frame, text="Date Only (e.g. AAPL_060725)", 
                        variable=self.suffix_var, value="date_only").pack(anchor="w")
        ttk.Radiobutton(suffix_frame, text="No Suffix (e.g. AAPL.csv)", 
                        variable=self.suffix_var, value="none").pack(anchor="w")
        
        # Output directory selection
        dir_frame = ttk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(dir_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "data"))
        self.output_dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Progress display
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.progress_text = scrolledtext.ScrolledText(self.progress_frame, height=5, state="disabled")
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Clear", command=self.clear_form, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Export Data", command=self.start_export, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_tag_selected(self, event):
        """Show custom tag entry when CUSTOM is selected"""
        if self.tag_var.get() == "CUSTOM":
            self.custom_tag_entry.grid()
        else:
            self.custom_tag_entry.grid_remove()
        
    def select_output_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if dir_path:
            self.output_dir_var.set(dir_path)
            
    def clear_form(self):
        self.symbols_text.delete(1.0, tk.END)
        self.progress_text.config(state="normal")
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state="disabled")
        self.progress_bar["value"] = 0
        self.status_var.set("Ready")
    
    def log_progress(self, message):
        self.progress_text.config(state="normal")
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state="disabled")
        self.root.update_idletasks()
    
    def start_export(self):
        # Get symbols from text area
        symbols_text = self.symbols_text.get(1.0, tk.END).strip()
        if not symbols_text:
            messagebox.showerror("Error", "Please enter at least one stock symbol")
            return
        
        # Parse symbols (handling various formats)
        symbols = []
        for line in symbols_text.splitlines():
            line = line.strip()
            if line:
                # Remove any quotes, commas, or other formatting
                clean_symbol = line.replace('"', '').replace(',', '').strip()
                if clean_symbol:
                    symbols.append(clean_symbol)
        
        if not symbols:
            messagebox.showerror("Error", "No valid symbols found")
            return
        
        # Confirm export
        if not messagebox.askyesno("Confirm Export", f"Export data for {len(symbols)} symbols?"):
            return
        
        # Start export in a separate thread to avoid freezing the UI
        threading.Thread(target=self.export_data, args=(symbols,), daemon=True).start()
    
    def get_tag(self):
        """Get the tag value (either from dropdown or custom entry)"""
        tag = self.tag_var.get()
        if tag == "CUSTOM":
            tag = self.custom_tag_var.get()
        return tag
    
    def clean_csv_file(self, file_path):
        """Remove the problematic 2nd row from a CSV file"""
        try:
            # Read all lines from the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip the 2nd line (index 1) if there are at least 2 lines
            if len(lines) >= 2:
                with open(file_path, 'w') as f:
                    f.write(lines[0])  # Write header
                    f.writelines(lines[2:])  # Write everything except the 2nd line
                return True
            return False
        except Exception as e:
            self.log_progress(f"Error cleaning CSV file: {str(e)}")
            return False
    
    def export_data(self, symbols):
        period = self.period_var.get()
        interval = self.interval_var.get()
        suffix_type = self.suffix_var.get()
        output_dir = self.output_dir_var.get()
        tag = self.get_tag()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date for filename
        today = datetime.now().strftime("%m%d%y")
        
        # Update UI
        self.status_var.set(f"Exporting {len(symbols)} symbols...")
        self.progress_bar["maximum"] = len(symbols)
        self.progress_bar["value"] = 0
        
        # Process each symbol
        success_count = 0
        error_count = 0
        
        for i, symbol in enumerate(symbols):
            try:
                self.log_progress(f"Downloading {symbol}...")
                
                # Download data
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if data.empty:
                    self.log_progress(f"Warning: No data available for {symbol}")
                    error_count += 1
                    continue
                
                # Calculate moving averages
                data['MA10'] = data['Close'].rolling(window=10).mean()
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()
                
                # Add symbol column
                data['Symbol'] = symbol
                
                # Convert period to shorthand (e.g., 6mo -> 6MO)
                period_short = period.replace('mo', 'MO').replace('y', 'Y')
                
                # Create filename based on selected suffix option
                if suffix_type == "full_format" and tag:
                    filename = f"{symbol}_{tag}_{period_short}_{today}.csv"
                elif suffix_type == "tag_date" and tag:
                    filename = f"{symbol}_{tag}_{today}.csv"
                elif suffix_type == "period_date":
                    filename = f"{symbol}_{period_short}_{today}.csv"
                elif suffix_type == "date_only":
                    filename = f"{symbol}_{today}.csv"
                else:  # "none"
                    filename = f"{symbol}.csv"
                
                # Save to CSV with clean, consistent format
                file_path = os.path.join(output_dir, filename)
                
                # Select and reorder columns to ensure consistent format
                columns_to_export = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
                                    'MA10', 'MA20', 'MA50', 'MA200', 'Symbol']
                
                # Ensure all columns exist (create empty ones if needed)
                for col in columns_to_export:
                    if col not in data.columns and col != 'Date':
                        data[col] = None
                
                # Export only the columns we want in the exact order we want
                export_data = data.reset_index()[columns_to_export]
                
                # Save without the extra header lines
                export_data.to_csv(file_path, index=False)
                
                # Clean the CSV file to remove the problematic 2nd row
                self.clean_csv_file(file_path)
                
                self.log_progress(f"Saved {symbol} data to {filename}")
                success_count += 1
                
            except Exception as e:
                self.log_progress(f"Error downloading {symbol}: {str(e)}")
                error_count += 1
            
            # Update progress bar
            self.progress_bar["value"] = i + 1
            self.root.update_idletasks()
        
        # Update status
        if error_count == 0:
            self.status_var.set(f"Export complete! {success_count} symbols exported successfully.")
        else:
            self.status_var.set(f"Export finished with {error_count} errors. {success_count} symbols exported successfully.")
        
        self.log_progress(f"Export completed. Files saved to {output_dir}")
        
        # Show completion message
        messagebox.showinfo("Export Complete", 
                           f"Exported {success_count} symbols to {output_dir}\n" +
                           (f"Errors: {error_count}" if error_count > 0 else ""))

if __name__ == "__main__":
    root = tk.Tk()
    app = StockDataExporterGUI(root)
    root.mainloop()