import time
import pandas as pd
from datetime import datetime

class BreakoutAlerter:
    def __init__(self, watchlist=None):
        self.watchlist = watchlist or {}
        self.triggered_alerts = []
    
    def add_to_watchlist(self, symbol, entry_level, stop_level):
        """Add a stock to watchlist with alert levels"""
        self.watchlist[symbol] = {
            'entry_level': entry_level,
            'stop_level': stop_level,
            'added_on': datetime.now().strftime('%Y-%m-%d'),
            'triggered': False
        }
    
    def check_for_breakouts(self, current_data):
        """Check if any stocks in watchlist are breaking out"""
        alerts = []
        
        for symbol, details in self.watchlist.items():
            if symbol in current_data and not details['triggered']:
                current_price = current_data[symbol]['last_price']
                
                if current_price >= details['entry_level']:
                    # Breakout detected
                    alert = {
                        'symbol': symbol,
                        'type': 'BREAKOUT',
                        'price': current_price,
                        'entry_level': details['entry_level'],
                        'stop_level': details['stop_level'],
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    alerts.append(alert)
                    self.triggered_alerts.append(alert)
                    self.watchlist[symbol]['triggered'] = True
        
        return alerts
    
    def generate_entry_plan(self, alert, account_size, risk_pct=0.02):
        """Generate trade plan when alert triggers"""
        entry = alert['entry_level']
        stop = alert['stop_level']
        
        # Calculate risk per share
        risk_per_share = entry - stop
        
        # Calculate position size based on 2% risk
        max_risk_amount = account_size * risk_pct
        position_size = int(max_risk_amount / risk_per_share)
        position_value = position_size * entry
        
        return {
            'symbol': alert['symbol'],
            'entry_price': entry,
            'stop_price': stop,
            'risk_per_share': risk_per_share,
            'position_size': position_size,
            'position_value': position_value,
            'account_percent': (position_value / account_size) * 100,
            'take_profit_1': entry + (risk_per_share * 3),  # 3R target for first partial
            'take_profit_2': entry + (risk_per_share * 5)   # 5R target for second partial
        }