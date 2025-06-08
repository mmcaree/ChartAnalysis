import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass

@dataclass
class BreakoutSetup:
    symbol: str
    breakout_probability: float  # 0-1 scale
    setup_quality: float  # 0-1 scale
    prior_move_pct: float
    consolidation_days: int
    volume_trend: str  # "Drying Up", "Stable", "Increasing"
    ma_alignment: float  # How well 10/20 SMAs are aligned and rising
    range_tightness: float  # How tight the recent range is
    last_price: float
    resistance_level: float  # Potential breakout level
    notes: list  # List of specific observations

class BreakoutAnalyzer:
    def __init__(self):
        self.min_prior_move_pct = 30  # Minimum prior move percentage
        self.min_consolidation_days = 3
        self.max_consolidation_days = 90  # 3 months
    
    def identify_prior_move(self, df, lookback_max=120):
        """Identify the most recent significant move up"""
        # Use at most lookback_max days for analysis
        analysis_df = df.tail(lookback_max).copy()
        
        # Calculate rolling max over 20-day periods
        analysis_df['RollingMax'] = analysis_df['High'].rolling(window=20).max()
        
        # Calculate percentage change from 20-day low to that high
        analysis_df['PriorMove'] = 0.0
        
        # Calculate prior moves
        for i in range(20, len(analysis_df)):
            prior_period = analysis_df.iloc[i-20:i]
            period_low = prior_period['Low'].min()
            period_high = prior_period['High'].max()
            move_pct = ((period_high / period_low) - 1) * 100
            analysis_df.iloc[i, analysis_df.columns.get_loc('PriorMove')] = move_pct
        
        # Find the maximum move percentage
        max_move_pct = analysis_df['PriorMove'].max()
        max_move_idx = analysis_df['PriorMove'].idxmax()
        
        # Calculate days since that max move
        days_since_max_move = len(analysis_df) - analysis_df.index.get_loc(max_move_idx)
        
        return {
            'max_move_pct': max_move_pct,
            'days_since_max_move': days_since_max_move,
            'meets_criteria': max_move_pct >= self.min_prior_move_pct
        }
    
    def identify_consolidation(self, df, after_move_days, lookback=60):
        """Identify consolidation pattern after a big move"""
        # Get recent data
        recent_df = df.tail(lookback).copy()
        
        # If not enough data after the move, return empty result
        if after_move_days >= len(recent_df):
            return {'consolidation_found': False}
        
        # Calculate daily ranges as percentage of price
        recent_df['DailyRange'] = (recent_df['High'] - recent_df['Low']) / recent_df['Close'] * 100
        
        # Get post-move data
        post_move_df = recent_df.iloc[-after_move_days:]
        
        # Calculate average range before and after the move
        pre_move_avg_range = recent_df.iloc[:-after_move_days]['DailyRange'].mean()
        post_move_avg_range = post_move_df['DailyRange'].mean()
        
        # Check if ranges are tightening (consolidating)
        range_reduction = (pre_move_avg_range - post_move_avg_range) / pre_move_avg_range
        
        # Calculate narrowing of ranges over the consolidation period
        if len(post_move_df) >= 5:
            early_cons_range = post_move_df.iloc[:len(post_move_df)//2]['DailyRange'].mean()
            late_cons_range = post_move_df.iloc[len(post_move_df)//2:]['DailyRange'].mean()
            range_narrowing = (early_cons_range - late_cons_range) / early_cons_range
        else:
            range_narrowing = 0
        
        # Calculate price range during consolidation
        cons_high = post_move_df['High'].max()
        cons_low = post_move_df['Low'].min()
        price_range_pct = (cons_high / cons_low - 1) * 100
        
        # Calculate higher lows pattern
        higher_lows = True
        for i in range(2, min(5, len(post_move_df))):
            if post_move_df['Low'].iloc[-i] < post_move_df['Low'].iloc[-i-1]:
                higher_lows = False
                break
        
        # Determine if this looks like a consolidation
        consolidation_found = (
            after_move_days >= self.min_consolidation_days and
            after_move_days <= self.max_consolidation_days and
            range_reduction > 0 and
            price_range_pct < 20  # Price isn't swinging too wildly
        )
        
        return {
            'consolidation_found': consolidation_found,
            'days': after_move_days,
            'range_reduction': range_reduction,
            'range_narrowing': range_narrowing,
            'price_range_pct': price_range_pct,
            'higher_lows': higher_lows,
            'range_tightness': 1 - (post_move_avg_range / pre_move_avg_range) if pre_move_avg_range > 0 else 0
        }
    
    def check_ma_alignment(self, df, lookback=20):
        """Check if SMAs are properly aligned and inclining"""
        # Get recent data
        recent_df = df.tail(lookback).copy()
        
        # Ensure we have moving averages
        if 'MA10' not in recent_df.columns or 'MA20' not in recent_df.columns:
            # Calculate them if needed
            recent_df['MA10'] = talib.SMA(recent_df['Close'].values, timeperiod=10)
            recent_df['MA20'] = talib.SMA(recent_df['Close'].values, timeperiod=20)
        
        # Check if MAs are inclining (measure the slope)
        ma10_slope = (recent_df['MA10'].iloc[-1] / recent_df['MA10'].iloc[-5] - 1) * 100
        ma20_slope = (recent_df['MA20'].iloc[-1] / recent_df['MA20'].iloc[-5] - 1) * 100
        
        # Check if price is above MAs
        price = recent_df['Close'].iloc[-1]
        above_ma10 = price > recent_df['MA10'].iloc[-1]
        above_ma20 = price > recent_df['MA20'].iloc[-1]
        
        # Check if 10MA > 20MA (proper alignment)
        ma_aligned = recent_df['MA10'].iloc[-1] > recent_df['MA20'].iloc[-1]
        
        # Calculate days price has been respecting MAs
        ma10_respect_days = 0
        ma20_respect_days = 0
        
        for i in range(1, min(10, len(recent_df))):
            if recent_df['Low'].iloc[-i] >= recent_df['MA10'].iloc[-i] * 0.98:  # Allow small violations
                ma10_respect_days += 1
            if recent_df['Low'].iloc[-i] >= recent_df['MA20'].iloc[-i] * 0.98:
                ma20_respect_days += 1
        
        # Calculate pullback to MA (surfing)
        ma10_proximity = abs(price / recent_df['MA10'].iloc[-1] - 1) * 100
        ma20_proximity = abs(price / recent_df['MA20'].iloc[-1] - 1) * 100
        surfing_mas = (ma10_proximity < 3 or ma20_proximity < 3)
        
        # Create an alignment score (0-1)
        alignment_score = (
            (1 if ma_aligned else 0) * 0.3 +
            (1 if ma10_slope > 0 else 0) * 0.2 +
            (1 if ma20_slope > 0 else 0) * 0.2 +
            (ma10_respect_days / 10) * 0.15 +
            (ma20_respect_days / 10) * 0.15
        )
        
        return {
            'ma_aligned': ma_aligned,
            'ma10_slope': ma10_slope,
            'ma20_slope': ma20_slope,
            'above_ma10': above_ma10,
            'above_ma20': above_ma20,
            'ma10_respect_days': ma10_respect_days,
            'ma20_respect_days': ma20_respect_days,
            'surfing_mas': surfing_mas,
            'alignment_score': alignment_score
        }
    
    def check_volume_pattern(self, df, lookback=30):
        """Check if volume is drying up during consolidation"""
        # Get recent data
        recent_df = df.tail(lookback).copy()
        
        # Calculate average volume for first and second half of period
        first_half = recent_df['Volume'].iloc[:len(recent_df)//2].mean()
        second_half = recent_df['Volume'].iloc[len(recent_df)//2:].mean()
        
        # Check if volume is drying up
        volume_change = (second_half / first_half - 1) * 100
        volume_drying = volume_change < -10
        
        # Look for very low volume in the most recent days (pre-breakout sign)
        recent_vol = recent_df['Volume'].iloc[-3:].mean()
        very_low_recent_vol = recent_vol < recent_df['Volume'].mean() * 0.7
        
        # Check for doji candle in last 3 days
        doji_found = False
        for i in range(1, min(4, len(recent_df))):
            candle = recent_df.iloc[-i]
            body_size = abs(candle['Close'] - candle['Open'])
            range_size = candle['High'] - candle['Low']
            if range_size > 0 and body_size / range_size < 0.3:
                doji_found = True
                break
        
        # Determine volume trend
        if volume_change < -20:
            volume_trend = "Strongly Drying Up"
        elif volume_change < 0:
            volume_trend = "Drying Up"
        elif volume_change < 20:
            volume_trend = "Stable"
        else:
            volume_trend = "Increasing"
        
        return {
            'volume_change': volume_change,
            'volume_drying': volume_drying,
            'very_low_recent_vol': very_low_recent_vol,
            'doji_found': doji_found,
            'volume_trend': volume_trend
        }
    
    def identify_breakout_level(self, df, lookback=30):
        """Identify the potential breakout level (resistance)"""
        # Get recent data
        recent_df = df.tail(lookback).copy()
        
        # Find the highest high in the recent period
        resistance = recent_df['High'].max()
        
        # Find closes near that resistance
        near_resistance = sum(recent_df['High'] > resistance * 0.98) 
        
        # Last price
        last_price = recent_df['Close'].iloc[-1]
        
        # Proximity to breakout (0-1 where 1 is at resistance)
        proximity = last_price / resistance
        
        return {
            'resistance': resistance,
            'last_price': last_price,
            'proximity': proximity,
            'near_resistance_count': near_resistance
        }
    
    def analyze_setup(self, symbol, df):
        """Analyze a stock for high tight flag/breakout setup"""
        try:
            # Initialize breakout_probability at the beginning of the function
            breakout_probability = 0.0
            setup_quality = 0.0
            
            # Add these new analyses
            pre_breakout_volume = self.identify_pre_breakout_volume(df)
            shakeout_pattern = self.identify_shakeout_pattern(df)
            
            # Add check for shallow consolidation at highs
            shallow_consolidation = self.identify_shallow_consolidation(df)
            
            # Check for shakeout days
            shakeout = self.identify_shakeout_day(df)
            
            # Add safety check at the beginning
            if df is None or df.empty:
                return BreakoutSetup(
                    symbol=symbol,
                    breakout_probability=0.0,
                    setup_quality=0.0,
                    prior_move_pct=0.0,
                    consolidation_days=0,
                    volume_trend="Error",
                    ma_alignment=0.0,
                    range_tightness=0.0,
                    last_price=0.0,
                    resistance_level=0.0,
                    notes=["No data available for analysis"]
                )
                
            # Add check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    return BreakoutSetup(
                        symbol=symbol,
                        breakout_probability=0.0,
                        setup_quality=0.0,
                        prior_move_pct=0.0,
                        consolidation_days=0,
                        volume_trend="Error",
                        ma_alignment=0.0,
                        range_tightness=0.0,
                        last_price=df['Close'].iloc[-1] if 'Close' in df.columns and not df.empty else 0.0,
                        resistance_level=0.0,
                        notes=[f"Missing required column: {col}"]
                    )
            
            # Step 1: Look specifically for bull flag pattern
            try:
                bull_flag = self.identify_bull_flag(df)
            except Exception as e:
                # If bull flag detection fails, log it but continue with other analyses
                import traceback
                bull_flag_error = traceback.format_exc()
                bull_flag = {'flag_found': False, 'reason': f"Error in bull flag detection: {str(e)}"}
            
            # Step 2: Also check for prior big move (traditional approach)
            prior_move = self.identify_prior_move(df)
            
            # Get recent data for analysis
            recent_data = df.tail(20).copy()
            last_close = recent_data['Close'].iloc[-1]
            
            # Step 3: Check for post-breakout conditions
            is_post_breakout = False
            breakout_date = None
            
            if len(recent_data) >= 10:
                # Calculate rolling standard deviation of price to identify consolidation followed by expansion
                recent_data['Price_Std_5d'] = recent_data['Close'].rolling(window=5).std() / recent_data['Close']
                
                # Calculate volume ratio to identify volume surge
                recent_data['Vol_Ratio'] = recent_data['Volume'] / recent_data['Volume'].rolling(window=10).mean()
                
                # Look for price expansion after consolidation in the last 5 days
                for i in range(5, min(10, len(recent_data))):
                    # Check if we had consolidation (low std dev) followed by expansion
                    pre_std = recent_data['Price_Std_5d'].iloc[-i-3:-i].mean()
                    post_std = recent_data['Price_Std_5d'].iloc[-i:-i+3].mean()
                    
                    # Check for price surge
                    pre_price = recent_data['Close'].iloc[-i-3]
                    post_price = recent_data['Close'].iloc[-i+3]
                    price_surge = (post_price / pre_price - 1) * 100
                    
                    # Check for volume surge
                    vol_surge = recent_data['Vol_Ratio'].iloc[-i:-i+3].mean() > 1.5
                    
                    # Define breakout as: consolidation → expansion with volume
                    if price_surge > 15 and post_std > pre_std * 1.5 and vol_surge:
                        is_post_breakout = True
                        breakout_date = recent_data.index[-i]
                        break
        
            # Step 4: Check consolidation, MA alignment, volume pattern
            consolidation = self.identify_consolidation(df, prior_move['days_since_max_move'])
            ma_data = self.check_ma_alignment(df)
            volume_data = self.check_volume_pattern(df)
            breakout_data = self.identify_breakout_level(df)
            
            # Collect notes about the setup
            notes = []
            
            # Handle post-breakout case
            if is_post_breakout:
                notes.append(f"ALREADY BROKEN OUT: Stock broke out recently (around {breakout_date.strftime('%Y-%m-%d') if breakout_date else 'recent days'})")
                notes.append(f"Current price (${last_close:.2f}) is after the breakout")
                
                # Check if it's extended after breakout
                week_ago_close = df['Close'].iloc[-6] if len(df) > 6 else df['Close'].iloc[0]
                if last_close / week_ago_close > 1.2:  # 20%+ move in a week
                    notes.append(f"Stock is extended after breakout ({((last_close / week_ago_close) - 1) * 100:.1f}% in a week)")
                
                # Return a setup with post-breakout information
                return BreakoutSetup(
                    symbol=symbol,
                    breakout_probability=0.0,  # Already broken out
                    setup_quality=0.0,  # Not applicable
                    prior_move_pct=prior_move['max_move_pct'],
                    consolidation_days=0,  # Not in consolidation
                    volume_trend="Post-Breakout",
                    ma_alignment=ma_data['alignment_score'],
                    range_tightness=0.0,  # Not applicable
                    last_price=last_close,
                    resistance_level=breakout_data['resistance'],
                    notes=notes
                )
            
            # Prioritize bull flag analysis if found
            if bull_flag['flag_found']:
                notes.append(f"BULL FLAG PATTERN DETECTED - {bull_flag['flag_quality']:.0%} quality")
                notes.append(f"Strong prior move of {bull_flag['flagpole_move_pct']:.1f}%")
                notes.append(f"Flag consolidation for {bull_flag['flag_days']} days")
                
                if bull_flag['range_tightening']:
                    notes.append("Flag showing tightening range (narrowing candles)")
                
                if bull_flag['volume_declining']:
                    notes.append(f"Volume declining during flag formation ({bull_flag['volume_ratio']:.2f}x ratio)")
                
                if bull_flag['ma10_supporting']:
                    notes.append("Price respecting the 10-day moving average")
                
                if bull_flag['potential_breakout']:
                    notes.append(f"Price near breakout level (${bull_flag['resistance_level']:.2f})")
                    notes.append(f"Projected target: ${bull_flag['target_price']:.2f} based on flagpole height")
                
                # Calculate breakout probability for bull flag
                bull_flag_probability = min(0.95, bull_flag['flag_quality'] * 1.2)  # Cap at 95%
                
                # Return the bull flag setup
                return BreakoutSetup(
                    symbol=symbol,
                    breakout_probability=bull_flag_probability,
                    setup_quality=bull_flag['flag_quality'],
                    prior_move_pct=bull_flag['flagpole_move_pct'],
                    consolidation_days=bull_flag['flag_days'],
                    volume_trend="Declining" if bull_flag['volume_declining'] else "Mixed",
                    ma_alignment=1.0 if bull_flag['ma10_supporting'] else 0.5,
                    range_tightness=0.8 if bull_flag['range_tightening'] else 0.4,
                    last_price=bull_flag['last_price'],
                    resistance_level=bull_flag['resistance_level'],
                    notes=notes
                )
            
            # Continue with original analysis if no bull flag found
            if prior_move['max_move_pct'] >= self.min_prior_move_pct:
                notes.append(f"Strong prior move of {prior_move['max_move_pct']:.1f}%")
            
            if consolidation.get('consolidation_found', False):
                notes.append(f"Consolidation for {consolidation['days']} days")
                if consolidation.get('range_narrowing', 0) > 0.1:
                    notes.append("Range is tightening (narrowing ranges)")
                if consolidation.get('higher_lows', False):
                    notes.append("Pattern showing higher lows")
            else:
                notes.append("No clear consolidation pattern detected")
            
            if ma_data.get('ma_aligned', False):
                notes.append("10 SMA above 20 SMA (proper alignment)")
            if ma_data.get('ma10_slope', 0) > 0 and ma_data.get('ma20_slope', 0) > 0:
                notes.append("Both 10 & 20 SMAs are inclining")
            if ma_data.get('surfing_mas', False):
                notes.append("Price surfing/respecting moving averages")
            
            if volume_data.get('volume_drying', False):
                notes.append("Volume drying up during consolidation (good)")
            if volume_data.get('very_low_recent_vol', False):
                notes.append("Very low recent volume (often precedes breakout)")
            if volume_data.get('doji_found', False):
                notes.append("Doji candle found recently (indecision before breakout)")
            
            # Add to notes based on new patterns
            if shallow_consolidation['consolidation_at_highs']:
                notes.append(f"Shallow consolidation near highs detected")
                if shallow_consolidation['is_nr7']:
                    notes.append(f"NR7 pattern (narrowest range of last 7 days)")
                if shallow_consolidation['is_nr4']:
                    notes.append(f"NR4 pattern (narrowest range of last 4 days)")
                if shallow_consolidation['is_inside_day']:
                    notes.append(f"Inside day pattern (today's range inside yesterday's)")
            
            if shakeout['shakeout_found']:
                most_recent = shakeout['most_recent']
                notes.append(f"Potential shakeout day detected {most_recent['days_ago']} days ago")
                notes.append(f"Volume was {most_recent['volume_ratio']:.1f}x average with recovery of {most_recent['recovery_strength']:.1f}%")
                
            # Add new notes for pre-breakout volume pattern
            if pre_breakout_volume.get('pre_breakout_signal', False):
                notes.append(f"PRE-BREAKOUT VOLUME SIGNAL DETECTED!")
                notes.append(f"Volume spike of {pre_breakout_volume['volume_increase_ratio']:.1f}x with {pre_breakout_volume['recent_price_change']:.1f}% price increase")
                notes.append("This volume pattern often precedes major price breakouts")
                
                # Significantly boost probability when we see pre-breakout volume with price action
                breakout_probability += pre_breakout_volume['confidence'] * 0.3  # Add up to 30% to probability
                setup_quality += pre_breakout_volume['confidence'] * 0.25  # Add up to 25% to quality
            
            # Add notes for shakeout pattern
            if shakeout_pattern.get('shakeout_found', False):
                most_recent = shakeout_pattern['most_recent']
                notes.append(f"SHAKEOUT PATTERN DETECTED {most_recent['days_ago']} days ago!")
                notes.append(f"Price made new low then recovered {most_recent['recovery_strength']:.1f}% on {most_recent['volume_ratio']:.1f}x volume")
                notes.append("Shakeouts often occur right before major breakouts (weak hands flushed)")
                
                # Boost probability for shakeout patterns (often very bullish)
                breakout_probability += shakeout_pattern.get('confidence', 0) * 0.25
                setup_quality += shakeout_pattern.get('confidence', 0) * 0.2
            
            # Give more weight to MA alignment when price is near highs
            if ma_data['alignment_score'] > 0.8 and breakout_data['proximity'] > 0.95:
                notes.append("STRONG PRICE POSITION: Trading near highs with excellent MA support")
                breakout_probability += 0.15  # Add 15% to probability for this high-quality setup
                setup_quality += 0.1  # Add 10% to quality
            
            # Calculate overall probability using the factors
            probability_factors = [
                prior_move.get('meets_criteria', False) * 0.15,
                (consolidation.get('consolidation_found', False) * 0.1),
                (consolidation.get('range_tightness', 0) * 0.1),
                (ma_data.get('alignment_score', 0) * 0.3),
                (1 if volume_data.get('volume_trend', "") == "Increasing" else 0) * 0.2,
                (1 if pre_breakout_volume.get('pre_breakout_signal', False) else 0) * 0.25,
                (breakout_data.get('proximity', 0) * 0.15)
            ]
            
            # Add these factors to the existing probability calculation
            additional_probability = sum(probability_factors)
            breakout_probability = max(breakout_probability, additional_probability)
            
            # Setup quality (how well it matches ideal criteria)
            setup_quality = (
                (1 if prior_move.get('max_move_pct', 0) >= 50 else prior_move.get('max_move_pct', 0)/50) * 0.2 +
                (consolidation.get('range_tightness', 0)) * 0.3 +
                (ma_data.get('alignment_score', 0)) * 0.3 +
                (1 if volume_data.get('volume_trend', "") == "Strongly Drying Up" else 
                 0.7 if volume_data.get('volume_trend', "") == "Drying Up" else 
                 0.4 if volume_data.get('volume_trend', "") == "Stable" else 0.1) * 0.2
            )
            
            return BreakoutSetup(
                symbol=symbol,
                breakout_probability=min(1.0, max(0.0, breakout_probability)),
                setup_quality=min(1.0, max(0.0, setup_quality)),
                prior_move_pct=prior_move.get('max_move_pct', 0),
                consolidation_days=consolidation.get('days', 0) if consolidation.get('consolidation_found', False) else 0,
                volume_trend=volume_data.get('volume_trend', "Unknown"),
                ma_alignment=ma_data.get('alignment_score', 0),
                range_tightness=consolidation.get('range_tightness', 0) if consolidation.get('consolidation_found', False) else 0,
                last_price=breakout_data.get('last_price', df['Close'].iloc[-1]),
                resistance_level=breakout_data.get('resistance', df['High'].max()),
                notes=notes
            )
        except Exception as e:
            # Get current exception details
            import traceback
            error_details = traceback.format_exc()
            
            # Return minimal setup with detailed error notes
            return BreakoutSetup(
                symbol=symbol,
                breakout_probability=0.0,
                setup_quality=0.0,
                prior_move_pct=0.0,
                consolidation_days=0,
                volume_trend="Error",
                ma_alignment=0.0,
                range_tightness=0.0,
                last_price=df['Close'].iloc[-1] if not df.empty else 0.0,
                resistance_level=df['High'].max() if not df.empty else 0.0,
                notes=[f"Error analyzing setup: {str(e)}", f"Details: {error_details[:500]}"]
            )
    
    def generate_analysis_text(self, setup):
        """Generate a human-readable analysis text from the setup"""
        # Check if this is a post-breakout analysis
        is_post_breakout = any("ALREADY BROKEN OUT" in note for note in setup.notes)
        is_bull_flag = any("BULL FLAG PATTERN DETECTED" in note for note in setup.notes)
        
        if is_post_breakout:
            analysis_text = f"""
# Post-Breakout Analysis for {setup.symbol}

## Overall Assessment
- **Status**: ALREADY BROKEN OUT
- **Current Price**: ${setup.last_price:.2f}

## Notes
"""
            for note in setup.notes:
                analysis_text += f"- {note}\n"
                
            analysis_text += "\n## Trade Considerations\n"
            analysis_text += "- Stock has already broken out, consider waiting for a pullback\n"
            analysis_text += "- Be cautious of chasing after a significant move\n"
            analysis_text += "- Watch for consolidation after this move for potential future setups\n"
            
            return analysis_text
        
        elif is_bull_flag:
            # Custom analysis for bull flag pattern
            quality_desc = "Excellent" if setup.setup_quality > 0.8 else \
                          "Good" if setup.setup_quality > 0.6 else \
                          "Fair" if setup.setup_quality > 0.4 else "Poor"
                          
            probability_desc = "Very High" if setup.breakout_probability > 0.8 else \
                              "High" if setup.breakout_probability > 0.6 else \
                              "Moderate" if setup.breakout_probability > 0.4 else \
                              "Low" if setup.breakout_probability > 0.2 else "Very Low"
            
            # Extract target price from notes
            target_price = setup.resistance_level * 1.3  # Default estimate
            for note in setup.notes:
                if "Projected target:" in note:
                    try:
                        target_text = note.split("$")[1].split(" ")[0]
                        target_price = float(target_text)
                    except:
                        pass
            
            analysis_text = f"""
# Bull Flag Analysis for {setup.symbol}

## Overall Assessment
- **Pattern**: Bull Flag (High Tight Flag)
- **Breakout Probability**: {probability_desc} ({setup.breakout_probability:.1%})
- **Setup Quality**: {quality_desc} ({setup.setup_quality:.1%})

## Key Metrics
- Flagpole Move: {setup.prior_move_pct:.1f}% 
- Flag Consolidation: {setup.consolidation_days} days
- Current Price: ${setup.last_price:.2f}
- Breakout Level: ${setup.resistance_level:.2f}
- Target Price: ${target_price:.2f}
- Volume Trend: {setup.volume_trend}
- MA Support: {setup.ma_alignment:.1%}
- Range Tightness: {setup.range_tightness:.1%}

## Pattern Notes
"""
            for note in setup.notes:
                analysis_text += f"- {note}\n"
            
            analysis_text += "\n## Trade Plan\n"
            analysis_text += f"- **Entry**: Buy on breakout above ${setup.resistance_level:.2f} with volume\n"
            
            # Calculate stop based on the low of the flag
            stop_price = setup.last_price * 0.95  # Default 5% below
            for note in setup.notes:
                if "Flag showing" in note:
                    # Tighter stop for high-quality flags
                    stop_price = setup.last_price * 0.97
                    
            analysis_text += f"- **Stop Loss**: ${stop_price:.2f}\n"
            analysis_text += f"- **Target**: ${target_price:.2f} (approximately 1x flagpole height)\n"
            
            # Calculate position size based on 2% risk
            risk_per_share = setup.resistance_level - stop_price
            if risk_per_share > 0:
                position_size = int((10000 * 0.02) / risk_per_share)  # Assuming $10k account
                position_value = position_size * setup.resistance_level
                analysis_text += f"- **Position Size**: {position_size} shares (${position_value:.2f}) for $10k account with 2% risk\n"
            
            return analysis_text
        
        # Original analysis for pre-breakout setups
        quality_desc = "Excellent" if setup.setup_quality > 0.8 else \
                      "Good" if setup.setup_quality > 0.6 else \
                      "Fair" if setup.setup_quality > 0.4 else "Poor"
                      
        probability_desc = "Very High" if setup.breakout_probability > 0.8 else \
                          "High" if setup.breakout_probability > 0.6 else \
                          "Moderate" if setup.breakout_probability > 0.4 else \
                          "Low" if setup.breakout_probability > 0.2 else "Very Low"
        
        analysis_text = f"""
# Breakout Analysis for {setup.symbol}

## Overall Assessment
- **Breakout Probability**: {probability_desc} ({setup.breakout_probability:.1%})
- **Setup Quality**: {quality_desc} ({setup.setup_quality:.1%})

## Key Metrics
- Prior Move: {setup.prior_move_pct:.1f}% 
- Consolidation: {setup.consolidation_days} days
- Current Price: ${setup.last_price:.2f}
- Potential Breakout Level: ${setup.resistance_level:.2f}
- Volume Trend: {setup.volume_trend}
- MA Alignment Score: {setup.ma_alignment:.1%}
- Range Tightness: {setup.range_tightness:.1%}

## Notes
"""
    
        for note in setup.notes:
            analysis_text += f"- {note}\n"
    
        if setup.breakout_probability > 0.6:
            analysis_text += "\n## Trade Considerations\n"
            analysis_text += f"- Potential entry on breakout above ${setup.resistance_level:.2f}\n"
            analysis_text += f"- Watch for increased volume on breakout day\n"
            analysis_text += f"- Look for close near high of day on breakout\n"
        
        # Add specific trade plan based on detected patterns
        if any("Inside day pattern" in note for note in setup.notes):
            analysis_text += "\n## Specific Entry Strategy\n"
            analysis_text += f"- Consider buying above today's high (${setup.last_price:.2f}) if tomorrow forms an inside day\n"
            analysis_text += f"- Use low of day as stop loss\n"
            analysis_text += f"- Watch for higher than average volume on breakout\n"
        
        if any("shakeout day detected" in note for note in setup.notes):
            analysis_text += "\n## Shakeout Analysis\n"
            analysis_text += f"- Recent shakeout indicates weak hands being flushed\n"
            analysis_text += f"- Often bullish when price recovers after high-volume selloff\n"
            analysis_text += f"- Consider buying on strength following the shakeout\n"
        
        # Return enhanced analysis
        return analysis_text
    
    def analyze_stock(self, symbol, df):
        """Analyze a stock and return formatted analysis"""
        setup = self.analyze_setup(symbol, df)
        analysis_text = self.generate_analysis_text(setup)
        return analysis_text
    
    def identify_bull_flag(self, df, lookback=60):
        """
        Identify a bull flag pattern with its 5 key components
        """
        # Need enough data to analyze
        if len(df) < 30:
            return {
                'flag_found': False,
                'reason': "Not enough data points"
            }
        
        # Get recent data for analysis
        recent_df = df.tail(lookback).copy()
        
        # Step 1: Find the rising flagpole (strong uptrend)
        # Calculate percentage moves to identify the flagpole
        recent_df['DailyReturn'] = recent_df['Close'].pct_change() * 100
        recent_df['CumReturn5d'] = recent_df['DailyReturn'].rolling(window=5).sum()
        recent_df['CumReturn10d'] = recent_df['DailyReturn'].rolling(window=10).sum()
        recent_df['CumReturn20d'] = recent_df['DailyReturn'].rolling(window=20).sum()
        
        # Fix: Convert NumPy boolean arrays to Python boolean values
        strong_move_5d = (recent_df['CumReturn5d'].rolling(window=15).max() > 30).astype(bool)
        strong_move_10d = (recent_df['CumReturn10d'].rolling(window=20).max() > 30).astype(bool)
        strong_move_20d = (recent_df['CumReturn20d'].max() > 30)
        
        # Identify the flagpole end (where the strong move ended)
        flagpole_found = False
        flagpole_end_idx = None
        flagpole_height = 0
        flagpole_low = 0
        flagpole_high = 0
        flagpole_move_pct = 0
        
        # Try to find the strongest and most recent flagpole
        if strong_move_20d:  # Changed from .any() to a simple boolean check
            # Find all periods with strong moves - convert to Python bool
            strong_periods = recent_df[strong_move_5d | strong_move_10d].index
            
            if len(strong_periods) > 0:
                # Find the most recent strong period
                latest_strong_idx = strong_periods[-1]
                
                # Find the peak within 5 days after this strong period
                potential_end = recent_df.loc[latest_strong_idx:].index[0:10]
                if len(potential_end) > 0:
                    # Get high during this period
                    prices_after_strong = recent_df.loc[potential_end, 'High']
                    flagpole_high_idx = prices_after_strong.idxmax()
                    flagpole_high = prices_after_strong.max()
                    
                    # Look back to find the start of this move (low point)
                    start_window = recent_df.loc[:flagpole_high_idx].index[-20:]
                    if len(start_window) > 0:
                        prices_before_high = recent_df.loc[start_window, 'Low']
                        flagpole_low_idx = prices_before_high.idxmin()
                        flagpole_low = prices_before_high.min()
                        
                        # Calculate the flagpole height
                        flagpole_height = flagpole_high - flagpole_low
                        flagpole_move_pct = (flagpole_high / flagpole_low - 1) * 100
                        
                        # Only consider it a flagpole if the move is significant
                        if flagpole_move_pct >= self.min_prior_move_pct:
                            flagpole_found = True
                            flagpole_end_idx = flagpole_high_idx
    
        if not flagpole_found:
            return {
                'flag_found': False,
                'reason': f"No strong flagpole found (need {self.min_prior_move_pct}%+ move)"
            }
        
        # Step 2: Identify the flag (consolidation after the flagpole)
        # Get data after the flagpole peak
        if flagpole_end_idx in recent_df.index:
            flag_start_idx = recent_df.index.get_loc(flagpole_end_idx)
            flag_df = recent_df.iloc[flag_start_idx:]
        else:
            return {
                'flag_found': False,
                'reason': "Could not locate flagpole end index"
            }
        
        # Need enough days after flagpole to form a flag
        if len(flag_df) < 5:
            return {
                'flag_found': False,
                'reason': "Not enough data after flagpole to form flag"
            }
        
        # Calculate flag metrics
        flag_days = len(flag_df)
        flag_high = flag_df['High'].max()
        flag_low = flag_df['Low'].min()
        last_close = flag_df['Close'].iloc[-1]
        
        # Check if price has retraced appropriately (not too deep, not too shallow)
        # Typically, a flag retraces 1/3 to 2/3 of the flagpole
        retracement_pct = (flagpole_high - flag_low) / flagpole_height * 100
        
        # Flag should not retrace too much (ideally 30-60% of the flagpole)
        if retracement_pct > 80:
            return {
                'flag_found': False,
                'reason': f"Retracement too deep ({retracement_pct:.1f}% of flagpole)"
            }
        
        # Step 3 & 4: Check for declining support and resistance (channel formation)
        # To do this properly, we need to identify swing highs and lows
        swing_highs = []
        swing_lows = []
        
        # Simple swing detection - look for local maxima/minima
        for i in range(1, len(flag_df) - 1):
            # Swing high
            if flag_df['High'].iloc[i] > flag_df['High'].iloc[i-1] and \
               flag_df['High'].iloc[i] > flag_df['High'].iloc[i+1]:
                swing_highs.append((flag_df.index[i], flag_df['High'].iloc[i]))
            
            # Swing low
            if flag_df['Low'].iloc[i] < flag_df['Low'].iloc[i-1] and \
               flag_df['Low'].iloc[i] < flag_df['Low'].iloc[i+1]:
                swing_lows.append((flag_df.index[i], flag_df['Low'].iloc[i]))
        
        # Need at least 2 swing highs and 2 swing lows to form a channel
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                'flag_found': False,
                'reason': f"Not enough swing points to form a channel (highs: {len(swing_highs)}, lows: {len(swing_lows)})"
            }
        
        # Check if the channel is declining (lower highs, lower lows)
        # For bull flag, we want to see downward sloping channel
        declining_highs = swing_highs[-1][1] < swing_highs[0][1]
        declining_lows = swing_lows[-1][1] < swing_lows[0][1]
        
        # Channel slope should be downward or flat
        channel_quality = 0.0
        if declining_highs and declining_lows:
            channel_quality = 1.0  # Perfect channel
        elif declining_highs or declining_lows:
            channel_quality = 0.5  # Partial channel
        
        # Check for tightening range (key characteristic of a flag)
        early_ranges = []
        late_ranges = []
        
        # Calculate price ranges in first half vs second half of flag
        mid_point = len(flag_df) // 2
        for i in range(mid_point):
            early_ranges.append(flag_df['High'].iloc[i] - flag_df['Low'].iloc[i])
        
        for i in range(mid_point, len(flag_df)):
            late_ranges.append(flag_df['High'].iloc[i] - flag_df['Low'].iloc[i])
        
        early_avg_range = sum(early_ranges) / len(early_ranges) if early_ranges else 0
        late_avg_range = sum(late_ranges) / len(late_ranges) if late_ranges else 0
        
        # Flag should show tightening range
        range_tightening = late_avg_range < early_avg_range
        
        # Step 5: Check volume pattern (should be declining during flag formation)
        early_avg_volume = flag_df['Volume'].iloc[:mid_point].mean() if mid_point > 0 else 0
        late_avg_volume = flag_df['Volume'].iloc[mid_point:].mean() if mid_point < len(flag_df) else 0
        
        volume_declining = late_avg_volume < early_avg_volume
        volume_ratio = late_avg_volume / early_avg_volume if early_avg_volume > 0 else 1.0
        
        # Check for very recent volume increase (potential breakout)
        recent_volume = flag_df['Volume'].iloc[-3:].mean()
        recent_volume_surge = recent_volume > late_avg_volume * 1.5
        
        # Look for breakout or proximity to breakout
        # Ideal entry is when price breaks above the upper flag boundary
        last_high = flag_df['High'].iloc[-1]
        last_resistance = max(p[1] for p in swing_highs[-2:])
        
        # Check if price is near resistance or has broken out
        proximity_to_breakout = last_high / last_resistance if last_resistance > 0 else 0
        potential_breakout = proximity_to_breakout > 0.95
        
        # Calculate Moving Average alignment
        ma10_aligned = False
        ma10_supporting = False
        
        if 'MA10' in flag_df.columns:
            ma10 = flag_df['MA10'].iloc[-1]
            ma10_aligned = flag_df['MA10'].iloc[-1] > flag_df['MA10'].iloc[0]
            ma10_supporting = flag_df['Low'].iloc[-5:].min() >= flag_df['MA10'].iloc[-5:].min() * 0.98
        
        # Calculate flag quality score (0-1)
        flag_quality = (
            (1.0 if flag_days >= 5 and flag_days <= 20 else 0.5) * 0.15 +  # Ideal flag duration
            (channel_quality * 0.25) +  # Channel quality
            (1.0 if range_tightening else 0.0) * 0.15 +  # Range tightening
            (1.0 if volume_declining else 0.0) * 0.15 +  # Volume pattern
            (1.0 if ma10_supporting else 0.0) * 0.15 +  # MA support
            (proximity_to_breakout * 0.15)  # Proximity to breakout
        )
        
        # Determine if we have a valid bull flag
        flag_found = (
            flagpole_found and
            flag_days >= 5 and
            channel_quality > 0 and
            retracement_pct <= 80 and
            flag_quality >= 0.5
        )
        
        # Calculate target based on flagpole height
        target_price = last_resistance + flagpole_height
        
        return {
            'flag_found': flag_found,
            'flagpole_move_pct': flagpole_move_pct,
            'flag_days': flag_days,
            'channel_quality': channel_quality,
            'range_tightening': range_tightening,
            'volume_declining': volume_declining,
            'volume_ratio': volume_ratio,
            'recent_volume_surge': recent_volume_surge,
            'ma10_aligned': ma10_aligned,
            'ma10_supporting': ma10_supporting,
            'proximity_to_breakout': proximity_to_breakout,
            'potential_breakout': potential_breakout,
            'flag_quality': flag_quality,
            'last_price': last_close,
            'resistance_level': last_resistance,
            'target_price': target_price,
            'reason': "Valid bull flag pattern detected" if flag_found else "Pattern doesn't meet bull flag criteria"
        }
    
    def identify_shallow_consolidation(self, df, lookback=10):
        """
        Identify a shallow consolidation pattern at/near highs
        This differs from a deeper pullback flag and captures tight ranges near highs
        """
        # Get recent data for analysis
        recent_df = df.tail(lookback).copy()
        
        # Calculate price range as percentage of price
        recent_df['DailyRange'] = (recent_df['High'] - recent_df['Low']) / recent_df['Close'] * 100
        
        # Calculate if current day is NR7 (narrowest range of last 7 days)
        is_nr7 = False
        is_nr4 = False
        if len(recent_df) >= 7:
            today_range = recent_df['DailyRange'].iloc[-1]
            nr7_check = recent_df['DailyRange'].iloc[-7:-1].min()
            is_nr7 = today_range < nr7_check
        
        # Check for NR4
        if len(recent_df) >= 4:
            today_range = recent_df['DailyRange'].iloc[-1]
            nr4_check = recent_df['DailyRange'].iloc[-4:-1].min()
            is_nr4 = today_range < nr4_check
        
        # Check for inside day
        is_inside_day = False
        if len(recent_df) >= 2:
            today_high = recent_df['High'].iloc[-1]
            today_low = recent_df['Low'].iloc[-1]
            yesterday_high = recent_df['High'].iloc[-2]
            yesterday_low = recent_df['Low'].iloc[-2]
            is_inside_day = (today_high <= yesterday_high) and (today_low >= yesterday_low)
        
        # Calculate % from all-time high
        recent_high = recent_df['High'].max()
        current_close = recent_df['Close'].iloc[-1]
        pct_from_high = ((recent_high / current_close) - 1) * 100
        
        # Determine if we have a tight consolidation at highs
        is_tight_at_highs = pct_from_high < 5 and recent_df['DailyRange'].iloc[-3:].mean() < recent_df['DailyRange'].iloc[-10:-3].mean()
        
        return {
            'consolidation_at_highs': is_tight_at_highs,
            'is_nr7': is_nr7,
            'is_nr4': is_nr4,
            'is_inside_day': is_inside_day,
            'pct_from_high': pct_from_high,
            'tight_range_quality': 1.0 if (is_nr7 or is_inside_day) and pct_from_high < 3 else 
                              0.7 if is_tight_at_highs else 0.3
        }

    def identify_shakeout_day(self, df, lookback=10):
        """
        Identify potential shakeout days where price drops but recovers
        These often represent good entry opportunities as weak hands are flushed
        """
        # Get recent data
        recent_df = df.tail(lookback).copy()
        
        shakeout_days = []
        
        # Need at least 3 days of data
        if len(recent_df) < 3:
            return {'shakeout_found': False}
        
        # Loop through recent days
        for i in range(1, min(lookback-1, len(recent_df)-1)):
            day = recent_df.iloc[-i]
            prev_day = recent_df.iloc[-(i+1)]
            next_day = recent_df.iloc[-(i-1)]
            
            # Calculate metrics
            day_range = day['High'] - day['Low']
            day_body = abs(day['Close'] - day['Open'])
            day_vol = day['Volume']
            avg_vol = recent_df['Volume'].iloc[-10:].mean() if len(recent_df) >= 10 else recent_df['Volume'].mean()
            
            # Shakeout characteristics:
            # 1. Above average volume
            # 2. Price closes in lower half of range
            # 3. Next day recovers
            high_volume = day_vol > avg_vol * 1.3
            lower_close = day['Close'] < (day['Low'] + day_range/2)
            next_recovery = next_day['Close'] > day['Close']
            
            if high_volume and lower_close and next_recovery:
                days_ago = i
                shakeout_days.append({
                    'days_ago': days_ago,
                    'date': recent_df.index[-i],
                    'volume_ratio': day_vol / avg_vol,
                    'recovery_strength': (next_day['Close'] / day['Close'] - 1) * 100
                })
        
        return {
            'shakeout_found': len(shakeout_days) > 0,
            'shakeout_days': shakeout_days,
            'most_recent': shakeout_days[0] if shakeout_days else None
        }
    
    def identify_pre_breakout_volume(self, df, lookback=20):
        """
        Identify unusual volume increases that often precede breakouts
        Specifically look for volume expansions near recent highs
        """
        recent_df = df.tail(lookback).copy()
        
        # Need at least 5 days of data
        if len(recent_df) < 5:
            return {'pre_breakout_signal': False}
        
        # Calculate average volume 
        avg_volume = recent_df['Volume'].iloc[:-3].mean()
        
        # Calculate recent volume metrics
        latest_volume = recent_df['Volume'].iloc[-1]
        prev_day_volume = recent_df['Volume'].iloc[-2]
        volume_surge_day = recent_df['Volume'].iloc[-1] > recent_df['Volume'].iloc[-2] * 1.5
    
        # Check if volume has been increasing over the last 3 days
        vol_3d_increasing = all(recent_df['Volume'].iloc[-3:].pct_change().dropna() > 0)
    
        # Check if price is near recent highs (within 5%)
        price = recent_df['Close'].iloc[-1]
        recent_high = recent_df['High'].max()
        near_highs = price > recent_high * 0.95
    
        # Check for significant price gain with volume
        recent_price_change = (recent_df['Close'].iloc[-1] / recent_df['Close'].iloc[-2] - 1) * 100
        bullish_price_action = recent_price_change > 0.5  # Even a small gain with big volume is bullish
    
        # Check if price closed in the upper half of its range (strong)
        latest_bar = recent_df.iloc[-1]
        strong_close = latest_bar['Close'] > (latest_bar['Low'] + (latest_bar['High'] - latest_bar['Low'])/2)
    
        # High volume PLUS price strength is a powerful pre-breakout signal
        volume_spike = latest_volume > avg_volume * 1.8  # Very high relative volume
        moderate_volume_increase = latest_volume > avg_volume * 1.3  # Moderate volume increase
    
        # Multiple conditions for pre-breakout signals
        pre_breakout_conditions = [
            # Condition 1: Very high volume spike with bullish price action
            volume_spike and bullish_price_action,
            
            # Condition 2: Increasing volume for 3 days with price near highs
            vol_3d_increasing and near_highs and moderate_volume_increase,
            
            # Condition 3: Volume higher than previous day with strong price close 
            (latest_volume > prev_day_volume * 1.3) and strong_close and near_highs
        ]
        
        # Calculate confidence based on strength of signals
        confidence = 0.0
        if volume_spike and bullish_price_action and near_highs:
            confidence = 0.9  # Very strong signal
        elif vol_3d_increasing and volume_spike:
            confidence = 0.8  # Strong signal
        elif moderate_volume_increase and bullish_price_action:
            confidence = 0.7  # Good signal
        elif vol_3d_increasing and near_highs:
            confidence = 0.6  # Moderate signal
        
        return {
            'pre_breakout_signal': any(pre_breakout_conditions),
            'volume_increase_ratio': latest_volume / avg_volume,
            'volume_increasing_3d': vol_3d_increasing,
            'recent_price_change': recent_price_change,
            'near_highs': near_highs,
            'strong_close': strong_close,
            'confidence': confidence
        }
    
    def identify_shakeout_pattern(self, df, lookback=10):
        """
        Identify shakeout patterns - false breakdowns followed by recovery
        These often precede major breakouts
        """
        recent_df = df.tail(lookback).copy()
        
        # Need at least 5 days of data
        if len(recent_df) < 5:
            return {'shakeout_found': False}
        
        shakeouts = []
        
        # Loop through days (except first and last 2)
        for i in range(2, len(recent_df) - 2):
            day = recent_df.iloc[i]
            prev_day = recent_df.iloc[i-1]
            next_day = recent_df.iloc[i+1]
            after_next = recent_df.iloc[i+2]
            
            # Conditions for a shakeout:
            # 1. Price makes a new low
            # 2. Volume increases on the low day
            # 3. Price recovers immediately after
            # 4. Price continues higher after recovery
            
            new_low = day['Low'] < recent_df['Low'].iloc[max(0,i-5):i].min()
            increased_vol = day['Volume'] > recent_df['Volume'].iloc[max(0,i-5):i].mean() * 1.3
            quick_recovery = next_day['Close'] > day['Close']
            continued_strength = after_next['Close'] > next_day['Close']
            
            if new_low and increased_vol and quick_recovery:
                recovery_strength = (after_next['Close'] / day['Low'] - 1) * 100
                days_ago = len(recent_df) - i - 2
                
                shakeouts.append({
                    'days_ago': days_ago,
                    'recovery_strength': recovery_strength,
                    'volume_ratio': day['Volume'] / recent_df['Volume'].iloc[max(0,i-5):i].mean(),
                    'continued_up': continued_strength
                })
        
        # Sort by recency
        shakeouts.sort(key=lambda x: x['days_ago'])
        
        return {
            'shakeout_found': len(shakeouts) > 0,
            'shakeouts': shakeouts,
            'most_recent': shakeouts[0] if shakeouts else None,
            'confidence': shakeouts[0]['recovery_strength'] / 10 if shakeouts else 0
        }
