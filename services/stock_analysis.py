import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from app.models.stock_models import StockResponse, ChartData, AnalysisMetrics
from app.services.data_cache import StockDataCache
import logging
import json
from time import time
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)

class StockAnalyzer:
    REQUIRED_HIGHER_SWINGS = 4  # Number of required higher highs and lows
    REQUIRED_WEEKS = 6  # Time window for swing analysis
    TARGET_DAYS_BETWEEN_SWINGS = 4  # Target average days between swings
    TARGET_SWING_AMPLITUDE = 8  # Target swing amplitude percentage

    def __init__(self):
        self.debug_msgs: List[str] = []
        self._reported_conditions = set()  # Track reported conditions to avoid duplicates
        
        # Volume analysis thresholds
        self.VOLUME_SPIKE_THRESHOLD = 2.0  # Volume > 2x average
        self.VOLUME_TREND_WINDOW = 5  # Days to analyze volume trend
        self.VOLUME_DELTA_THRESHOLD = 0.2  # 20% difference for significant delta
        self.VOLUME_DELTA_LOOKBACK = 10  # Days to check for average volume delta
        self.HIGH_VOLUME_THRESHOLD = 1.5  # 50% above average for high volume
        
    def _add_debug_msg(self, msg: str, condition_key: str = None) -> None:
        """
        Add a debug message, optionally tracking it as a condition to avoid duplicates.
        
        Args:
            msg: The message to add
            condition_key: Optional unique key to track this condition
        """
        if condition_key:
            if condition_key in self._reported_conditions:
                return  # Skip if we've already reported this condition
            self._reported_conditions.add(condition_key)
        
        self.debug_msgs.append(msg)

    def identify_higher_lows_highs(self, series: pd.Series) -> bool:
        """Helper to determine if swing lows/highs are making higher lows/highs."""
        vals = series.dropna().values
        if len(vals) < 2:
            return False
        return all(x < y for x, y in zip(vals, vals[1:]))

    def count_higher_lows_highs(self, series: pd.Series) -> int:
        """Count how many higher highs or higher lows occur sequentially."""
        vals = series.dropna().values
        count = 0
        prev = -np.inf
        for val in vals:
            if val > prev:
                count += 1
                prev = val
        return count

    def avg_days_between_swings(self, dates: pd.Index) -> Optional[float]:
        """Calculate average days between swings."""
        if len(dates) < 2:
            return None
        diffs = pd.Series(dates).diff().dropna()
        return diffs.dt.days.mean()

    def avg_pct_diff_between_swings(self, series: pd.Series) -> Optional[float]:
        """Calculate average percentage difference between consecutive swings."""
        vals = series.dropna().values
        if len(vals) < 2:
            return None
        pct_diffs = []
        for i in range(1, len(vals)):
            prev, curr = vals[i-1], vals[i]
            if prev != 0:
                pct_diff = abs((curr - prev) / prev * 100)
                pct_diffs.append(pct_diff)
        return round(np.mean(pct_diffs), 2) if pct_diffs else None

    def check_recent_swing_conditions(self, df: pd.DataFrame, swing_lows: pd.Series, swing_highs: pd.Series) -> bool:
        """Check if we have required number of higher highs and lows in time window."""
        cutoff_date = df.index[-1] - pd.Timedelta(weeks=self.REQUIRED_WEEKS)
        recent_lows = swing_lows[swing_lows.index >= cutoff_date]
        recent_highs = swing_highs[swing_highs.index >= cutoff_date]
        
        higher_lows_count = self.count_higher_lows_highs(recent_lows)
        higher_highs_count = self.count_higher_lows_highs(recent_highs)
        
        return (higher_lows_count >= self.REQUIRED_HIGHER_SWINGS and 
                higher_highs_count >= self.REQUIRED_HIGHER_SWINGS)

    def check_ma_alignment(self, row: pd.Series) -> bool:
        """Check if moving averages are properly aligned."""
        return row['MA20'] > row['MA50'] > row['MA100']

    def check_volume_with_highs(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if volume increases with higher highs."""
        # Get local highs (price peaks)
        highs = df[df['High'] == df['High'].rolling(window=5, center=True).max()]
        if len(highs) < 2:
            return False, 0.0
            
        # Calculate volume trend during highs
        high_volumes = highs['Volume'].values
        volume_changes = np.diff(high_volumes) / high_volumes[:-1]
        avg_volume_change = np.mean(volume_changes)
        
        # Return True if average volume change is positive
        return avg_volume_change > 0, avg_volume_change

    def check_recent_low_volume(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if recent low is followed by high buying volume."""
        # Find most recent low
        recent_lows = df[df['Low'] == df['Low'].rolling(window=5, center=True).min()]
        if recent_lows.empty:
            return False, 0.0
            
        last_low_idx = recent_lows.index[-1]
        # Get next 3 days after low
        post_low_period = df.loc[last_low_idx:].head(4)
        
        if len(post_low_period) < 2:
            return False, 0.0
            
        # Check if buying volume increased after low
        avg_buying_volume = post_low_period['Buying_Volume'].mean()
        normal_buying_volume = df['Buying_Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = avg_buying_volume / normal_buying_volume if normal_buying_volume > 0 else 0
        
        return volume_ratio > self.HIGH_VOLUME_THRESHOLD, volume_ratio

    def _load_data_from_cache(self, symbol: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Load stock data from cache and convert to DataFrame"""
        logger.info(f"Loading cached data for {symbol}")
        cache = StockDataCache()
        data, last_updated = cache.get_cached_data(symbol)
        
        if not data or not data.get('records'):
            logger.warning(f"No cached data found for {symbol}")
            return pd.DataFrame(), "No cached data available"
            
        # Convert records to DataFrame
        df = pd.DataFrame(data['records'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        # Rename columns to match expected format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        logger.info(f"Loaded {len(df)} records for {symbol}")
        return df, None

    def analyze_volume_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze various volume patterns and return findings."""
        # Calculate buying and selling volume
        df['Buying_Volume'] = df['Volume'].where(df['Close'] > df['Open'], 0)
        df['Selling_Volume'] = df['Volume'].where(df['Close'] <= df['Open'], 0)
        
        # Volume spikes
        df['Volume_Spike'] = df['Volume'] > (df['Volume_MA20'] * self.VOLUME_SPIKE_THRESHOLD)
        
        # Volume trends
        df['Buying_Volume_Trend'] = df['Buying_Volume'].rolling(window=self.VOLUME_TREND_WINDOW).mean()
        df['Selling_Volume_Trend'] = df['Selling_Volume'].rolling(window=self.VOLUME_TREND_WINDOW).mean()
        
        # Volume Delta
        df['Volume_Delta'] = df['Buying_Volume'] - df['Selling_Volume']
        df['Volume_Delta_MA'] = df['Volume_Delta'].rolling(window=self.VOLUME_TREND_WINDOW).mean()
        
        # Analyze patterns (using last VOLUME_TREND_WINDOW days)
        recent_data = df.tail(self.VOLUME_TREND_WINDOW)
        
        patterns = {
            'volume_spikes': [],
            'volume_reversal': False,
            'bearish_divergence': False,
            'bullish_reversal': False,
            'volume_delta_strength': 0,
            'positive_volume_delta': False,
            'high_volume_after_low': False,
            'volume_confirms_highs': False,
            'volume_metrics': {
                'delta_avg': 0.0,
                'post_low_volume_ratio': 0.0,
                'high_volume_change': 0.0
            }
        }
        
        # Check average volume delta in lookback period
        recent_delta = df['Volume_Delta'].tail(self.VOLUME_DELTA_LOOKBACK).mean()
        patterns['positive_volume_delta'] = recent_delta > 0
        patterns['volume_metrics']['delta_avg'] = recent_delta
        
        # Check volume after recent low
        high_vol_after_low, vol_ratio = self.check_recent_low_volume(df)
        patterns['high_volume_after_low'] = high_vol_after_low
        patterns['volume_metrics']['post_low_volume_ratio'] = vol_ratio
        
        # Check if volume increases with higher highs
        vol_confirms, vol_change = self.check_volume_with_highs(df)
        patterns['volume_confirms_highs'] = vol_confirms
        patterns['volume_metrics']['high_volume_change'] = vol_change
        
        # Check for volume spikes
        spike_days = recent_data[recent_data['Volume_Spike']].index
        patterns['volume_spikes'] = spike_days.tolist()
        
        # Check for volume reversal (decreasing selling + increasing buying)
        selling_trend = recent_data['Selling_Volume_Trend'].diff().mean()
        buying_trend = recent_data['Buying_Volume_Trend'].diff().mean()
        patterns['volume_reversal'] = (selling_trend < 0 and buying_trend > 0)
        
        # Check for bearish divergence
        price_trend = recent_data['Close'].diff().mean()
        volume_delta_trend = recent_data['Volume_Delta'].mean()
        patterns['bearish_divergence'] = (price_trend > 0 and volume_delta_trend < 0)
        
        # Check for potential bullish reversal
        patterns['bullish_reversal'] = (
            price_trend < 0 and  # Price falling
            recent_data['Volume_Spike'].any() and  # Recent volume spike
            recent_data['Volume_Delta'].tail(3).mean() > 0  # Recent buying pressure
        )
        
        # Calculate volume delta strength (-1 to 1)
        patterns['volume_delta_strength'] = (
            recent_data['Volume_Delta'].mean() / recent_data['Volume'].mean()
        )
        
        return patterns

    def check_volume_spike_near_last_low(self, df: pd.DataFrame, last_low_idx: pd.Timestamp, window: int = 5) -> bool:
        """Check if there's a volume spike near the last swing low."""
        if pd.isna(last_low_idx):
            return False
        
        try:
            idx_position = df.index.get_loc(last_low_idx)
            end_position = min(idx_position + window, len(df))
            end_idx = df.index[end_position] if end_position < len(df) else df.index[-1]
            relevant_period = df.loc[last_low_idx:end_idx]
            return any(relevant_period['Volume_Spike'])
        except:
            return False

    def analyze_stock(self, ticker: str, window: int, start_date: date, end_date: date) -> StockResponse:
        """Main analysis function that processes stock data and returns analysis results."""
        logger.info(f"[Analyzer] Starting analysis for {ticker}")
        logger.info(f"[Analyzer] Parameters: window={window}, start_date={start_date}, end_date={end_date}")
        
        self.debug_msgs = []
        analysis_start_time = datetime.now()
        
        # Use June 1st, 2023 as the hard start date for historical data
        HARD_START_DATE = date(2023, 6, 1)
        actual_start_date = max(HARD_START_DATE, start_date)
        logger.info(f"[Analyzer] Using actual start date: {actual_start_date}")
        
        # Load data from cache
        try:
            logger.info(f"[Analyzer] Loading data for {ticker} from cache...")
            start_time = time()
            df, error_msg = self._load_data_from_cache(ticker.upper())
            if error_msg:
                raise ValueError(error_msg)
            
            load_time = time() - start_time
            logger.info(f"[Analyzer] Loaded {len(df)} rows of data in {load_time:.2f} seconds")
            if df.empty:
                raise ValueError(f"No data found for {ticker.upper()}")
            
            # Filter data by date range
            df = df[df.index.date >= actual_start_date]
            df = df[df.index.date <= end_date]
            logger.info(f"[Analyzer] Using {len(df)} rows after date filtering")
        except Exception as e:
            self.debug_msgs.append(f"Error downloading data: {str(e)}")
            raise ValueError(f"Failed to download data for {ticker.upper()}: {str(e)}")

        try:
            logger.info(f"[Analyzer] Starting technical analysis for {ticker}")
            
            # Create a copy of the dataframe to ensure alignment
            df = df.copy()
            
            try:
                # Calculate indicators
                logger.info("[Analyzer] Calculating moving averages...")
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['MA100'] = df['Close'].rolling(window=100).mean()
                
                # Calculate volume indicators ensuring alignment
                logger.info("[Analyzer] Calculating volume indicators...")
                volume_ma20 = df['Volume'].rolling(window=20).mean()
                df['Volume_MA20'] = volume_ma20
                df['Volume_Spike'] = df['Volume'].gt(volume_ma20)  # Using gt() instead of > for better alignment handling
                df['Volume_Color'] = np.where(df['Close'] > df['Open'], 'green', 'red')
                logger.info("[Analyzer] Technical indicators calculated successfully")
            except Exception as e:
                logger.error(f"[Analyzer] Error calculating indicators: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to calculate technical indicators: {str(e)}")
        except Exception as e:
            self.debug_msgs.append(f"Error calculating indicators: {str(e)}")
            raise ValueError(f"Failed to calculate indicators: {str(e)}")

        # Identify swing points
        logger.info(f"[Analyzer] Identifying swing points with window={window}")
        try:
            df['Swing_High'] = df['High'][(df['High'].shift(window) < df['High']) & 
                                        (df['High'].shift(-window) < df['High'])]
            df['Swing_Low'] = df['Low'][(df['Low'].shift(window) > df['Low']) & 
                                       (df['Low'].shift(-window) > df['Low'])]

            # Get swing series
            swing_lows = df['Swing_Low'].dropna()
            swing_highs = df['Swing_High'].dropna()
            
            logger.info(f"[Analyzer] Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
        except Exception as e:
            logger.error(f"[Analyzer] Error identifying swing points: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to identify swing points: {str(e)}")

        # Calculate metrics
        logger.info("[Analyzer] Calculating swing metrics...")
        try:
            avg_pct_diff_hl = self.avg_pct_diff_between_swings(swing_lows)
            avg_pct_diff_hh = self.avg_pct_diff_between_swings(swing_highs)
            higher_lows_count = self.count_higher_lows_highs(swing_lows)
            higher_highs_count = self.count_higher_lows_highs(swing_highs)

            logger.info(f"[Analyzer] Higher lows: {higher_lows_count}, Higher highs: {higher_highs_count}")
            logger.info(f"[Analyzer] Avg % diff - Lows: {avg_pct_diff_hl}, Highs: {avg_pct_diff_hh}")

            # Calculate average days between swings
            avg_days_hl = self.avg_days_between_swings(swing_lows.index)
            avg_days_hh = self.avg_days_between_swings(swing_highs.index)
            valid_days = [d for d in [avg_days_hl, avg_days_hh] if d is not None]
            avg_days_combined = round(sum(valid_days) / len(valid_days), 2) if valid_days else None
            
            logger.info(f"[Analyzer] Average days between swings: {avg_days_combined}")

            # Calculate score
            trading_days = len(df)
            max_possible_swings = trading_days / window if window > 0 else 1
            total_swings = higher_lows_count + higher_highs_count
            score_pct = min(100, (total_swings / max_possible_swings) * 100)
            
            logger.info(f"[Analyzer] Score: {score_pct:.2f}%")
        except Exception as e:
            logger.error(f"[Analyzer] Error calculating metrics: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to calculate metrics: {str(e)}")

        # Check conditions for strong buy
        recent_swings_valid = self.check_recent_swing_conditions(df, swing_lows, swing_highs)
        avg_days_ok = avg_days_combined is not None and abs(avg_days_combined - self.TARGET_DAYS_BETWEEN_SWINGS) <= 1
        avg_amplitude = max(avg_pct_diff_hl or 0, avg_pct_diff_hh or 0)
        amplitude_ok = avg_amplitude is not None and abs(avg_amplitude - self.TARGET_SWING_AMPLITUDE) <= 2

        # Initialize Strong_Buy column
        df['Strong_Buy'] = False

        # Check for strong buy conditions
        if recent_swings_valid and avg_days_ok and amplitude_ok:
            for idx, row in df.iterrows():
                if pd.isna(row['Swing_Low']):
                    continue
                
                last_low_idx = swing_lows.index[-1] if not swing_lows.empty else None
                
                if (self.check_ma_alignment(row) and
                    row['Close'] > row['MA20'] and
                    self.check_volume_spike_near_last_low(df, last_low_idx)):
                    df.at[idx, 'Strong_Buy'] = True

        # Convert DataFrame data to Python native types with robust null handling
        def safe_convert(value, type_func):
            """Safely convert a value to a given type, handling None and NaN"""
            if pd.isna(value):
                return None
            try:
                return type_func(value)
            except (ValueError, TypeError):
                return None

        chart_data = ChartData(
            # Convert dates to ISO format strings for better JSON serialization
            dates=[d.isoformat() for d in df.index],
            
            # Convert numeric columns with null handling
            prices=[safe_convert(x, float) for x in df['Close']],
            volumes=[safe_convert(x, int) for x in df['Volume']],
            ma20=[safe_convert(x, float) for x in df['MA20']],
            ma50=[safe_convert(x, float) for x in df['MA50']],
            ma100=[safe_convert(x, float) for x in df['MA100']],
            volume_ma20=[safe_convert(x, float) for x in df['Volume_MA20']],
            swing_highs=[safe_convert(x, float) for x in df['Swing_High']],
            swing_lows=[safe_convert(x, float) for x in df['Swing_Low']],
            
            # Convert boolean values
            strong_buy_signals=[bool(x) if not pd.isna(x) else False for x in df['Strong_Buy']],
            
            # Ensure strings are Python strings
            volume_colors=[str(x) if not pd.isna(x) else 'gray' for x in df['Volume_Color']]
        )

        # Prepare metrics
        metrics = AnalysisMetrics(
            higher_lows_count=int(higher_lows_count),
            higher_highs_count=int(higher_highs_count),
            avg_days_between_swings=float(avg_days_combined) if avg_days_combined is not None else None,
            avg_pct_diff_highs=float(avg_pct_diff_hh) if avg_pct_diff_hh is not None else None,
            avg_pct_diff_lows=float(avg_pct_diff_hl) if avg_pct_diff_hl is not None else None,
            score_percentage=float(round(score_pct, 2)),
            conditions_met={
                "recent_swings": bool(recent_swings_valid),
                "avg_days": bool(avg_days_ok),
                "amplitude": bool(amplitude_ok),
                "ma_alignment": bool(self.check_ma_alignment(df.iloc[-1])),
                "price_above_ma20": bool(df.iloc[-1]['Close'] > df.iloc[-1]['MA20']),
                "volume_spike": bool(not swing_lows.empty and self.check_volume_spike_near_last_low(df, swing_lows.index[-1]))
            }
        )

        # Reset reported conditions for new analysis
        self._reported_conditions.clear()
        
        # Add detailed conditions check to debug messages
        self._add_debug_msg("📊 𝗦𝗪𝗜𝗡𝗚 𝗔𝗡𝗔𝗟𝗬𝗦𝗜𝗦", "header_swing")
        
        # Check each condition with unique keys to avoid repetition
        self._add_debug_msg(
            f"🔹 Higher Swings (4w) {'✅' if recent_swings_valid else '❌'} "
            "ⓘ [4+ higher highs & lows in past 6 weeks]",
            "recent_swings"
        )
        
        self._add_debug_msg(
            f"🔹 Swing Timing {'✅' if avg_days_ok else '❌'} "
            f"[{f'{avg_days_combined:.1f}d' if avg_days_combined is not None else 'N/A'}] "
            "ⓘ [Ideal: 4-5 days between swings]",
            "avg_days"
        )
        
        self._add_debug_msg(
            f"🔹 Swing Amplitude {'✅' if amplitude_ok else '❌'} "
            f"[{f'{avg_amplitude:.1f}%' if avg_amplitude is not None else 'N/A'}] "
            "ⓘ [Target: 8% price movement]",
            "amplitude"
        )
        
        self._add_debug_msg(
            f"🔹 MA Alignment {'✅' if self.check_ma_alignment(df.iloc[-1]) else '❌'} "
            "[20>50>100] "
            "ⓘ [Bullish when faster MAs > slower]",
            "ma_alignment"
        )
        
        self._add_debug_msg(
            f"🔹 Price > MA20 {'✅' if df.iloc[-1]['Close'] > df.iloc[-1]['MA20'] else '❌'} "
            "ⓘ [Uptrend confirmation]",
            "price_above_ma20"
        )
        
        self._add_debug_msg(
            f"🔹 Vol @ Low {'✅' if not swing_lows.empty and self.check_volume_spike_near_last_low(df, swing_lows.index[-1]) else '❌'} "
            "ⓘ [Volume surge at recent low]",
            "volume_spike"
        )

        # Add volume analysis results
        volume_patterns = self.analyze_volume_patterns(df)
        
        self._add_debug_msg("\n📈 𝗩𝗢𝗟𝗨𝗠𝗘 𝗣𝗔𝗧𝗧𝗘𝗥𝗡𝗦\n", "volume_header")
        
        # Volume Spikes
        if volume_patterns['volume_spikes']:
            spike_dates = [d.strftime('%m/%d') for d in volume_patterns['volume_spikes']]
            self._add_debug_msg(
                f"Recent Spikes: ✅ [{', '.join(spike_dates)}] "
                "ⓘ [Volume > 2x average]",
                "volume_spikes"
            )
        
        # Volume Reversal
        self._add_debug_msg(
            f"Buy/Sell Shift: {'✅' if volume_patterns['volume_reversal'] else '❌'} "
            "[↓Sell + ↑Buy] "
            "ⓘ [Selling pressure decreasing, buying increasing]",
            "volume_reversal"
        )
        
        # Volume Delta
        delta_strength = volume_patterns['volume_delta_strength']
        delta_emoji = '✅' if abs(delta_strength) > self.VOLUME_DELTA_THRESHOLD else '❌'
        delta_type = "Buy" if delta_strength > 0 else "Sell"
        self._add_debug_msg(
            f"Vol Delta: {delta_emoji} [{delta_type} {abs(delta_strength):.1%}] "
            "ⓘ [Buying vs Selling pressure]",
            "volume_delta"
        )
        
        # Divergence/Reversal Patterns
        if volume_patterns['bearish_divergence']:
            self._add_debug_msg(
                "⚠️ BEARISH DIVERGENCE [↑Price + ↑Selling]",
                "bearish_divergence"
            )
        
        if volume_patterns['bullish_reversal']:
            self._add_debug_msg(
                "💡 BULLISH SIGNAL [↓Price + ↑Buying]",
                "bullish_reversal"
            )
            
        # Add new volume pattern messages
        self._add_debug_msg("\n🔍 𝗩𝗢𝗟𝗨𝗠𝗘 𝗧𝗥𝗘𝗡𝗗𝗦\n", "advanced_volume_header")
        
        # Volume Delta trend
        delta_avg = volume_patterns['volume_metrics']['delta_avg']
        self._add_debug_msg(
            f"{self.VOLUME_DELTA_LOOKBACK}d Delta: {'✅' if volume_patterns['positive_volume_delta'] else '❌'} "
            f"[{delta_avg:,.0f}] "
            "ⓘ [Net buying pressure over period]",
            "volume_delta_trend"
        )
        
        # Volume after recent low
        vol_ratio = volume_patterns['volume_metrics']['post_low_volume_ratio']
        self._add_debug_msg(
            f"Post-Low Vol: {'✅' if volume_patterns['high_volume_after_low'] else '❌'} "
            f"[{vol_ratio:.1f}x avg] "
            "ⓘ [High volume after price bottom]",
            "post_low_volume"
        )
        
        # Volume confirming highs
        vol_change = volume_patterns['volume_metrics']['high_volume_change']
        self._add_debug_msg(
            f"High Vol @ Highs: {'✅' if volume_patterns['volume_confirms_highs'] else '❌'} "
            f"[{vol_change:+.1%}] "
            "ⓘ [Volume increases with price peaks]",
            "volume_confirms_highs"
        )

        # Add a summary line (no need to track this as unique since it's always at the end)
        conditions_met = sum(1 for condition in metrics.conditions_met.values() if condition)
        total_conditions = len(metrics.conditions_met)
        self._add_debug_msg(f"\nTotal Conditions Met: {conditions_met}/{total_conditions}")

        # Ensure debug messages are strings
        clean_debug_msgs = [str(msg) for msg in self.debug_msgs]
        
        # Log analysis duration and create response
        analysis_duration = (datetime.now() - analysis_start_time).total_seconds()
        logger.info(f"[Analyzer] Analysis completed for {ticker} in {analysis_duration:.2f} seconds")
        
        try:
            # Create response object
            response = StockResponse(
                chart_data=chart_data,
                metrics=metrics,
                debug_messages=clean_debug_msgs
            )
            
            # Validate JSON serialization
            try:
                encoded_response = jsonable_encoder(response)
                json.dumps(encoded_response)  # Test JSON serialization
                logger.info(f"[Analyzer] Successfully validated response JSON for {ticker}")
            except Exception as e:
                logger.error(f"[Analyzer] Response JSON validation failed: {str(e)}", exc_info=True)
                # Return minimal response instead
                logger.warning("[Analyzer] Falling back to minimal response")
                return StockResponse(
                    chart_data=ChartData(
                        dates=[d.isoformat() for d in df.index][-10:],
                        prices=[safe_convert(x, float) for x in df['Close']][-10:],
                        volumes=[],
                        ma20=[],
                        ma50=[],
                        ma100=[],
                        volume_ma20=[],
                        swing_highs=[],
                        swing_lows=[],
                        strong_buy_signals=[],
                        volume_colors=[]
                    ),
                    metrics=AnalysisMetrics(
                        higher_lows_count=higher_lows_count,
                        higher_highs_count=higher_highs_count,
                        avg_days_between_swings=None,
                        avg_pct_diff_highs=None,
                        avg_pct_diff_lows=None,
                        score_percentage=score_pct,
                        conditions_met={}
                    ),
                    debug_messages=[f"Error in full response: {str(e)}"]
                )
            
            logger.info(f"[Analyzer] Successfully created response object for {ticker}")
            return response
        except Exception as e:
            logger.error(f"[Analyzer] Error creating response object: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create response object: {str(e)}")