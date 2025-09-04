import json
import os
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class StockDataCache:
    def __init__(self):
        # Set cache directory path
        self.cache_dir = Path("app/data/stock_cache")
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory not found at {self.cache_dir}, creating...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Using cache directory: {self.cache_dir}")
            
        # List existing cache files
        cache_files = list(self.cache_dir.glob("*.json"))
        logger.info(f"Found {len(cache_files)} cache files")
        
        # Initialize or load the index file
        self.index_file = self.cache_dir / "index.json"
        self._init_index()
        
    def _init_index(self):
        """Initialize or load the index file"""
        if not self.index_file.exists():
            self._save_index({
                "last_updated": datetime.now().isoformat(),
                "tickers": {}
            })
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        except Exception as e:
            logger.error(f"Error loading index file: {str(e)}")
            self.index = {
                "last_updated": datetime.now().isoformat(),
                "tickers": {}
            }
            
    def _save_index(self, index_data=None):
        """Save the index file"""
        try:
            if index_data is None:
                index_data = self.index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index file: {str(e)}")
            
    def get_ticker_last_date(self, symbol: str) -> Optional[date]:
        """Get the last date for which we have data for a ticker"""
        try:
            return datetime.fromisoformat(self.index["tickers"].get(symbol, {}).get("last_date", "2023-06-01")).date()
        except Exception:
            return date(2023, 6, 1)
            
    def update_ticker_index(self, symbol: str, last_date: date):
        """Update the last known date for a ticker"""
        self.index["tickers"][symbol] = {
            "last_date": last_date.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._save_index()
        
    def get_cached_data(self, symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[date]]:
        """Get cached data for a symbol"""
        try:
            cache_file = self.cache_dir / f"{symbol}.json"
            logger.debug(f"Looking for cache file: {cache_file}")
            
            if not cache_file.exists():
                logger.warning(f"Cache file not found for {symbol}")
                return None, None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded cache data for {symbol}: {len(data.get('records', []))} records")
                
            # Always return cached data - we'll refresh in background if needed
            last_updated = datetime.fromisoformat(data.get('last_updated', '2023-06-01'))
            return data, last_updated.date()
            
        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {str(e)}")
            return None, None
            
    def get_last_available_date(self, symbol: str) -> Optional[date]:
        """Get the last available date from cached data"""
        try:
            data, _ = self.get_cached_data(symbol)
            if not data or not data.get('records'):
                return None
            
            records = data['records']
            if not records:
                return None
                
            # Get the last record's date
            last_date_str = records[-1]['date']
            return datetime.strptime(last_date_str, '%Y-%m-%d').date()
            
        except Exception as e:
            logger.error(f"Error getting last available date for {symbol}: {str(e)}")
            return None
            
    def get_performance_data(self, symbol: str, days: int = 1) -> Optional[Dict[str, Any]]:
        """Get performance data for specified number of days"""
        try:
            data, _ = self.get_cached_data(symbol)
            if not data or not data.get('records'):
                return None
                
            records = data['records']
            if len(records) < days + 1:  # Need at least days+1 records to calculate performance
                return None
                
            # Get latest and comparison records
            latest = records[-1]
            comparison = records[-(days + 1)]  # Get record from N+1 days ago
            
            def safe_float(value):
                try:
                    if isinstance(value, pd.Series):
                        return float(value.iloc[0])
                    return float(value)
                except (ValueError, TypeError, IndexError):
                    return 0.0
                    
            def safe_int(value):
                try:
                    if isinstance(value, pd.Series):
                        return int(value.iloc[0])
                    return int(value)
                except (ValueError, TypeError, IndexError):
                    return 0
                    
            # Calculate performance over the period
            latest_close = safe_float(latest['close'])
            comparison_close = safe_float(comparison['close'])
            latest_volume = safe_int(latest['volume'])
            
            # Calculate average daily volume over the period
            volume_slice = records[-days:]
            avg_daily_volume = sum(safe_int(r['volume']) for r in volume_slice) / len(volume_slice)
            
            # Calculate price change percentage
            price_change = ((latest_close - comparison_close) / comparison_close) * 100
            volume_change = ((latest_volume - avg_daily_volume) / avg_daily_volume) * 100 if avg_daily_volume > 0 else 0
            
            return {
                'date': latest['date'],
                'price': latest_close,
                'change': price_change,
                'volume': latest_volume,
                'volume_change': volume_change,
                'period_days': days,
                'start_date': comparison['date'],
                'volume_color': 'green' if latest_close > safe_float(latest['open']) else 'red',
                'open': safe_float(latest['open']),
                'high': safe_float(latest['high']),
                'low': safe_float(latest['low'])
            }
                
        except Exception as e:
            logger.error(f"Error getting performance data for {symbol}: {str(e)}")
            return None

    def get_latest_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price data from cache"""
        try:
            data, _ = self.get_cached_data(symbol)
            if not data or not data.get('records'):
                return None
                
            records = data['records']
            if not records:
                return None
                
            # Get latest and previous record
            latest = records[-1]
            prev = records[-2] if len(records) > 1 else None
            
            def safe_float(value):
                try:
                    if isinstance(value, pd.Series):
                        return float(value.iloc[0])
                    return float(value)
                except (ValueError, TypeError, IndexError):
                    return 0.0
                    
            def safe_int(value):
                try:
                    if isinstance(value, pd.Series):
                        return int(value.iloc[0])
                    return int(value)
                except (ValueError, TypeError, IndexError):
                    return 0
                    
            if prev:
                # Calculate changes using safe conversion
                latest_close = safe_float(latest['close'])
                prev_close = safe_float(prev['close'])
                latest_volume = safe_int(latest['volume'])
                prev_volume = safe_int(prev['volume'])
                
                price_change = ((latest_close - prev_close) / prev_close) * 100
                volume_change = ((latest_volume - prev_volume) / prev_volume) * 100
            else:
                price_change = 0
                volume_change = 0
                latest_close = safe_float(latest['close'])
                latest_volume = safe_int(latest['volume'])
                
            return {
                'date': latest['date'],  # Keep as ISO string
                'price': latest_close,
                'change': price_change,
                'volume': latest_volume,
                'volume_change': volume_change,
                'volume_color': 'green' if safe_float(latest['close']) > safe_float(latest['open']) else 'red',
                'open': safe_float(latest['open']),
                'high': safe_float(latest['high']),
                'low': safe_float(latest['low'])
            }
                
        except Exception as e:
            logger.error(f"Error getting latest cached data for {symbol}: {str(e)}")
            return None
            
    def validate_data_continuity(self, records: list, is_new_data: bool = False, last_existing_date: Optional[date] = None) -> Tuple[bool, Optional[str]]:
        """Validate data continuity and integrity
        
        Args:
            records: List of records to validate
            is_new_data: If True, only validate the new data being appended
            last_existing_date: The last date from existing data, used to validate the connection point
        """
        if not records:
            return False, "No records to validate"
            
        try:
            # Check for date sequence
            dates = [datetime.strptime(r['date'], '%Y-%m-%d').date() for r in records]
            dates_set = set(dates)
            
            # Check for duplicates
            if len(dates) != len(dates_set):
                return False, "Duplicate dates found in data"
                
            if is_new_data and last_existing_date:
                # Only validate the gap between existing and new data
                first_new_date = dates[0]
                date_diff = (first_new_date - last_existing_date).days
                if date_diff > 5:  # Increased threshold for holidays and long weekends
                    return False, f"Large gap found between existing data ({last_existing_date}) and new data ({first_new_date})"
                    
                # Then only validate continuity within new data
                start_idx = 0
            else:
                start_idx = 0
                
            # Check for gaps in the relevant range
            for i in range(start_idx, len(dates) - 1):
                date_diff = (dates[i+1] - dates[i]).days
                if date_diff > 5:  # Increased threshold for holidays and long weekends
                    return False, f"Large gap found between {dates[i]} and {dates[i+1]}"
                    
            # Validate data values
            for record in records:
                if not all(isinstance(record.get(field), (int, float)) 
                          for field in ['open', 'high', 'low', 'close', 'volume']):
                    return False, f"Invalid data types in record for date {record['date']}"
                    
                if not (record['low'] <= record['open'] <= record['high'] and 
                       record['low'] <= record['close'] <= record['high']):
                    return False, f"Price range validation failed for date {record['date']}"
                    
                if record['volume'] < 0:
                    return False, f"Negative volume found for date {record['date']}"
                    
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
            
    def save_to_cache(self, symbol: str, df, existing_data: dict = None) -> None:
        """Save DataFrame to cache with improved handling of gaps and duplicates"""
        try:
            cache_file = self.cache_dir / f"{symbol}.json"
            
            # Convert DataFrame to records
            records_dict = {}  # Use dict for O(1) lookup and update
            
            # Load existing data if available
            if existing_data and 'records' in existing_data:
                for record in existing_data['records']:
                    records_dict[record['date']] = record
            
            # Process new data
            for idx, row in df.iterrows():
                record_date = idx.date().isoformat()
                
                # Always update if it's today's data or if it's new data
                if record_date == date.today().isoformat() or record_date not in records_dict:
                    records_dict[record_date] = {
                        'date': record_date,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume'])
                    }
                    
            # Convert dict back to sorted list
            new_records = sorted(records_dict.values(), key=lambda x: x['date'])
            
            # Get last existing date if we're appending data
            last_existing_date = None
            if existing_data and 'records' in existing_data and existing_data['records']:
                last_record = existing_data['records'][-1]
                last_existing_date = datetime.strptime(last_record['date'], '%Y-%m-%d').date()
            
            # Validate data continuity and integrity
            is_valid, error_msg = self.validate_data_continuity(
                new_records,
                is_new_data=bool(existing_data),
                last_existing_date=last_existing_date
            )
            if not is_valid:
                logger.error(f"Data validation failed for {symbol}: {error_msg}")
                raise ValueError(f"Data validation failed: {error_msg}")
                
            # Save to file
            data = {
                'symbol': symbol,
                'last_updated': datetime.now().isoformat(),
                'records': new_records
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache for {symbol}: {str(e)}")
            raise