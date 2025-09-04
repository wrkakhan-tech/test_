import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import logging
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class YFinanceClient:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)  # Limit concurrent downloads
        self.download_semaphore = asyncio.Semaphore(5)  # Rate limit
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds
        
    async def download_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        include_prepost: bool = False
    ) -> Optional[pd.DataFrame]:
        """Download stock data with retry logic and rate limiting"""
        async with self.download_semaphore:
            for attempt in range(self.retry_attempts):
                try:
                    logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
                    
                    # Use ThreadPoolExecutor for the blocking yfinance call
                    df = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: yf.download(
                            symbol.upper(),
                            start=start_date,
                            end=end_date,
                            auto_adjust=True,
                            threads=False,
                            progress=False,
                            prepost=include_prepost
                        )
                    )
                    
                    if df.empty:
                        logger.warning(f"No data returned for {symbol}")
                        return None
                        
                    # Handle multi-index columns if returned
                    if isinstance(df.columns, pd.MultiIndex):
                        if symbol.upper() in df.columns.levels[1]:
                            df = df.xs(symbol.upper(), axis=1, level=1, drop_level=True)
                        else:
                            logger.error(f"Symbol {symbol} not found in multi-index columns")
                            return None
                    
                    # Validate data
                    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
                    if not all(col in df.columns for col in required_columns):
                        logger.error(f"Missing required columns for {symbol}")
                        return None
                        
                    # Sort by date and handle any duplicates
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep='last')]
                    
                    logger.info(f"Successfully downloaded {len(df)} records for {symbol}")
                    return df
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} (attempt {attempt + 1}): {str(e)}")
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"All retry attempts failed for {symbol}")
                        raise ValueError(f"Failed to download data for {symbol} after {self.retry_attempts} attempts: {str(e)}")
                        
    async def get_latest_data(
        self,
        symbol: str,
        days_back: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Get latest price data for a symbol"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            df = await self.download_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                return None
                
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            result = {
                'date': df.index[-1].isoformat(),
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume'])
            }
            
            if prev is not None:
                price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                volume_change = ((latest['Volume'] - prev['Volume']) / prev['Volume']) * 100
                result.update({
                    'change': price_change,
                    'volume_change': volume_change
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return None