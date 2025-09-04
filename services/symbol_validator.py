from typing import Optional, Dict, Any, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SymbolValidator:
    def __init__(self):
        self.valid_exchanges = {'NYSE', 'NASDAQ', 'AMEX', 'NYQ', 'NMS'}
        
    async def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate stock symbol and return company info if valid.
        Returns (is_valid, company_info)
        """
        try:
            # Get stock info from yfinance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Basic validation
            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"No market data available for {symbol}")
                return False, None
                
            # Check exchange
            exchange = info.get('exchange', '')
            if exchange not in self.valid_exchanges:
                logger.warning(f"Invalid exchange {exchange} for {symbol}")
                return False, None
                
            # Check data availability
            history = stock.history(period="1mo")
            if len(history) < 5:  # Require at least 5 days of data
                logger.warning(f"Insufficient historical data for {symbol}")
                return False, None
                
            # Return company info
            company_info = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', ''),
                'market_cap': info.get('marketCap'),
                'exchange': exchange
            }
            
            logger.info(f"Successfully validated symbol {symbol}")
            return True, company_info
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False, None
            
    def check_data_availability(self, symbol: str) -> Dict[str, Any]:
        """Check data availability for the symbol"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get recent data
            end = datetime.now()
            start = end - timedelta(days=30)
            history = stock.history(start=start, end=end)
            
            return {
                'available': len(history) > 0,
                'days_available': len(history),
                'start_date': history.index[0].date().isoformat() if len(history) > 0 else None,
                'end_date': history.index[-1].date().isoformat() if len(history) > 0 else None,
                'has_volume': 'Volume' in history.columns,
                'has_price': all(col in history.columns for col in ['Open', 'High', 'Low', 'Close'])
            }
            
        except Exception as e:
            logger.error(f"Error checking data availability for {symbol}: {str(e)}")
            return {
                'available': False,
                'error': str(e)
            }