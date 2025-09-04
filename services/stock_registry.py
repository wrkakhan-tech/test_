from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from app.models.stock_models import StockInfo, StockSegment

logger = logging.getLogger(__name__)

class StockRegistry:
    def __init__(self):
        self.registry_file = Path("app/data/stock_registry.json")
        self.stocks: Dict[str, StockInfo] = {}
        self.load_registry()
        self.ensure_default_stocks()
        
    def load_registry(self) -> None:
        """Load stock registry from file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.stocks = {
                        symbol: StockInfo(**info)
                        for symbol, info in data.items()
                    }
                logger.info(f"Loaded {len(self.stocks)} stocks from registry")
            else:
                logger.info("Registry file not found, creating new registry")
                self.registry_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_registry()
                
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            
    def save_registry(self) -> None:
        """Save registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(
                    {s: stock.dict() for s, stock in self.stocks.items()},
                    f, indent=2
                )
            logger.info(f"Saved {len(self.stocks)} stocks to registry")
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            
    def add_stock(self, symbol: str, segment: StockSegment, info: Dict[str, Any]) -> StockInfo:
        """Add new stock to registry"""
        if symbol in self.stocks:
            logger.warning(f"Stock {symbol} already exists in registry")
            raise ValueError(f"Stock {symbol} already exists in registry")
            
        stock = StockInfo(
            ticker=symbol,
            segment=segment,
            name=info.get('name', symbol),
            description=info.get('description', '')
        )
        
        self.stocks[symbol] = stock
        self.save_registry()
        logger.info(f"Added new stock {symbol} to registry")
        return stock
        
    def remove_stock(self, symbol: str) -> bool:
        """Remove stock from registry"""
        if symbol in self.stocks:
            del self.stocks[symbol]
            self.save_registry()
            logger.info(f"Removed stock {symbol} from registry")
            return True
        return False
        
    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get stock info from registry"""
        return self.stocks.get(symbol)
        
    def get_all_stocks(self) -> List[StockInfo]:
        """Get all registered stocks"""
        return list(self.stocks.values())
        
    def get_stocks_by_segment(self, segment: StockSegment) -> List[StockInfo]:
        """Get all stocks in a segment"""
        return [
            stock for stock in self.stocks.values()
            if stock.segment == segment
        ]
        
    def update_stock(self, symbol: str, **updates) -> Optional[StockInfo]:
        """Update stock information"""
        if symbol not in self.stocks:
            return None
            
        stock = self.stocks[symbol]
        for key, value in updates.items():
            if hasattr(stock, key):
                setattr(stock, key, value)
                
        self.save_registry()
        logger.info(f"Updated stock {symbol} information")
        return stock

    def ensure_default_stocks(self) -> None:
        """Ensure all default stocks are in registry"""
        try:
            # Load default stocks from config
            config_file = Path("app/config/stocks.json")
            if not config_file.exists():
                logger.error("Stock configuration file not found")
                return
                
            with open(config_file, 'r') as f:
                config = json.load(f)
                default_stocks = config['segments']
                
            logger.info(f"Loaded stock configuration version {config.get('version', 'unknown')}")
            
            # Count total stocks before
            stocks_before = len(self.stocks)
            
            # Add any missing stocks
            for segment, symbols in default_stocks.items():
                for symbol in symbols:
                    if symbol not in self.stocks:
                        logger.info(f"Adding default stock {symbol} to registry")
                        self.stocks[symbol] = StockInfo(
                            ticker=symbol,
                            segment=segment,
                            name=symbol,  # Use symbol as name initially
                            description=f"Default stock in {segment} segment"
                        )
            
            # If any stocks were added, save registry
            if len(self.stocks) > stocks_before:
                logger.info(f"Added {len(self.stocks) - stocks_before} default stocks to registry")
                self.save_registry()
            else:
                logger.info("All default stocks already in registry")
                
        except Exception as e:
            logger.error(f"Error ensuring default stocks: {str(e)}")