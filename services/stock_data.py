from typing import Dict, List, Optional
from app.models.stock_models import StockInfo, StockSegment

import asyncio
import logging

logger = logging.getLogger(__name__)

class StockDataService:
    """Service to manage stock data and segments"""
    
    def __init__(self):
        self.stock_data: Dict[str, StockInfo] = {}
        self.initialized = False
        
    async def initialize_async(self):
        """Initialize stock data asynchronously"""
        try:
            logger.info("Starting background initialization of StockDataService...")
            self.stock_data = await asyncio.to_thread(self._initialize_stock_data)
            self.initialized = True
            logger.info("StockDataService initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize StockDataService: {str(e)}", exc_info=True)
            raise
        
    def _initialize_stock_data(self) -> Dict[str, StockInfo]:
        """Initialize stock data with segments"""
        stocks = {}
        
        # Quantum stocks
        quantum_stocks = [
            ("IONQ", "IonQ", "Quantum computing company"),
            ("QBTS", "D-Wave", "Quantum computing solutions"),
            ("RGTI", "Rigetti", "Quantum processors"),
            ("LAES", "LAES", "Quantum technology")
        ]
        for ticker, name, desc in quantum_stocks:
            stocks[ticker] = StockInfo(
                ticker=ticker,
                segment=StockSegment.QUANTUM,
                name=name,
                description=desc
            )
        
        # Chip stocks
        chip_stocks = [
            ("NVDA", "NVIDIA", "GPU and AI computing"),
            ("AMD", "Advanced Micro Devices", "Semiconductors"),
            ("AVGO", "Broadcom", "Semiconductor solutions"),
            ("ASML", "ASML Holding", "Semiconductor equipment"),
            ("TSM", "TSMC", "Semiconductor manufacturing"),
            ("POET", "POET Technologies", "Photonic solutions"),
            ("TER", "Teradyne", "Automation equipment"),
            ("SMCI", "Super Micro Computer", "Server solutions")
        ]
        for ticker, name, desc in chip_stocks:
            stocks[ticker] = StockInfo(
                ticker=ticker,
                segment=StockSegment.CHIP,
                name=name,
                description=desc
            )
            
        # Add all other segments similarly...
        # Bitcoin Miners
        btc_stocks = [
            ("MARA", "Marathon Digital", "Bitcoin mining"),
            ("HUT", "Hut 8 Mining", "Cryptocurrency mining"),
            ("CIFR", "Cipher Mining", "Bitcoin mining"),
            ("IREN", "Iris Energy", "Bitcoin mining")
        ]
        for ticker, name, desc in btc_stocks:
            stocks[ticker] = StockInfo(
                ticker=ticker,
                segment=StockSegment.BITCOIN_MINERS,
                name=name,
                description=desc
            )
            
        # Continue with other segments...
        
        return stocks
    
    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Get stock info by ticker"""
        return self.stock_data.get(ticker.upper())
    
    def get_stocks_by_segment(self, segment: StockSegment) -> List[StockInfo]:
        """Get all stocks in a segment"""
        return [
            stock for stock in self.stock_data.values()
            if stock.segment == segment
        ]
    
    def get_all_segments(self) -> Dict[StockSegment, List[StockInfo]]:
        """Get all segments with their stocks"""
        segments = {}
        for segment in StockSegment:
            segments[segment] = self.get_stocks_by_segment(segment)
        return segments
    
    def search_stocks(self, query: str) -> List[StockInfo]:
        """Search stocks by ticker or name"""
        query = query.upper()
        return [
            stock for stock in self.stock_data.values()
            if query in stock.ticker.upper() or
               (stock.name and query in stock.name.upper())
        ]