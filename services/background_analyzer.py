import asyncio
import logging
from datetime import datetime, date
from typing import Dict, Set, Union, List, Optional
from app.services.stock_analysis import StockAnalyzer
from app.models.stock_models import StockInfo
from app.services.stock_data import StockDataService

logger = logging.getLogger(__name__)

class BackgroundAnalyzer:
    def __init__(self):
        self.analyzer = StockAnalyzer()
        self.analysis_cache: Dict[str, Dict] = {}
        self.analysis_in_progress: Set[str] = set()
        self.last_update = {}
        self.update_interval = 300  # 5 minutes
        
    def _get_ticker(self, symbol: Union[str, StockInfo]) -> str:
        """Extract ticker from symbol object or string"""
        return symbol.ticker if hasattr(symbol, 'ticker') else str(symbol)
        
    def get_active_stocks(self) -> Dict[str, Dict[str, float]]:
        """Get stocks with volume significantly above their average
        
        Returns:
            Dictionary mapping ticker to dict containing:
            - volume: latest volume
            - avg_volume: 20-day average volume
            - volume_ratio: how many times above average
            - segment: stock segment
            - volume_trend: 5-day volume trend percentage
        """
        active_stocks = {}
        stock_service = StockDataService()
        
        for symbol, cached in self.analysis_cache.items():
            analysis = cached['results'].get(5)  # Use 5-day window analysis
            if not analysis or not analysis.chart_data.volumes:
                continue
            
            # Get volumes and calculate metrics
            volumes = analysis.chart_data.volumes
            latest_volume = volumes[-1]
            
            # Calculate 20-day average volume if we have enough data
            if len(volumes) >= 20:
                avg_volume = sum(volumes[-20:]) / 20
                volume_ratio = latest_volume / avg_volume
                
                # Calculate 5-day volume trend
                if len(volumes) >= 5:
                    five_day_avg = sum(volumes[-5:]) / 5
                    prev_five_day_avg = sum(volumes[-10:-5]) / 5 if len(volumes) >= 10 else five_day_avg
                    volume_trend = ((five_day_avg - prev_five_day_avg) / prev_five_day_avg) * 100
                else:
                    volume_trend = 0
                
                # Consider active if volume is 50% above average
                if volume_ratio > 1.5:
                    stock_info = stock_service.get_stock_info(symbol)
                    active_stocks[symbol] = {
                        "volume": latest_volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": volume_ratio,
                        "segment": stock_info.segment if stock_info else None,
                        "volume_trend": volume_trend
                    }
                
        return active_stocks
        
    def get_top_gainers(self) -> Dict[str, Dict[str, float]]:
        """Get stocks with positive price change
        
        Returns:
            Dictionary mapping ticker to dict containing:
            - change: price change percentage
            - volume: latest volume
            - segment: stock segment
        """
        gainers = {}
        stock_service = StockDataService()
        
        for symbol, cached in self.analysis_cache.items():
            analysis = cached['results'].get(5)
            if not analysis or not analysis.chart_data.prices:
                continue
                
            # Calculate price change
            prices = analysis.chart_data.prices
            latest_price = prices[-1]
            prev_price = prices[-2] if len(prices) > 1 else None
            
            if prev_price and latest_price > prev_price:
                change_pct = ((latest_price - prev_price) / prev_price) * 100
                stock_info = stock_service.get_stock_info(symbol)
                gainers[symbol] = {
                    "change": change_pct,
                    "volume": analysis.chart_data.volumes[-1],
                    "segment": stock_info.segment if stock_info else None
                }
                
        return gainers