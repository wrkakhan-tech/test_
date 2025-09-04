from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from datetime import datetime, date, timedelta
import pandas as pd
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)
from app.models.stock_models import (
    StockRequest, StockResponse, StockSegment,
    StockInfo, SegmentResponse, ChartData, AnalysisMetrics,
    AddStockRequest, AddStockResponse
)
from app.services.stock_analysis import StockAnalyzer
from app.services.stock_data import StockDataService

def get_service_or_503(request: Request) -> StockDataService:
    """Helper to get service and check if it's ready"""
    service: StockDataService = request.app.state.stock_data_service
    if not getattr(service, "initialized", False):
        logger.warning("StockDataService accessed before initialization completed")
        raise HTTPException(
            status_code=503, 
            detail="StockDataService is still loading. Please try again in a moment."
        )
    return service
from app.services.data_cache import StockDataCache
from app.services.symbol_validator import SymbolValidator
from app.services.stock_registry import StockRegistry
from app.services.yfinance_client import YFinanceClient

router = APIRouter()

@router.post("/analyze", response_model=StockResponse)
async def analyze_stock(request: StockRequest):
    logger.info(f"Analyzing stock: {request.ticker} from {request.start_date} to {request.end_date}")
    try:
        # Log request parameters
        logger.info(f"Analysis parameters: window={request.window}, segment={request.segment}")
        
        # Create analyzer instance
        analyzer = StockAnalyzer()
        
        # Get analysis results
        result = analyzer.analyze_stock(
            ticker=request.ticker,
            window=request.window,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Log success
        logger.info(f"Analysis completed successfully for {request.ticker}")
        return jsonable_encoder(result)
        
    except ValueError as e:
        # Handle known validation errors
        logger.warning(f"Validation error for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error analyzing {request.ticker}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred during analysis",
                "error": str(e),
                "ticker": request.ticker
            }
        )

@router.get("/segments", response_model=SegmentResponse)
async def get_segments(request: Request):
    """Get all stock segments and their stocks"""
    service = get_service_or_503(request)
    segments = service.get_all_segments()
    total_stocks = sum(len(stocks) for stocks in segments.values())
    return SegmentResponse(segments=segments, total_stocks=total_stocks)

@router.get("/stocks/{segment}", response_model=List[StockInfo])
async def get_stocks_by_segment(segment: StockSegment, request: Request):
    """Get all stocks in a segment"""
    service = get_service_or_503(request)
    return service.get_stocks_by_segment(segment)

@router.get("/all-stocks-performance")
async def get_all_stocks_performance(days: int = Query(1, ge=1, le=30)):
    """Get performance data for all stocks for specified number of days"""
    try:
        cache = StockDataCache()
        registry = StockRegistry()
        all_stocks = []
        
        # Get all registered stocks
        registered_stocks = registry.get_all_stocks()
        
        for stock in registered_stocks:
            try:
                # Get performance data from cache
                perf_data = cache.get_performance_data(stock.ticker, days)
                if perf_data:
                    # Add performance data
                    stock_data = {
                        "symbol": stock.ticker,  # Fixed: using ticker instead of symbol
                        "segment": stock.segment,
                        "performance": perf_data.get("change", 0),
                        "price": perf_data.get("price", 0),
                        "volume": perf_data.get("volume", 0),
                        "period_days": days,
                        "start_date": perf_data.get("start_date"),
                        "end_date": perf_data.get("date")
                    }
                    all_stocks.append(stock_data)
            except Exception as e:
                logger.error(f"Error getting data for {stock.ticker}: {str(e)}")
                continue
        
        return all_stocks
        
    except Exception as e:
        logger.error(f"Error getting all stocks performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Error getting stocks performance", "error": str(e)}
        )

@router.get("/search", response_model=List[StockInfo])
async def search_stocks(request: Request, q: str = Query(..., min_length=1)):
    """Search stocks by ticker or name"""
    service = get_service_or_503(request)
    return service.search_stocks(q)

@router.post("/refresh/{symbol}")
async def refresh_stock_data(symbol: str):
    """Refresh stock data if last update was more than a day ago"""
    try:
        logger.info(f"=== Starting refresh for {symbol} ===")
        logger.debug("Request details: POST /refresh/%s", symbol)
        
        cache = StockDataCache()
        client = YFinanceClient()
        
        logger.info("Checking cached data...")
        # Get current cached data and last available date
        data, _ = cache.get_cached_data(symbol)
        last_available_date = cache.get_last_available_date(symbol)
        
        logger.debug("Cache status - Data exists: %s, Last available date: %s", 
                    bool(data), last_available_date.isoformat() if last_available_date else None)
        
        today = date.today()
        
        if not data or not last_available_date:
            logger.warning(f"No cached data found for {symbol}, downloading full history")
            start_date = date(2024, 1, 1)  # Start from 2024
            try:
                df = await client.download_stock_data(symbol, start_date, today)
                if df is None or df.empty:
                    raise ValueError(f"No data available for {symbol} in the specified date range")
                cache.save_to_cache(symbol, df)
                data, _ = cache.get_cached_data(symbol)
                last_available_date = cache.get_last_available_date(symbol)
            except Exception as download_error:
                logger.error(f"Error downloading data for {symbol}: {str(download_error)}")
                raise ValueError(f"Failed to download data for {symbol}: {str(download_error)}")
        else:
            # Check if we need to refresh (last available date < today)
            if last_available_date < today:
                logger.info(f"Data for {symbol} needs update (last_available: {last_available_date}, today: {today})")
                # Download only missing days
                start_date = last_available_date + timedelta(days=1)
                if start_date <= today:
                    logger.info(f"Downloading data from {start_date} to {today}")
                    df = await client.download_stock_data(symbol, start_date, today)
                    if df is not None and not df.empty:
                        logger.info(f"Downloaded {len(df)} new records, appending to cache")
                        cache.save_to_cache(symbol, df, existing_data=data)
                        data, _ = cache.get_cached_data(symbol)
                        new_last_date = cache.get_last_available_date(symbol)
                        logger.info(f"Cache updated, now contains data up to {new_last_date}")
                    else:
                        logger.info(f"No new data available for {symbol} between {start_date} and {today}")
            else:
                logger.info(f"Data for {symbol} is up to date (last_available: {last_available_date})")
        
        if not data or not data.get('records'):
            raise ValueError(f"No data available for {symbol}")
            
        # Get latest data for UI update
        latest_data = cache.get_latest_cached_data(symbol)
        if not latest_data:
            raise ValueError(f"Failed to get latest data for {symbol}")
            
        # Get analysis data
        analyzer = StockAnalyzer()
        analysis = analyzer.analyze_stock(
            ticker=symbol,
            window=20,  # Default window
            start_date=date(2024, 1, 1),
            end_date=date.today()
        )
        
        return {
            "price_data": latest_data,
            "analysis": analysis,
            "last_updated": last_available_date.isoformat() if last_available_date else None,
            "message": "Data refreshed successfully"
        }
        
    except ValueError as e:
        logger.warning(f"Validation error refreshing {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error refreshing {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"message": f"Error refreshing {symbol}", "error": str(e)}
        )

@router.get("/price/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price and change percentage"""
    try:
        # First try to get data from cache
        cache = StockDataCache()
        cached_data = cache.get_latest_cached_data(symbol)
        
        if cached_data:
            # Return cached data immediately
            logger.info(f"Using cached data for {symbol}")
            return {
                "price": cached_data["price"],
                "change": cached_data["change"],
                "volume": cached_data["volume"],
                "timestamp": cached_data["date"],
                "source": "cache"
            }
        
        # Get all data from cache
        logger.info(f"Getting all cached data for {symbol}")
        data, _ = cache.get_cached_data(symbol)
        
        if not data or not data.get('records'):
            raise ValueError(f"No cached data available for {symbol}")
            
        # Get all records and sort by date
        records = sorted(data['records'], key=lambda x: x['date'])
        
        if len(records) < 2:
            raise ValueError(f"Insufficient data for {symbol}")
            
        # Get latest and previous record
        latest = records[-1]
        prev = records[-2]
        
        # Calculate change percentage
        latest_close = float(latest['close'])
        prev_close = float(prev['close'])
        change_pct = ((latest_close - prev_close) / prev_close) * 100
        
        return {
            "price": latest_close,
            "change": change_pct,
            "volume": int(latest['volume']),
            "timestamp": latest['date'],
            "source": "cache",
            "total_records": len(records)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": f"Error fetching price for {symbol}", "error": str(e)}
        )

@router.post("/debug-analyze", response_model=StockResponse)
async def debug_analyze(request: StockRequest):
    """Debug endpoint that returns mock data for testing frontend"""
    logger.info(f"[Debug] Returning mock data for {request.ticker}")
    return StockResponse(
        chart_data=ChartData(
            dates=["2024-01-01", "2024-01-02"],
            prices=[100.0, 102.5],
            volumes=[1000000, 1200000],
            ma20=[99.5, 100.5],
            ma50=[98.0, 98.5],
            ma100=[97.0, 97.5],
            volume_ma20=[1100000, 1150000],
            swing_highs=[None, 103.0],
            swing_lows=[99.0, None],
            strong_buy_signals=[False, True],
            volume_colors=["green", "red"]
        ),
        metrics=AnalysisMetrics(
            higher_lows_count=2,
            higher_highs_count=3,
            avg_days_between_swings=4.0,
            avg_pct_diff_highs=8.1,
            avg_pct_diff_lows=7.9,
            score_percentage=78.0,
            conditions_met={
                "recent_swings": True,
                "avg_days": True,
                "amplitude": True,
                "ma_alignment": True,
                "price_above_ma20": True,
                "volume_spike": True
            }
        ),
        debug_messages=["Mock analysis completed successfully"]
    )

@router.post("/add", response_model=AddStockResponse)
async def add_stock(request: AddStockRequest):
    """Add a new stock to track"""
    try:
        symbol = request.symbol
        logger.info(f"Received request to add stock {symbol}")
        logger.info(f"Request data: {request.dict()}")
        
        # Initialize services
        registry = StockRegistry()
        cache = StockDataCache()
        
        # Check current status
        existing_stock = registry.get_stock_info(symbol)
        cached_data, _ = cache.get_cached_data(symbol)
        
        status = {
            "in_registry": bool(existing_stock),
            "has_cache": bool(cached_data),
            "needs_download": not cached_data
        }
        logger.info(f"Stock {symbol} status: {status}")
        
        # If stock exists but no cache, we'll download data
        if existing_stock and not cached_data:
            logger.info(f"Stock {symbol} found in registry but needs data download")
            stock_info = existing_stock  # Use existing registry entry
        elif existing_stock and cached_data:
            logger.info(f"Stock {symbol} is complete with data")
            return AddStockResponse(
                message=f"Stock {symbol} already exists with data",
                stock_info=existing_stock,
                data_status={"available": True, "status": status}
            )
        
        # Validate symbol
        logger.info(f"Starting validation for symbol {symbol}")
        validator = SymbolValidator()
        is_valid, company_info = await validator.validate_symbol(symbol)
        
        if not is_valid:
            logger.warning(f"Symbol validation failed for {symbol}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
            
        logger.info(f"Symbol {symbol} validated successfully. Company info: {company_info}")
        
        # Add company info from request if provided
        if request.name:
            company_info['name'] = request.name
        if request.description:
            company_info['description'] = request.description
            
        # Add or get from registry
        if existing_stock:
            stock_info = existing_stock
            logger.info(f"Using existing registry entry for {symbol}")
        else:
            stock_info = registry.add_stock(
                symbol=symbol,
                segment=request.segment,
                info=company_info
            )
            logger.info(f"Added new registry entry for {symbol}")
        
        # Check data availability
        data_status = validator.check_data_availability(symbol)
        
        if not data_status['available']:
            raise HTTPException(
                status_code=400,
                detail=f"No data available for {symbol}"
            )
            
        # Download and cache data if needed
        if not cached_data:
            logger.info(f"Downloading data for stock {symbol}")
            client = YFinanceClient()
            # Download data from 2024-01-01 to today
            start_date = date(2024, 1, 1)
            end_date = date.today()
            df = await client.download_stock_data(symbol, start_date, end_date)
            if df is None:
                raise ValueError(f"Failed to download data for {symbol}")
                
            # Save the downloaded data to cache
            cache.save_to_cache(symbol, df)  # First download, no existing data to merge
            logger.info(f"Successfully downloaded and cached data for {symbol}")
            
            # Verify data was cached
            cached_data, _ = cache.get_cached_data(symbol)
            if not cached_data or not cached_data.get('records'):
                raise ValueError("Data download succeeded but cache verification failed")
                
            logger.info(f"Verified cache data for {symbol}: {len(cached_data['records'])} records")
            
            # Update data status after download
            data_status = validator.check_data_availability(symbol)
            data_status["just_downloaded"] = True
            data_status["records_count"] = len(cached_data.get('records', []))
        
        return AddStockResponse(
            message=f"Successfully added {symbol}",
            stock_info=stock_info,
            data_status=data_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding stock {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding stock: {str(e)}"
        )

@router.get("/registry", response_model=List[StockInfo])
async def get_registry():
    """Get all stocks from registry"""
    try:
        registry = StockRegistry()
        return registry.get_all_stocks()
    except Exception as e:
        logger.error(f"Error getting registry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting registry: {str(e)}"
        )

@router.get("/config")
async def get_stock_config():
    """Get stock configuration including segments and version"""
    try:
        config_file = Path("app/config/stocks.json")
        if not config_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Stock configuration not found"
            )
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        return config
        
    except Exception as e:
        logger.error(f"Error reading stock config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading stock configuration: {str(e)}"
        )

@router.get("/health")
async def health_check():
    return {"status": "healthy"}