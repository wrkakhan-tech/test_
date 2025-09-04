from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum

class StockSegment(str, Enum):
    QUANTUM = "Quantum"
    CHIP = "Chip"
    BITCOIN_MINERS = "Bitcoin Miners"
    AI = "AI"
    FINANCE = "Finance"
    TECH = "Tech"
    ECOMMERCE_FOOD = "E-commerce/Food"
    CONSUMER_GOODS = "Consumer Goods"
    TRAVEL = "Travel"
    SPACE = "Space"
    CYBERSECURITY = "Cybersecurity"
    ENERGY = "Energy"
    FIVE_G = "5G"
    ADTECH = "Ad-tech"
    TRANSPORT = "Transport"
    ROBOTICS = "Robotics"
    CAR_ECOMMERCE = "Car E-commerce"
    SHOE_APPAREL = "Shoe/Apparel"
    HEALTH_TECH = "Health Tech"
    CLOUD_TECH = "Cloud Tech"
    OIL_AND_GAS = "Oil and Gas"
    DATA_CENTERS = "Data Centers"
    DATA_ANALYTICS = "Data Analytics"
    MINERALS_TECH = "Minerals Tech"
    HEALTHCARE = "Healthcare"
    FOOD_RESTAURANTS = "Food/Restaurants"

class StockInfo(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    segment: StockSegment = Field(..., description="Stock segment/category")
    name: Optional[str] = Field(None, description="Company name")
    description: Optional[str] = Field(None, description="Brief company description")
    notes: Optional[str] = Field(None, description="Additional trading notes")

class StockRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    window: int = Field(5, ge=1, le=20, description="Window size for swing analysis")
    start_date: date = Field(..., description="Start date for analysis")
    end_date: date = Field(..., description="End date for analysis")
    segment: Optional[StockSegment] = Field(None, description="Stock segment/category")

class ChartData(BaseModel):
    dates: List[str]  # ISO format datetime strings
    prices: List[float]
    volumes: List[int]
    ma20: List[Optional[float]]
    ma50: List[Optional[float]]
    ma100: List[Optional[float]]
    volume_ma20: List[Optional[float]]
    swing_highs: List[Optional[float]]
    swing_lows: List[Optional[float]]
    strong_buy_signals: List[bool]
    volume_colors: List[str]

class AnalysisMetrics(BaseModel):
    higher_lows_count: int
    higher_highs_count: int
    avg_days_between_swings: Optional[float]
    avg_pct_diff_highs: Optional[float]
    avg_pct_diff_lows: Optional[float]
    score_percentage: float
    conditions_met: dict

class StockResponse(BaseModel):
    chart_data: ChartData
    metrics: AnalysisMetrics
    debug_messages: List[str]
    stock_info: Optional[StockInfo] = None

class SegmentResponse(BaseModel):
    segments: Dict[StockSegment, List[StockInfo]]
    total_stocks: int

class AddStockRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol to add")
    segment: StockSegment = Field(..., description="Stock segment/category")
    name: Optional[str] = Field(None, description="Company name")
    description: Optional[str] = Field(None, description="Company description")

    @validator('symbol')
    def validate_symbol_format(cls, v):
        if not v.isalnum():
            raise ValueError("Symbol must be alphanumeric")
        return v.upper()

class AddStockResponse(BaseModel):
    message: str
    stock_info: StockInfo
    data_status: Dict[str, Any]