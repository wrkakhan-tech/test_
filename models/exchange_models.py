from enum import Enum
from typing import Dict, Set

class Exchange(str, Enum):
    NYSE = "NYSE"  # New York Stock Exchange
    NASDAQ = "NASDAQ"  # NASDAQ Global Select
    NASDAQ_CM = "NCM"  # NASDAQ Capital Market
    NASDAQ_GM = "NGM"  # NASDAQ Global Market
    AMEX = "AMEX"  # American Stock Exchange
    NYQ = "NYQ"   # NYSE (alternate code)
    NMS = "NMS"   # NASDAQ Global Market (alternate)
    NCM = "NCM"   # NASDAQ Capital Market (alternate)
    NGM = "NGM"   # NASDAQ Global Market (alternate)

# Mapping of exchange codes to standardized names
EXCHANGE_MAPPING: Dict[str, str] = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "NCM": "NASDAQ",  # NASDAQ Capital Market
    "NGM": "NASDAQ",  # NASDAQ Global Market
    "NMS": "NASDAQ",  # NASDAQ Global Select
    "NYQ": "NYSE",    # NYSE (alternate)
    "AMEX": "AMEX",
    "NYSE ARCA": "NYSE",
    "NASDAQ-GM": "NASDAQ",
    "NASDAQ-CM": "NASDAQ",
    "NASDAQ-GS": "NASDAQ"
}

# Set of all valid exchange codes
VALID_EXCHANGES: Set[str] = set(EXCHANGE_MAPPING.keys())

def normalize_exchange(exchange: str) -> str:
    """Normalize exchange code to standard name"""
    return EXCHANGE_MAPPING.get(exchange.upper(), exchange.upper())