import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    app_name: str = "Personal Finance AI"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Security
    secret_key: str
    jwt_secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # Database
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]
    
    # API Keys
    alpha_vantage_api_key: Optional[str] = None
    nse_api_key: Optional[str] = None
    yahoo_finance_api_key: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    
    # Email
    from_email: str = "noreply@financeai.com"
    
    # File Upload
    max_file_size: int = 10485760  # 10MB
    upload_folder: str = "./uploads/"
    allowed_extensions: List[str] = ["csv", "xlsx", "xls", "pdf"]
    
    # ML Models
    ml_model_path: str = "./models/"
    retrain_models: bool = False
    model_update_frequency: int = 30
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    log_level: str = "INFO"
    
    # Feature Flags
    enable_email_notifications: bool = True
    enable_sms_alerts: bool = False
    enable_market_data: bool = True
    enable_pdf_parsing: bool = True
    enable_bulk_upload: bool = True
    
    # Indian Market Specific
    default_currency: str = "INR"
    default_city: str = "Mumbai"
    inflation_rate: float = 0.06
    supported_exchanges: List[str] = ["NSE", "BSE"]
    
    # Cache
    cache_ttl: int = 3600
    cache_max_entries: int = 10000
    
    # Indian Cities with Cost of Living Index
    city_cost_index: dict = {
        "Mumbai": 1.0, "Delhi": 0.95, "Bangalore": 0.88, "Chennai": 0.82,
        "Pune": 0.78, "Hyderabad": 0.73, "Kolkata": 0.75, "Ahmedabad": 0.70,
        "Jaipur": 0.65, "Lucknow": 0.60, "Indore": 0.58, "Bhopal": 0.55,
        "Nagpur": 0.57, "Visakhapatnam": 0.52, "Ludhiana": 0.54, "Agra": 0.50,
        "Nashik": 0.56, "Vadodara": 0.59, "Rajkot": 0.53, "Varanasi": 0.48,
        "Srinagar": 0.51, "Aurangabad": 0.54, "Dhanbad": 0.49, "Amritsar": 0.52,
        "Allahabad": 0.47, "Ranchi": 0.50, "Howrah": 0.48, "Coimbatore": 0.55,
        "Jabalpur": 0.46, "Gwalior": 0.45, "Vijayawada": 0.51, "Jodhpur": 0.48,
        "Madurai": 0.50, "Raipur": 0.47, "Kota": 0.49, "Chandigarh": 0.62,
        "Guwahati": 0.52, "Solapur": 0.48, "Hubli": 0.50, "Tiruchirappalli": 0.49,
        "Bareilly": 0.44, "Moradabad": 0.43, "Mysore": 0.52, "Gurgaon": 0.85,
        "Aligarh": 0.42, "Jalandhar": 0.50, "Bhubaneswar": 0.51, "Salem": 0.48,
        "Mira-Bhayandar": 0.82, "Warangal": 0.46, "Thiruvananthapuram": 0.54,
        "Guntur": 0.45, "Bhiwandi": 0.75, "Saharanpur": 0.41, "Gorakhpur": 0.40
    }
    
    @validator('cors_origins', pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('allowed_hosts', pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('allowed_extensions', pre=True)
    def assemble_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('supported_exchanges', pre=True)
    def assemble_supported_exchanges(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Indian financial constants
INDIAN_FINANCIAL_CATEGORIES = [
    "Housing", "Food & Groceries", "Transport", "Utilities", 
    "Entertainment", "Self Care", "Healthcare", "Education",
    "Insurance", "Investments", "Shopping", "Travel", "Other"
]

INDIAN_PAYMENT_PLATFORMS = [
    "paytm", "phonepe", "googlepay", "bhim", "mobikwik", "freecharge",
    "amazonpay", "jiomoney", "airtel money", "ola money", "uber"
]

INDIAN_BANKS = [
    "sbi", "hdfc", "icici", "axis", "kotak", "indusind", "yes bank",
    "pnb", "bob", "canara", "union bank", "indian bank", "boi"
]

INDIAN_FOOD_PLATFORMS = [
    "zomato", "swiggy", "uber eats", "foodpanda", "dominos", "pizza hut",
    "kfc", "mcdonalds", "subway", "dunkin", "starbucks", "cafe coffee day"
]

INDIAN_TRANSPORT_PLATFORMS = [
    "ola", "uber", "rapido", "bounce", "vogo", "yulu", "metro", "bus",
    "auto", "taxi", "cab", "rickshaw", "train", "flight", "irctc"
]

INDIAN_SHOPPING_PLATFORMS = [
    "amazon", "flipkart", "myntra", "ajio", "nykaa", "bigbasket",
    "grofers", "dunzo", "1mg", "pharmeasy", "lenskart", "urban ladder"
]

# NSE/BSE Stock symbols mapping
POPULAR_INDIAN_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS", 
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "WIPRO": "WIPRO.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "POWERGRID": "POWERGRID.NS"
}

# Mutual Fund categories for Indian market
INDIAN_MUTUAL_FUND_CATEGORIES = [
    "Large Cap", "Mid Cap", "Small Cap", "Multi Cap", "Flexi Cap",
    "ELSS", "Hybrid", "Debt", "Liquid", "Ultra Short Duration",
    "Short Duration", "Medium Duration", "Long Duration", "Dynamic Bond",
    "Corporate Bond", "Credit Risk", "Banking & PSU", "Gilt",
    "Index Fund", "ETF", "Gold ETF", "International", "Sectoral"

]

