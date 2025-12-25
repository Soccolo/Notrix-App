import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import requests
from textblob import TextBlob
import feedparser
import os
from pathlib import Path
import logging
import json

# Suppress yfinance and urllib3 warnings/errors
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)

# Suppress peewee logger if present
try:
    logging.getLogger('peewee').setLevel(logging.CRITICAL)
except:
    pass

# ============ API CONFIGURATION ============
# Get free API keys from:
# - Finnhub: https://finnhub.io/ (60 calls/min free)
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key (25 calls/day free)
# - Twelve Data: https://twelvedata.com/ (800 calls/day free)
# - FMP: https://site.financialmodelingprep.com/developer/docs (250 calls/day free)

# You can set these as environment variables or enter them in the sidebar
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

# Page config
st.set_page_config(
    page_title="Notrix Investment Fund",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .ticker-bar {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #0f3460;
    }
    .ticker-item {
        display: inline-block;
        margin: 0 20px;
        font-family: 'Courier New', monospace;
    }
    .price-up { color: #00ff88; }
    .price-down { color: #ff4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #0f3460;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f3460;
    }
    .news-card {
        background: #1a1a2e;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    .sentiment-positive { color: #00ff88; font-weight: bold; }
    .sentiment-negative { color: #ff4444; font-weight: bold; }
    .sentiment-neutral { color: #ffaa00; font-weight: bold; }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============ ALTERNATIVE DATA SOURCE FUNCTIONS ============

def safe_get_column(df, col_name):
    """Safely get a column from DataFrame that might have MultiIndex columns"""
    if df is None or len(df) == 0:
        return None
    try:
        # If it's a MultiIndex column DataFrame from yf.download
        if isinstance(df.columns, pd.MultiIndex):
            # Try to get the column for any ticker
            if col_name in df.columns.get_level_values(0):
                return df[col_name].iloc[:, 0] if isinstance(df[col_name], pd.DataFrame) else df[col_name]
            # Try the second level
            for col in df.columns:
                if col[0] == col_name or col[1] == col_name:
                    return df[col]
        # Regular DataFrame
        if col_name in df.columns:
            col_data = df[col_name]
            # If it's still a DataFrame with one column, convert to Series
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            return col_data
    except Exception:
        pass
    return None

def safe_get_value(df, col_name, idx=-1):
    """Safely get a single value from a DataFrame column"""
    col = safe_get_column(df, col_name)
    if col is not None and len(col) > 0:
        try:
            val = col.iloc[idx]
            return float(val)
        except:
            pass
    return None

def normalize_dataframe(df):
    """Normalize DataFrame columns to simple format (handle MultiIndex from yf.download)"""
    if df is None or len(df) == 0:
        return df
    
    try:
        # If MultiIndex columns, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            # Get the first level of column names
            new_df = pd.DataFrame(index=df.index)
            for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                col_data = safe_get_column(df, col_name)
                if col_data is not None:
                    new_df[col_name] = col_data
            return new_df
        return df
    except Exception:
        return df

def get_finnhub_quote(symbol, api_key):
    """Get quote from Finnhub API"""
    if not api_key:
        return None
    try:
        # Finnhub free tier works best with US stocks/ETFs
        # Map commodities to ETFs, indices to ETFs
        symbol_map = {
            'GC=F': 'GLD',   # Gold ETF (instead of forex)
            'SI=F': 'SLV',   # Silver ETF
            'HG=F': 'CPER',  # Copper ETF
            '^IXIC': 'QQQ',  # NASDAQ ETF proxy
            '^DJI': 'DIA',   # Dow Jones ETF proxy
        }
        
        finnhub_symbol = symbol_map.get(symbol, symbol)
        if finnhub_symbol is None:
            return None
            
        url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Finnhub returns: c=current, pc=previous close, d=change, dp=percent change
            if data and data.get('c') and data.get('c') > 0:
                return {
                    'price': float(data['c']),
                    'change': float(data.get('dp', 0)),
                    'prev_close': float(data.get('pc', data['c']))
                }
    except Exception as e:
        pass
    return None

def get_twelve_data_quote(symbol, api_key):
    """Get quote from Twelve Data API"""
    if not api_key:
        return None
    try:
        # Twelve Data symbol mapping
        symbol_map = {
            'GC=F': 'XAU/USD',   # Gold
            'SI=F': 'XAG/USD',   # Silver
            'HG=F': None,        # Copper - limited support
            '^IXIC': 'IXIC',     # NASDAQ Composite
            '^DJI': 'DJI',       # Dow Jones
        }
        
        td_symbol = symbol_map.get(symbol, symbol)
        if td_symbol is None:
            return None
        
        url = f"https://api.twelvedata.com/quote?symbol={td_symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and 'close' in data and data.get('close'):
                try:
                    current = float(data['close'])
                    prev = float(data.get('previous_close', current))
                    change_pct = float(data.get('percent_change', 0))
                    if change_pct == 0 and prev > 0:
                        change_pct = ((current - prev) / prev) * 100
                    return {
                        'price': current,
                        'change': change_pct,
                        'prev_close': prev
                    }
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        pass
    return None

def get_alpha_vantage_quote(symbol, api_key):
    """Get quote from Alpha Vantage API"""
    if not api_key:
        return None
    try:
        # Alpha Vantage - mainly for stocks, limited commodity support
        if symbol in ['GC=F', 'SI=F', 'HG=F']:
            return None  # Use other APIs for commodities
        
        # Clean symbol for Alpha Vantage
        clean_symbol = symbol.replace('^', '').replace('=F', '')
        if symbol == '^IXIC':
            clean_symbol = 'QQQ'  # Use QQQ as proxy
        elif symbol == '^DJI':
            clean_symbol = 'DIA'  # Use DIA as proxy
        
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={clean_symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            quote = data.get('Global Quote', {})
            if quote and quote.get('05. price'):
                try:
                    price = float(quote['05. price'])
                    change_str = quote.get('10. change percent', '0%').replace('%', '')
                    change = float(change_str)
                    prev = float(quote.get('08. previous close', price))
                    return {
                        'price': price,
                        'change': change,
                        'prev_close': prev
                    }
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        pass
    return None

def get_fmp_quote(symbol, api_key):
    """Get quote from Financial Modeling Prep API"""
    if not api_key:
        return None
    try:
        # FMP symbol mapping
        symbol_map = {
            'GC=F': 'GCUSD',     # Gold
            'SI=F': 'SIUSD',     # Silver  
            'HG=F': 'HGUSD',     # Copper
            '^IXIC': 'QQQ',      # NASDAQ proxy
            '^DJI': 'DIA',       # Dow proxy
        }
        
        fmp_symbol = symbol_map.get(symbol, symbol)
        
        url = f"https://financialmodelingprep.com/api/v3/quote/{fmp_symbol}?apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                quote = data[0]
                if quote.get('price'):
                    return {
                        'price': float(quote['price']),
                        'change': float(quote.get('changesPercentage', 0)),
                        'prev_close': float(quote.get('previousClose', quote['price']))
                    }
    except Exception as e:
        pass
    return None

def fetch_from_yahoo_download(symbol, period='5d'):
    """Use yf.download which sometimes works better than Ticker.history"""
    try:
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if data is not None and len(data) > 0:
            return data
    except Exception:
        pass
    return None

def fetch_from_ticker_history(symbol, period='5d'):
    """Fallback to Ticker.history method"""
    try:
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if data is not None and len(data) > 0:
            return data
    except Exception:
        pass
    return None

def safe_yf_download(symbol, period='5d', max_retries=2, delay=1):
    """Safely download data from Yahoo Finance with multiple methods"""
    for attempt in range(max_retries):
        data = fetch_from_yahoo_download(symbol, period)
        if data is not None and len(data) > 0:
            return normalize_dataframe(data)
        time.sleep(delay)
    
    for attempt in range(max_retries):
        data = fetch_from_ticker_history(symbol, period)
        if data is not None and len(data) > 0:
            return normalize_dataframe(data)
        time.sleep(delay)
    
    return None

def safe_get_info(symbol, max_retries=2, delay=1):
    """Safely get ticker info with retry logic"""
    import io
    import sys
    
    for attempt in range(max_retries):
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            return info
        except Exception:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if attempt < max_retries - 1:
                time.sleep(delay)
    return {}

def get_multi_source_quote(symbol, api_keys, debug_info=None):
    """Try multiple data sources in order of reliability"""
    
    # Track which APIs we tried for debugging
    tried = []
    
    # Try Finnhub first (most generous free tier)
    if api_keys.get('finnhub'):
        tried.append('Finnhub')
        quote = get_finnhub_quote(symbol, api_keys['finnhub'])
        if quote and quote.get('price', 0) > 0:
            if debug_info is not None:
                debug_info['tried'] = tried
            return quote, 'Finnhub'
    
    # Try Twelve Data
    if api_keys.get('twelve_data'):
        tried.append('Twelve Data')
        quote = get_twelve_data_quote(symbol, api_keys['twelve_data'])
        if quote and quote.get('price', 0) > 0:
            if debug_info is not None:
                debug_info['tried'] = tried
            return quote, 'Twelve Data'
    
    # Try Alpha Vantage
    if api_keys.get('alpha_vantage'):
        tried.append('Alpha Vantage')
        quote = get_alpha_vantage_quote(symbol, api_keys['alpha_vantage'])
        if quote and quote.get('price', 0) > 0:
            if debug_info is not None:
                debug_info['tried'] = tried
            return quote, 'Alpha Vantage'
    
    # Try FMP
    if api_keys.get('fmp'):
        tried.append('FMP')
        quote = get_fmp_quote(symbol, api_keys['fmp'])
        if quote and quote.get('price', 0) > 0:
            if debug_info is not None:
                debug_info['tried'] = tried
            return quote, 'FMP'
    
    # Try Yahoo Finance as last resort
    tried.append('Yahoo')
    hist = safe_yf_download(symbol, period='5d', max_retries=1, delay=0.5)
    if hist is not None and len(hist) >= 1:
        try:
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[0]) if len(hist) > 1 else current
            change = ((current - prev) / prev) * 100 if prev > 0 else 0
            if debug_info is not None:
                debug_info['tried'] = tried
            return {'price': current, 'change': change, 'prev_close': prev}, 'Yahoo'
        except:
            pass
    
    if debug_info is not None:
        debug_info['tried'] = tried
    return None, None

# ============ CACHED/FALLBACK DATA ============

def get_cached_market_data():
    """Return cached/static market data as fallback with recent approximate values"""
    return {
        'Gold': {'price': 2650.00, 'change': 0.0, 'symbol': 'GC=F', 'available': False, 'cached': True, 'source': 'Cached'},
        'Silver': {'price': 31.50, 'change': 0.0, 'symbol': 'SI=F', 'available': False, 'cached': True, 'source': 'Cached'},
        'Copper': {'price': 4.25, 'change': 0.0, 'symbol': 'HG=F', 'available': False, 'cached': True, 'source': 'Cached'},
        'SPY': {'price': 595.00, 'change': 0.0, 'symbol': 'SPY', 'available': False, 'cached': True, 'source': 'Cached'},
        'NASDAQ': {'price': 19800.00, 'change': 0.0, 'symbol': '^IXIC', 'available': False, 'cached': True, 'source': 'Cached'},
        'Dow Jones': {'price': 42800.00, 'change': 0.0, 'symbol': '^DJI', 'available': False, 'cached': True, 'source': 'Cached'}
    }

# ============ MARKET DATA FUNCTIONS ============

def get_market_data(api_keys, show_debug=False):
    """Fetch current prices for major indices and commodities from multiple sources"""
    symbols = {
        'Gold': 'GC=F',
        'Silver': 'SI=F', 
        'Copper': 'HG=F',
        'SPY': 'SPY',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI'
    }
    
    # Start with empty data - don't pre-fill with cached
    data = {}
    debug_messages = []
    
    for name, symbol in symbols.items():
        debug_info = {}
        quote, source = get_multi_source_quote(symbol, api_keys, debug_info)
        
        if quote and quote.get('price', 0) > 0:
            data[name] = {
                'price': quote['price'],
                'change': quote['change'],
                'symbol': symbol,
                'available': True,
                'cached': False,
                'source': source
            }
            debug_messages.append(f"✅ {name}: Got data from {source}")
        else:
            # Only use cached data if API fails
            cached = get_cached_market_data()
            data[name] = cached[name]
            tried_apis = debug_info.get('tried', ['None'])
            debug_messages.append(f"❌ {name}: Failed (tried: {', '.join(tried_apis)}) - using cached")
        
        time.sleep(0.2)  # Small delay between requests
    
    if show_debug:
        return data, debug_messages
    return data

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(symbol, period='1y'):
    """Fetch historical stock data - tries Yahoo Finance first, then alternatives"""
    hist = safe_yf_download(symbol, period=period, max_retries=3, delay=1)
    info = safe_get_info(symbol, max_retries=2, delay=1)
    return hist, info

def get_stock_data_multi_source(symbol, period='1y', api_keys=None):
    """Fetch stock data from multiple sources with fallbacks"""
    if api_keys is None:
        api_keys = {}
    
    # First try Yahoo Finance (best for historical data)
    hist = safe_yf_download(symbol, period=period, max_retries=2, delay=1)
    info = safe_get_info(symbol, max_retries=1, delay=1)
    
    if hist is not None and len(hist) > 0:
        return hist, info, 'Yahoo Finance'
    
    # If Yahoo fails, try to get at least current quote from alternatives
    quote, source = get_multi_source_quote(symbol, api_keys)
    if quote:
        # Create a minimal dataframe with the current quote
        today = datetime.now()
        hist = pd.DataFrame({
            'Open': [quote['prev_close']],
            'High': [quote['price']],
            'Low': [quote['prev_close']],
            'Close': [quote['price']],
            'Volume': [0]
        }, index=[today])
        return hist, {}, source
    
    return None, None, None

@st.cache_data(ttl=600)
def get_stock_news(symbol):
    """Fetch news for a specific stock"""
    import io
    import sys
    try:
        time.sleep(0.3)  # Rate limiting
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if news:
            # Normalize the news format - Yahoo Finance changed their structure
            normalized_news = []
            for item in news[:10]:
                # Handle both old and new Yahoo Finance news formats
                if isinstance(item, dict):
                    # Try to extract title from various possible keys
                    title = (item.get('title') or 
                            item.get('headline') or 
                            item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else None or
                            'No title')
                    
                    # Try to extract link from various possible keys
                    link = (item.get('link') or 
                           item.get('url') or 
                           item.get('content', {}).get('canonicalUrl', {}).get('url') if isinstance(item.get('content'), dict) else None or
                           '#')
                    
                    # Try to extract publisher from various possible keys
                    publisher = (item.get('publisher') or 
                                item.get('source') or
                                item.get('content', {}).get('provider', {}).get('displayName') if isinstance(item.get('content'), dict) else None or
                                'Unknown')
                    
                    # Handle nested content structure (new format)
                    if item.get('content') and isinstance(item.get('content'), dict):
                        content = item['content']
                        if not title or title == 'No title':
                            title = content.get('title', 'No title')
                        if link == '#':
                            canonical = content.get('canonicalUrl', {})
                            if isinstance(canonical, dict):
                                link = canonical.get('url', '#')
                            elif isinstance(canonical, str):
                                link = canonical
                        if publisher == 'Unknown':
                            provider = content.get('provider', {})
                            if isinstance(provider, dict):
                                publisher = provider.get('displayName', 'Unknown')
                    
                    normalized_news.append({
                        'title': title,
                        'link': link,
                        'publisher': publisher
                    })
            
            return normalized_news if normalized_news else []
        return []
    except Exception as e:
        return []

def get_finnhub_news(symbol, api_key):
    """Get news from Finnhub API"""
    if not api_key:
        return []
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today}&token={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            news = response.json()
            if news and isinstance(news, list):
                return [{
                    'title': n.get('headline', 'No title'),
                    'link': n.get('url', '#'),
                    'publisher': n.get('source', 'Unknown'),
                    'summary': n.get('summary', '')
                } for n in news[:10]]
    except Exception as e:
        pass
    return []

def get_stock_news_multi_source(symbol, api_keys=None):
    """Get news from multiple sources - prioritize Finnhub for better news data"""
    if api_keys is None:
        api_keys = {}
    
    # Try Finnhub first (better news quality)
    if api_keys.get('finnhub'):
        news = get_finnhub_news(symbol, api_keys['finnhub'])
        if news and len(news) > 0 and news[0].get('title') != 'No title':
            return news, 'Finnhub'
    
    # Try Yahoo as fallback
    news = get_stock_news(symbol)
    if news and len(news) > 0 and news[0].get('title') != 'No title':
        return news, 'Yahoo'
    
    return [], None

# ============ FUNDAMENTAL ANALYSIS FUNCTIONS ============

def get_fundamental_data(symbol, api_keys=None):
    """Fetch comprehensive fundamental data for a stock"""
    if api_keys is None:
        api_keys = {}
    
    fundamentals = {
        'valuation': {},
        'profitability': {},
        'financial_health': {},
        'growth': {},
        'dividends': {},
        'analyst': {},
        'source': None
    }
    
    try:
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if info:
            # Valuation Metrics
            fundamentals['valuation'] = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
            }
            
            # Profitability Metrics
            fundamentals['profitability'] = {
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'ebitda_margin': info.get('ebitdaMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue': info.get('totalRevenue'),
                'net_income': info.get('netIncomeToCommon'),
                'eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
            }
            
            # Financial Health
            fundamentals['financial_health'] = {
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
            }
            
            # Growth Metrics
            fundamentals['growth'] = {
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                'revenue_per_share': info.get('revenuePerShare'),
                'book_value': info.get('bookValue'),
            }
            
            # Dividends
            fundamentals['dividends'] = {
                'dividend_rate': info.get('dividendRate'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'ex_dividend_date': info.get('exDividendDate'),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
            }
            
            # Analyst Recommendations
            fundamentals['analyst'] = {
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'target_mean': info.get('targetMeanPrice'),
                'target_median': info.get('targetMedianPrice'),
                'recommendation': info.get('recommendationKey'),
                'recommendation_mean': info.get('recommendationMean'),
                'num_analysts': info.get('numberOfAnalystOpinions'),
            }
            
            # Company Info
            fundamentals['company_info'] = {
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else '',
            }
            
            fundamentals['source'] = 'Yahoo Finance'
            
    except Exception as e:
        pass
    
    return fundamentals

def get_financial_statements(symbol):
    """Fetch financial statements for Piotroski F-Score calculation"""
    try:
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        ticker = yf.Ticker(symbol)
        
        # Get financial statements
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        return {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
    except Exception as e:
        return None

def calculate_piotroski_score(symbol):
    """
    Calculate Piotroski F-Score (0-9 points)
    
    Profitability (4 points):
    1. Positive ROA
    2. Positive Operating Cash Flow
    3. ROA increasing (vs prior year)
    4. Cash Flow > Net Income (accruals)
    
    Leverage/Liquidity (3 points):
    5. Decrease in long-term debt ratio
    6. Increase in current ratio
    7. No new shares issued
    
    Operating Efficiency (2 points):
    8. Increase in gross margin
    9. Increase in asset turnover
    """
    
    score = 0
    details = []
    
    try:
        statements = get_financial_statements(symbol)
        if not statements:
            return None, ["Could not fetch financial statements"]
        
        income_stmt = statements['income_statement']
        balance_sheet = statements['balance_sheet']
        cash_flow = statements['cash_flow']
        
        if income_stmt is None or balance_sheet is None or cash_flow is None:
            return None, ["Missing financial data"]
        
        if len(income_stmt.columns) < 2 or len(balance_sheet.columns) < 2:
            return None, ["Insufficient historical data (need at least 2 years)"]
        
        # Helper function to safely get values
        def safe_get(df, row_names, col_idx=0):
            if df is None:
                return None
            for name in row_names if isinstance(row_names, list) else [row_names]:
                if name in df.index:
                    try:
                        val = df.loc[name].iloc[col_idx]
                        return float(val) if pd.notna(val) else None
                    except:
                        continue
            return None
        
        # Current and prior year indices
        curr = 0
        prior = 1
        
        # Get values - trying multiple possible row names
        net_income_curr = safe_get(income_stmt, ['Net Income', 'Net Income Common Stockholders', 'NetIncome'], curr)
        net_income_prior = safe_get(income_stmt, ['Net Income', 'Net Income Common Stockholders', 'NetIncome'], prior)
        
        total_assets_curr = safe_get(balance_sheet, ['Total Assets', 'TotalAssets'], curr)
        total_assets_prior = safe_get(balance_sheet, ['Total Assets', 'TotalAssets'], prior)
        
        operating_cf_curr = safe_get(cash_flow, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'OperatingCashFlow'], curr)
        
        long_term_debt_curr = safe_get(balance_sheet, ['Long Term Debt', 'LongTermDebt', 'Long Term Debt And Capital Lease Obligation'], curr)
        long_term_debt_prior = safe_get(balance_sheet, ['Long Term Debt', 'LongTermDebt', 'Long Term Debt And Capital Lease Obligation'], prior)
        
        current_assets_curr = safe_get(balance_sheet, ['Current Assets', 'Total Current Assets', 'CurrentAssets'], curr)
        current_assets_prior = safe_get(balance_sheet, ['Current Assets', 'Total Current Assets', 'CurrentAssets'], prior)
        
        current_liab_curr = safe_get(balance_sheet, ['Current Liabilities', 'Total Current Liabilities', 'CurrentLiabilities'], curr)
        current_liab_prior = safe_get(balance_sheet, ['Current Liabilities', 'Total Current Liabilities', 'CurrentLiabilities'], prior)
        
        shares_curr = safe_get(balance_sheet, ['Share Issued', 'Common Stock Shares Outstanding', 'Ordinary Shares Number'], curr)
        shares_prior = safe_get(balance_sheet, ['Share Issued', 'Common Stock Shares Outstanding', 'Ordinary Shares Number'], prior)
        
        gross_profit_curr = safe_get(income_stmt, ['Gross Profit', 'GrossProfit'], curr)
        gross_profit_prior = safe_get(income_stmt, ['Gross Profit', 'GrossProfit'], prior)
        
        revenue_curr = safe_get(income_stmt, ['Total Revenue', 'Revenue', 'TotalRevenue'], curr)
        revenue_prior = safe_get(income_stmt, ['Total Revenue', 'Revenue', 'TotalRevenue'], prior)
        
        # === PROFITABILITY (4 points) ===
        
        # 1. Positive ROA
        if net_income_curr is not None and total_assets_curr is not None and total_assets_curr > 0:
            roa_curr = net_income_curr / total_assets_curr
            if roa_curr > 0:
                score += 1
                details.append("✅ Positive ROA ({:.2%})".format(roa_curr))
            else:
                details.append("❌ Negative ROA ({:.2%})".format(roa_curr))
        else:
            details.append("⚠️ ROA: Insufficient data")
        
        # 2. Positive Operating Cash Flow
        if operating_cf_curr is not None:
            if operating_cf_curr > 0:
                score += 1
                details.append("✅ Positive Operating Cash Flow (${:,.0f})".format(operating_cf_curr))
            else:
                details.append("❌ Negative Operating Cash Flow (${:,.0f})".format(operating_cf_curr))
        else:
            details.append("⚠️ Operating Cash Flow: Insufficient data")
        
        # 3. ROA increasing
        if all(v is not None for v in [net_income_curr, net_income_prior, total_assets_curr, total_assets_prior]) and total_assets_curr > 0 and total_assets_prior > 0:
            roa_curr = net_income_curr / total_assets_curr
            roa_prior = net_income_prior / total_assets_prior
            if roa_curr > roa_prior:
                score += 1
                details.append("✅ ROA Increasing ({:.2%} → {:.2%})".format(roa_prior, roa_curr))
            else:
                details.append("❌ ROA Declining ({:.2%} → {:.2%})".format(roa_prior, roa_curr))
        else:
            details.append("⚠️ ROA Trend: Insufficient data")
        
        # 4. Operating Cash Flow > Net Income (Quality of Earnings)
        if operating_cf_curr is not None and net_income_curr is not None:
            if operating_cf_curr > net_income_curr:
                score += 1
                details.append("✅ Cash Flow > Net Income (Quality earnings)")
            else:
                details.append("❌ Cash Flow < Net Income (Accrual concerns)")
        else:
            details.append("⚠️ Accruals: Insufficient data")
        
        # === LEVERAGE & LIQUIDITY (3 points) ===
        
        # 5. Decrease in Long-term Debt Ratio
        if all(v is not None for v in [long_term_debt_curr, long_term_debt_prior, total_assets_curr, total_assets_prior]) and total_assets_curr > 0 and total_assets_prior > 0:
            debt_ratio_curr = long_term_debt_curr / total_assets_curr
            debt_ratio_prior = long_term_debt_prior / total_assets_prior
            if debt_ratio_curr <= debt_ratio_prior:
                score += 1
                details.append("✅ Debt Ratio Stable/Decreasing ({:.2%} → {:.2%})".format(debt_ratio_prior, debt_ratio_curr))
            else:
                details.append("❌ Debt Ratio Increasing ({:.2%} → {:.2%})".format(debt_ratio_prior, debt_ratio_curr))
        else:
            # No debt could be good
            if long_term_debt_curr == 0 or long_term_debt_curr is None:
                score += 1
                details.append("✅ No/Minimal Long-term Debt")
            else:
                details.append("⚠️ Debt Ratio: Insufficient data")
        
        # 6. Increase in Current Ratio
        if all(v is not None for v in [current_assets_curr, current_assets_prior, current_liab_curr, current_liab_prior]) and current_liab_curr > 0 and current_liab_prior > 0:
            curr_ratio_curr = current_assets_curr / current_liab_curr
            curr_ratio_prior = current_assets_prior / current_liab_prior
            if curr_ratio_curr >= curr_ratio_prior:
                score += 1
                details.append("✅ Current Ratio Stable/Improving ({:.2f} → {:.2f})".format(curr_ratio_prior, curr_ratio_curr))
            else:
                details.append("❌ Current Ratio Declining ({:.2f} → {:.2f})".format(curr_ratio_prior, curr_ratio_curr))
        else:
            details.append("⚠️ Current Ratio: Insufficient data")
        
        # 7. No Dilution (shares not increased)
        if shares_curr is not None and shares_prior is not None:
            if shares_curr <= shares_prior:
                score += 1
                details.append("✅ No Share Dilution")
            else:
                dilution = ((shares_curr - shares_prior) / shares_prior) * 100
                details.append("❌ Share Dilution ({:.1f}% increase)".format(dilution))
        else:
            details.append("⚠️ Share Count: Insufficient data")
        
        # === OPERATING EFFICIENCY (2 points) ===
        
        # 8. Gross Margin Increasing
        if all(v is not None for v in [gross_profit_curr, gross_profit_prior, revenue_curr, revenue_prior]) and revenue_curr > 0 and revenue_prior > 0:
            gm_curr = gross_profit_curr / revenue_curr
            gm_prior = gross_profit_prior / revenue_prior
            if gm_curr >= gm_prior:
                score += 1
                details.append("✅ Gross Margin Stable/Improving ({:.2%} → {:.2%})".format(gm_prior, gm_curr))
            else:
                details.append("❌ Gross Margin Declining ({:.2%} → {:.2%})".format(gm_prior, gm_curr))
        else:
            details.append("⚠️ Gross Margin: Insufficient data")
        
        # 9. Asset Turnover Increasing
        if all(v is not None for v in [revenue_curr, revenue_prior, total_assets_curr, total_assets_prior]) and total_assets_curr > 0 and total_assets_prior > 0:
            at_curr = revenue_curr / total_assets_curr
            at_prior = revenue_prior / total_assets_prior
            if at_curr >= at_prior:
                score += 1
                details.append("✅ Asset Turnover Stable/Improving ({:.2f} → {:.2f})".format(at_prior, at_curr))
            else:
                details.append("❌ Asset Turnover Declining ({:.2f} → {:.2f})".format(at_prior, at_curr))
        else:
            details.append("⚠️ Asset Turnover: Insufficient data")
        
        return score, details
        
    except Exception as e:
        return None, [f"Error calculating score: {str(e)}"]

def get_earnings_calendar(symbol, api_keys=None):
    """Get upcoming earnings dates for a stock"""
    if api_keys is None:
        api_keys = {}
    
    earnings_data = {
        'next_earnings': None,
        'earnings_history': [],
        'source': None
    }
    
    # Try Yahoo Finance first
    try:
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        earnings_dates = ticker.earnings_dates
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if calendar is not None and not calendar.empty:
            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.columns:
                    earnings_data['next_earnings'] = calendar['Earnings Date'].iloc[0] if len(calendar) > 0 else None
                elif len(calendar.columns) > 0:
                    # Try to find earnings date in the calendar
                    for col in calendar.columns:
                        if 'earning' in col.lower():
                            earnings_data['next_earnings'] = calendar[col].iloc[0]
                            break
        
        if earnings_dates is not None and not earnings_dates.empty:
            history = []
            for date, row in earnings_dates.head(8).iterrows():
                history.append({
                    'date': date,
                    'eps_estimate': row.get('EPS Estimate'),
                    'eps_actual': row.get('Reported EPS'),
                    'surprise': row.get('Surprise(%)'),
                })
            earnings_data['earnings_history'] = history
        
        earnings_data['source'] = 'Yahoo Finance'
        
    except Exception as e:
        pass
    
    # Try Finnhub if available
    if api_keys.get('finnhub') and not earnings_data['earnings_history']:
        try:
            url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={symbol}&token={api_keys['finnhub']}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('earningsCalendar'):
                    for item in data['earningsCalendar'][:8]:
                        earnings_data['earnings_history'].append({
                            'date': item.get('date'),
                            'eps_estimate': item.get('epsEstimate'),
                            'eps_actual': item.get('epsActual'),
                            'surprise': None,
                            'quarter': item.get('quarter'),
                        })
                    earnings_data['source'] = 'Finnhub'
        except:
            pass
    
    return earnings_data

def get_economic_calendar(api_keys=None):
    """Get upcoming economic events"""
    if api_keys is None:
        api_keys = {}
    
    events = []
    
    # Try Finnhub Economic Calendar
    if api_keys.get('finnhub'):
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            next_week = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={next_week}&token={api_keys['finnhub']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('economicCalendar'):
                    for item in data['economicCalendar'][:30]:
                        impact = item.get('impact', 'low')
                        events.append({
                            'date': item.get('time', item.get('date')),
                            'event': item.get('event'),
                            'country': item.get('country', 'US'),
                            'impact': impact,
                            'actual': item.get('actual'),
                            'estimate': item.get('estimate'),
                            'previous': item.get('prev'),
                            'unit': item.get('unit', ''),
                        })
                return events, 'Finnhub'
        except:
            pass
    
    # Fallback: Get from RSS feeds or hardcoded major events
    # This is a simplified fallback - major recurring events
    try:
        major_events = [
            {'event': 'Federal Reserve Interest Rate Decision', 'impact': 'high', 'country': 'US', 'recurring': 'Monthly'},
            {'event': 'Non-Farm Payrolls', 'impact': 'high', 'country': 'US', 'recurring': 'First Friday of month'},
            {'event': 'CPI Inflation Data', 'impact': 'high', 'country': 'US', 'recurring': 'Monthly'},
            {'event': 'GDP Growth Rate', 'impact': 'high', 'country': 'US', 'recurring': 'Quarterly'},
            {'event': 'Unemployment Rate', 'impact': 'medium', 'country': 'US', 'recurring': 'Monthly'},
            {'event': 'Retail Sales', 'impact': 'medium', 'country': 'US', 'recurring': 'Monthly'},
        ]
        return major_events, 'Reference (add Finnhub API for live data)'
    except:
        return [], None

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'Positive', polarity
        elif polarity < -0.1:
            return 'Negative', polarity
        else:
            return 'Neutral', polarity
    except:
        return 'Neutral', 0

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_economic_news():
    """Fetch economic news from RSS feeds"""
    feeds = [
        ('Reuters Business', 'https://feeds.reuters.com/reuters/businessNews'),
        ('MarketWatch', 'https://feeds.marketwatch.com/marketwatch/topstories/'),
        ('CNBC', 'https://www.cnbc.com/id/100003114/device/rss/rss.html'),
    ]
    
    all_news = []
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                all_news.append({
                    'source': source,
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', '')[:200] if entry.get('summary') else ''
                })
        except:
            continue
    
    return all_news[:15]

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_insurance_news():
    """Fetch insurance industry news"""
    feeds = [
        ('Insurance Journal', 'https://www.insurancejournal.com/feed/'),
        ('Insurance News Net', 'https://insurancenewsnet.com/feed'),
    ]
    
    all_news = []
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                all_news.append({
                    'source': source,
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', '')[:200] if entry.get('summary') else ''
                })
        except:
            continue
    
    return all_news[:10]

# ============ TECHNICAL ANALYSIS FUNCTIONS ============

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def predict_stock_movement(hist):
    """Generate stock movement prediction based on technical indicators"""
    if hist is None or len(hist) < 30:
        return None, None, None
    
    close = hist['Close']
    
    # Calculate indicators
    rsi = calculate_rsi(close).iloc[-1]
    macd, signal, _ = calculate_macd(close)
    macd_current = macd.iloc[-1]
    signal_current = signal.iloc[-1]
    
    # Moving averages
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
    current_price = close.iloc[-1]
    
    # Score calculation
    score = 0
    signals = []
    
    # RSI analysis
    if rsi < 30:
        score += 2
        signals.append("RSI oversold (bullish)")
    elif rsi > 70:
        score -= 2
        signals.append("RSI overbought (bearish)")
    elif rsi < 50:
        score += 0.5
        signals.append("RSI below 50 (neutral-bearish)")
    else:
        score -= 0.5
        signals.append("RSI above 50 (neutral-bullish)")
    
    # MACD analysis
    if macd_current > signal_current:
        score += 1.5
        signals.append("MACD above signal (bullish)")
    else:
        score -= 1.5
        signals.append("MACD below signal (bearish)")
    
    # Moving average analysis
    if current_price > sma_20:
        score += 1
        signals.append("Price above 20-day SMA (bullish)")
    else:
        score -= 1
        signals.append("Price below 20-day SMA (bearish)")
    
    if current_price > sma_50:
        score += 1
        signals.append("Price above 50-day SMA (bullish)")
    else:
        score -= 1
        signals.append("Price below 50-day SMA (bearish)")
    
    # Golden/Death cross
    if sma_20 > sma_50:
        score += 0.5
        signals.append("Short-term trend up (20 > 50 SMA)")
    else:
        score -= 0.5
        signals.append("Short-term trend down (20 < 50 SMA)")
    
    # Normalize score to -100 to 100
    max_score = 6.5
    normalized_score = (score / max_score) * 100
    
    # Determine prediction
    if normalized_score > 30:
        prediction = "BULLISH"
    elif normalized_score < -30:
        prediction = "BEARISH"
    else:
        prediction = "NEUTRAL"
    
    return prediction, normalized_score, signals

# ============ CALCULATOR FUNCTIONS (from original app) ============

def filter_liquid_options(df, min_volume, max_spread_ratio):
    spread = df["ask"] - df["bid"]
    liquid_df = df[
        (df["volume"] >= min_volume) &
        (df["bid"] > 0) &
        ((df["ask"] - df["bid"]) / df["ask"] <= max_spread_ratio)
    ]
    return liquid_df

def call_bs_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def implied_vol_call(price, S, K, T, r):
    if T <= 0:
        return np.nan
    def objective(sigma):
        return call_bs_price(S, K, T, r, sigma) - price
    try:
        implied_vol = brentq(objective, 1e-9, 5.0)
        return implied_vol
    except ValueError:
        return np.nan

def build_pdf(K_grid, iv_spline_tck, S, T, r):
    iv_vals = BSpline(*iv_spline_tck)(K_grid)
    call_prices = np.array([call_bs_price(S, K, T, r, iv) for K, iv in zip(K_grid, iv_vals)])
    first_derivative = np.gradient(call_prices, K_grid)
    second_derivative = np.gradient(first_derivative, K_grid)
    pdf_raw = np.exp(r * T) * second_derivative
    pdf_raw = np.clip(pdf_raw, 0, None)
    return K_grid, pdf_raw

def smooth_pdf(K_grid, pdf_raw):
    kde = gaussian_kde(K_grid, weights=pdf_raw)
    pdf_smooth = kde(K_grid)
    area = np.trapz(pdf_smooth, K_grid)
    if area > 0:
        pdf_smooth /= area
    return pdf_smooth

def union_of_lists(lists):
    union = set()
    for lst in lists:
        union.update(lst)
    return list(union)

def convolve_pdfs(x_lists, pdf_lists):
    x_result = np.array(x_lists[0])
    pdf_result = np.array(pdf_lists[0])

    for i in range(1, len(x_lists)):
        x_i = np.array(x_lists[i])
        pdf_i = np.array(pdf_lists[i])
        dx_result = np.mean(np.diff(x_result))
        dx_i = np.mean(np.diff(x_i))
        dx = min(dx_result, dx_i)
        x_result_uniform = np.arange(x_result.min(), x_result.max(), dx)
        x_i_uniform = np.arange(x_i.min(), x_i.max(), dx)
        f_result = interpolate.interp1d(x_result, pdf_result, bounds_error=False, fill_value=0)
        f_i = interpolate.interp1d(x_i, pdf_i, bounds_error=False, fill_value=0)
        pdf_result_uniform = f_result(x_result_uniform)
        pdf_i_uniform = f_i(x_i_uniform)
        pdf_conv = np.convolve(pdf_result_uniform, pdf_i_uniform) * dx
        x_min_new = x_result_uniform.min() + x_i_uniform.min()
        x_result = x_min_new + np.arange(len(pdf_conv)) * dx
        pdf_result = pdf_conv

    pdf_result = pdf_result / np.trapz(pdf_result, x_result)
    return x_result, pdf_result

def calculate_percentile(x_values, pdf_values, percentile):
    dx = np.diff(x_values)
    pdf_midpoints = (pdf_values[:-1] + pdf_values[1:]) / 2
    cdf = np.concatenate([[0], np.cumsum(pdf_midpoints * dx)])
    cdf = cdf / cdf[-1]
    target = percentile / 100.0
    idx = np.searchsorted(cdf, target)
    if idx == 0:
        return x_values[0]
    if idx >= len(x_values):
        return x_values[-1]
    x0, x1 = x_values[idx-1], x_values[idx]
    c0, c1 = cdf[idx-1], cdf[idx]
    if c1 == c0:
        return x0
    return x0 + (x1 - x0) * (target - c0) / (c1 - c0)

def calculate_probability_below(x_values, pdf_values, threshold):
    mask = x_values <= threshold
    if not np.any(mask):
        return 0.0
    x_below = x_values[mask]
    pdf_below = pdf_values[mask]
    return np.trapz(pdf_below, x_below)

def ticker_prediction(ticker_idx, stock_list, possible_expirations, expiration_idx, risk_free_rate, min_volume, max_spread_ratio):
    import io
    import sys
    
    investment = stock_list['Value'].iloc[ticker_idx]
    ticker_symbol = stock_list['Stocks'].iloc[ticker_idx]
    
    time.sleep(0.5)  # Rate limiting
    
    # Suppress error output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
        number_of_shares = investment / current_price

        selected_expiry = possible_expirations[expiration_idx]

        option_chain = ticker_data.option_chain(selected_expiry)
        calls_df = option_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']].copy()

        filtered_calls_df = filter_liquid_options(calls_df, min_volume, max_spread_ratio)

        T = (pd.to_datetime(selected_expiry) - pd.Timestamp.today()).days / 365.0
        S = ticker_data.history().iloc[-1]['Close']

        filtered_calls_df['iv'] = filtered_calls_df.apply(
            lambda row: implied_vol_call(row['lastPrice'], S, row['strike'], T, risk_free_rate), axis=1)

        filtered_calls_df.dropna(subset=['iv'], inplace=True)

        if len(filtered_calls_df) < 4:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return None, None

        strikes = filtered_calls_df['strike'].values
        ivs = filtered_calls_df['iv'].values

        iv_spline_tck = splrep(strikes, ivs, s=10, k=3)
        K_grid = np.linspace(strikes.min(), strikes.max(), 300)
        
        K_grid_for_pdf, pdf_raw = build_pdf(K_grid, iv_spline_tck, S, T, risk_free_rate)
        pdf_smooth_over_K_grid = smooth_pdf(K_grid_for_pdf, pdf_raw)
        investment_grid_values = np.array([K * number_of_shares for K in K_grid_for_pdf])
        pdf_smooth_over_investment_values = pdf_smooth_over_K_grid / number_of_shares

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return investment_grid_values, pdf_smooth_over_investment_values
    
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise e


# ============ MAIN APP ============

def main():
    # Get the directory where app.py is located
    app_dir = Path(__file__).parent
    logo_path = app_dir / "notrix_logo.png"
    
    # ============ SIDEBAR: API CONFIGURATION ============
    with st.sidebar:
        st.image(str(logo_path), width=120) if logo_path.exists() else st.markdown("### 📊 Notrix")
        st.markdown("---")
        
        st.header("🔑 API Configuration")
        st.caption("Get FREE API keys to enable live market data:")
        
        with st.expander("📋 How to get free API keys", expanded=False):
            st.markdown("""
            **Recommended (most generous free tier):**
            1. **Finnhub** - [finnhub.io](https://finnhub.io/)
               - 60 calls/minute FREE
               - Sign up → Get API key instantly
            
            **Alternatives:**
            2. **Twelve Data** - [twelvedata.com](https://twelvedata.com/)
               - 800 calls/day FREE
            
            3. **Alpha Vantage** - [alphavantage.co](https://www.alphavantage.co/support/#api-key)
               - 25 calls/day FREE
            
            4. **FMP** - [financialmodelingprep.com](https://site.financialmodelingprep.com/)
               - 250 calls/day FREE
            """)
        
        # API Key inputs
        finnhub_key = st.text_input(
            "Finnhub API Key",
            value="d55ei4hr01qu4ccg9omgd55ei4hr01qu4ccg9on0",
            type="password",
            help="Get free key at finnhub.io"
        )
        
        twelve_data_key = st.text_input(
            "Twelve Data API Key",
            value=TWELVE_DATA_API_KEY,
            type="password",
            help="Get free key at twelvedata.com"
        )
        
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            value=ALPHA_VANTAGE_API_KEY,
            type="password",
            help="Get free key at alphavantage.co"
        )
        
        fmp_key = st.text_input(
            "FMP API Key",
            value=FMP_API_KEY,
            type="password",
            help="Get free key at financialmodelingprep.com"
        )
        
        # Collect API keys
        api_keys = {
            'finnhub': finnhub_key,
            'twelve_data': twelve_data_key,
            'alpha_vantage': alpha_vantage_key,
            'fmp': fmp_key
        }
        
        # Show status
        active_apis = [k for k, v in api_keys.items() if v]
        if active_apis:
            st.success(f"✅ {len(active_apis)} API(s) configured: {', '.join(active_apis)}")
        else:
            st.warning("⚠️ No API keys configured. Using cached data.")
        
        # Debug mode toggle
        show_debug = st.checkbox("🔧 Show API debug info", value=False, help="See which APIs are being tried")
        
        # Test API button
        if st.button("🧪 Test API Connection"):
            with st.spinner("Testing APIs..."):
                # Test with SPY as it's most likely to work
                test_results = []
                
                if api_keys.get('finnhub'):
                    quote = get_finnhub_quote('SPY', api_keys['finnhub'])
                    if quote and quote.get('price', 0) > 0:
                        test_results.append(f"✅ Finnhub: ${quote['price']:.2f}")
                    else:
                        test_results.append("❌ Finnhub: No data")
                
                if api_keys.get('twelve_data'):
                    quote = get_twelve_data_quote('SPY', api_keys['twelve_data'])
                    if quote and quote.get('price', 0) > 0:
                        test_results.append(f"✅ Twelve Data: ${quote['price']:.2f}")
                    else:
                        test_results.append("❌ Twelve Data: No data")
                
                if api_keys.get('alpha_vantage'):
                    quote = get_alpha_vantage_quote('SPY', api_keys['alpha_vantage'])
                    if quote and quote.get('price', 0) > 0:
                        test_results.append(f"✅ Alpha Vantage: ${quote['price']:.2f}")
                    else:
                        test_results.append("❌ Alpha Vantage: No data")
                
                if api_keys.get('fmp'):
                    quote = get_fmp_quote('SPY', api_keys['fmp'])
                    if quote and quote.get('price', 0) > 0:
                        test_results.append(f"✅ FMP: ${quote['price']:.2f}")
                    else:
                        test_results.append("❌ FMP: No data")
                
                if test_results:
                    for result in test_results:
                        st.write(result)
                else:
                    st.warning("No API keys to test")
        
        st.markdown("---")
        st.caption("💡 Tip: Set environment variables to avoid re-entering keys:\n`FINNHUB_API_KEY`, `TWELVE_DATA_API_KEY`, etc.")
    
    # Header with logo
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=150)
        else:
            st.markdown("### 📊")
    
    with col_title:
        st.markdown("# Notrix Investment Fund")
        st.caption("Financial Command Center")
    
    st.divider()
    
    # Market ticker bar with loading state
    with st.container():
        with st.spinner("Loading market data..."):
            if show_debug:
                market_data, debug_messages = get_market_data(api_keys, show_debug=True)
            else:
                market_data = get_market_data(api_keys, show_debug=False)
                debug_messages = []
        
        # Show debug info if enabled
        if show_debug and debug_messages:
            with st.expander("🔧 API Debug Log", expanded=True):
                for msg in debug_messages:
                    st.text(msg)
        
        cols = st.columns(6)
        
        unavailable_count = 0
        cached_count = 0
        sources_used = set()
        
        for i, (name, data) in enumerate(market_data.items()):
            with cols[i]:
                if data.get('available', False) and data['price'] > 0:
                    change_sign = "+" if data['change'] >= 0 else ""
                    sources_used.add(data.get('source', 'Unknown'))
                    if name in ['Gold', 'Silver', 'Copper']:
                        st.metric(
                            label=f"{name}",
                            value=f"${data['price']:,.2f}",
                            delta=f"{change_sign}{data['change']:.2f}%"
                        )
                    else:
                        st.metric(
                            label=f"{name}",
                            value=f"{data['price']:,.2f}",
                            delta=f"{change_sign}{data['change']:.2f}%"
                        )
                elif data.get('cached', False) and data['price'] > 0:
                    cached_count += 1
                    if name in ['Gold', 'Silver', 'Copper']:
                        st.metric(
                            label=f"📦 {name}",
                            value=f"${data['price']:,.2f}",
                            delta="cached"
                        )
                    else:
                        st.metric(
                            label=f"📦 {name}",
                            value=f"{data['price']:,.2f}",
                            delta="cached"
                        )
                else:
                    unavailable_count += 1
                    st.metric(label=f"{name}", value="--", delta="unavailable")
        
        # Show data source info
        if sources_used:
            st.caption(f"📡 Data sources: {', '.join(sources_used)}")
        
        if cached_count > 0:
            st.caption("📦 = Cached data. Add API keys in the sidebar for live data.")
        
        # Add refresh button
        if st.button("🔄 Refresh Market Data", key="refresh_market"):
            st.cache_data.clear()
            st.rerun()
    
    st.divider()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Stock Research", 
        "🧮 Portfolio Calculator", 
        "📅 Economic Calendar",
        "📰 Economic News",
        "🏢 Insurance News"
    ])
    
    # ============ TAB 1: STOCK RESEARCH ============
    with tab1:
        st.header("Stock Research & Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_symbol = st.text_input("🔍 Enter Stock Symbol", value="AAPL", key="stock_search").upper()
        
        with col2:
            period = st.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
        
        if search_symbol:
            search_button = st.button("🔍 Search Stock", type="primary")
            
            if search_button or ('last_search' in st.session_state and st.session_state.last_search == search_symbol):
                st.session_state.last_search = search_symbol
                
                with st.spinner(f"Fetching data for {search_symbol}... (this may take a moment)"):
                    hist, info, data_source = get_stock_data_multi_source(search_symbol, period, api_keys)
                
                if hist is not None and len(hist) > 0:
                    if data_source:
                        st.caption(f"📡 Data source: {data_source}")
                    
                    # Safely get price values
                    current_price = safe_get_value(hist, 'Close', -1)
                    prev_price = safe_get_value(hist, 'Close', -2) if len(hist) > 1 else current_price
                    high_price = safe_get_column(hist, 'High')
                    low_price = safe_get_column(hist, 'Low')
                    
                    if current_price is None:
                        st.error(f"Could not parse price data for {search_symbol}")
                    else:
                        # Company info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            day_change = ((current_price - prev_price) / prev_price) * 100 if prev_price and prev_price > 0 else 0
                            st.metric("Day Change", f"{day_change:+.2f}%")
                        with col3:
                            high_val = float(high_price.max()) if high_price is not None else current_price
                            st.metric("Period High", f"${high_val:.2f}")
                        with col4:
                            low_val = float(low_price.min()) if low_price is not None else current_price
                            st.metric("Period Low", f"${low_val:.2f}")
                        
                        # Stock chart with technical indicators
                        st.subheader("📊 Price Chart & Technical Analysis")
                        
                        # Ensure we have proper Series for calculations
                        close_series = safe_get_column(hist, 'Close')
                        open_series = safe_get_column(hist, 'Open')
                        high_series = safe_get_column(hist, 'High')
                        low_series = safe_get_column(hist, 'Low')
                        volume_series = safe_get_column(hist, 'Volume')
                        
                        if close_series is not None and len(close_series) > 20:
                            # Calculate indicators
                            sma_20 = close_series.rolling(20).mean()
                            sma_50 = close_series.rolling(50).mean()
                            upper, middle, lower = calculate_bollinger_bands(close_series)
                            
                            # Create subplot figure
                            fig = make_subplots(
                                rows=3, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=('Price & Indicators', 'Volume', 'RSI')
                            )
                            
                            # Candlestick chart
                            fig.add_trace(go.Candlestick(
                                x=hist.index,
                                open=open_series,
                                high=high_series,
                                low=low_series,
                                close=close_series,
                                name='Price'
                            ), row=1, col=1)
                            
                            # Moving averages
                            fig.add_trace(go.Scatter(x=hist.index, y=sma_20, name='SMA 20', 
                                                    line=dict(color='orange', width=1)), row=1, col=1)
                            fig.add_trace(go.Scatter(x=hist.index, y=sma_50, name='SMA 50', 
                                                    line=dict(color='blue', width=1)), row=1, col=1)
                            
                            # Bollinger Bands
                            fig.add_trace(go.Scatter(x=hist.index, y=upper, name='BB Upper',
                                                    line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                            fig.add_trace(go.Scatter(x=hist.index, y=lower, name='BB Lower',
                                                    line=dict(color='gray', width=1, dash='dot'),
                                                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
                            
                            # Volume
                            if volume_series is not None and open_series is not None:
                                colors = ['red' if close_series.iloc[i] < open_series.iloc[i] else 'green' 
                                         for i in range(len(close_series))]
                                fig.add_trace(go.Bar(x=hist.index, y=volume_series, name='Volume',
                                                    marker_color=colors), row=2, col=1)
                            
                            # RSI
                            rsi = calculate_rsi(close_series)
                            fig.add_trace(go.Scatter(x=hist.index, y=rsi, name='RSI',
                                                    line=dict(color='purple', width=2)), row=3, col=1)
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                            
                            fig.update_layout(
                                template='plotly_dark',
                                height=700,
                                showlegend=True,
                                xaxis_rangeslider_visible=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction section
                            st.subheader("🔮 Stock Movement Prediction")
                            
                            prediction, score, signals = predict_stock_movement(hist)
                            
                            if prediction:
                                pred_col1, pred_col2 = st.columns([1, 2])
                                
                                with pred_col1:
                                    if prediction == "BULLISH":
                                        st.success(f"### 📈 {prediction}")
                                        st.metric("Confidence Score", f"{score:.1f}/100")
                                    elif prediction == "BEARISH":
                                        st.error(f"### 📉 {prediction}")
                                        st.metric("Confidence Score", f"{score:.1f}/100")
                                    else:
                                        st.warning(f"### ➡️ {prediction}")
                                        st.metric("Confidence Score", f"{score:.1f}/100")
                                
                                with pred_col2:
                                    st.write("**Technical Signals:**")
                                    for signal in signals:
                                        if "bullish" in signal.lower():
                                            st.markdown(f"- 🟢 {signal}")
                                        elif "bearish" in signal.lower():
                                            st.markdown(f"- 🔴 {signal}")
                                        else:
                                            st.markdown(f"- 🟡 {signal}")
                                
                                st.info("⚠️ This prediction is based on technical analysis only. Always do your own research before making investment decisions.")
                        else:
                            st.warning("Not enough historical data for technical analysis (need at least 20 data points)")
                        
                        # News and Sentiment
                        st.subheader("📰 News & Sentiment Analysis")
                        
                        with st.spinner("Fetching news..."):
                            news, news_source = get_stock_news_multi_source(search_symbol, api_keys)
                        
                        # Filter out news items with no real title
                        valid_news = [n for n in news if n.get('title') and n.get('title') != 'No title' and len(n.get('title', '')) > 5]
                        
                        if valid_news:
                            if news_source:
                                st.caption(f"📡 News source: {news_source}")
                            
                            sentiments = []
                            for article in valid_news[:5]:
                                title = article.get('title', 'No title')
                                link = article.get('link', '#')
                                publisher = article.get('publisher', 'Unknown')
                                
                                sentiment, polarity = analyze_sentiment(title)
                                sentiments.append(polarity)
                                
                                if sentiment == 'Positive':
                                    sentiment_badge = "🟢 Positive"
                                elif sentiment == 'Negative':
                                    sentiment_badge = "🔴 Negative"
                                else:
                                    sentiment_badge = "🟡 Neutral"
                                
                                with st.container():
                                    st.markdown(f"""
                                    **{title}**  
                                    *{publisher}* | {sentiment_badge}  
                                    [Read more]({link})
                                    """)
                                    st.divider()
                            
                            # Overall sentiment
                            avg_sentiment = np.mean(sentiments) if sentiments else 0
                            st.metric("Overall News Sentiment", 
                                     "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral",
                                     f"{avg_sentiment:.2f}")
                        else:
                            st.info("📭 No recent news available for this stock. Try adding a Finnhub API key for better news coverage.")
                        
                        # ============ FUNDAMENTAL ANALYSIS SECTION ============
                        st.subheader("📊 Fundamental Analysis")
                        
                        with st.spinner("Fetching fundamental data..."):
                            fundamentals = get_fundamental_data(search_symbol, api_keys)
                        
                        if fundamentals and fundamentals.get('source'):
                            st.caption(f"📡 Data source: {fundamentals['source']}")
                            
                            # Company Overview
                            if fundamentals.get('company_info'):
                                company = fundamentals['company_info']
                                if company.get('name'):
                                    st.markdown(f"**{company.get('name')}** | {company.get('sector', 'N/A')} | {company.get('industry', 'N/A')}")
                                if company.get('description'):
                                    with st.expander("Company Description"):
                                        st.write(company['description'])
                            
                            # Key Metrics in columns
                            st.markdown("#### Key Valuation Metrics")
                            val = fundamentals.get('valuation', {})
                            m1, m2, m3, m4, m5 = st.columns(5)
                            
                            with m1:
                                mc = val.get('market_cap')
                                if mc:
                                    if mc >= 1e12:
                                        mc_str = f"${mc/1e12:.2f}T"
                                    elif mc >= 1e9:
                                        mc_str = f"${mc/1e9:.2f}B"
                                    else:
                                        mc_str = f"${mc/1e6:.2f}M"
                                    st.metric("Market Cap", mc_str)
                                else:
                                    st.metric("Market Cap", "N/A")
                            
                            with m2:
                                pe = val.get('pe_ratio')
                                st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                            
                            with m3:
                                fpe = val.get('forward_pe')
                                st.metric("Forward P/E", f"{fpe:.2f}" if fpe else "N/A")
                            
                            with m4:
                                peg = val.get('peg_ratio')
                                st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
                            
                            with m5:
                                pb = val.get('price_to_book')
                                st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
                            
                            # Profitability Metrics
                            st.markdown("#### Profitability")
                            prof = fundamentals.get('profitability', {})
                            p1, p2, p3, p4, p5 = st.columns(5)
                            
                            with p1:
                                pm = prof.get('profit_margin')
                                st.metric("Profit Margin", f"{pm*100:.1f}%" if pm else "N/A")
                            
                            with p2:
                                om = prof.get('operating_margin')
                                st.metric("Operating Margin", f"{om*100:.1f}%" if om else "N/A")
                            
                            with p3:
                                roe = prof.get('return_on_equity')
                                st.metric("ROE", f"{roe*100:.1f}%" if roe else "N/A")
                            
                            with p4:
                                roa = prof.get('return_on_assets')
                                st.metric("ROA", f"{roa*100:.1f}%" if roa else "N/A")
                            
                            with p5:
                                eps = prof.get('eps')
                                st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
                            
                            # Financial Health
                            st.markdown("#### Financial Health")
                            health = fundamentals.get('financial_health', {})
                            h1, h2, h3, h4 = st.columns(4)
                            
                            with h1:
                                dte = health.get('debt_to_equity')
                                color = "normal" if dte is None else ("inverse" if dte > 100 else "normal")
                                st.metric("Debt/Equity", f"{dte:.1f}%" if dte else "N/A")
                            
                            with h2:
                                cr = health.get('current_ratio')
                                st.metric("Current Ratio", f"{cr:.2f}" if cr else "N/A")
                            
                            with h3:
                                fcf = health.get('free_cash_flow')
                                if fcf:
                                    fcf_str = f"${fcf/1e9:.2f}B" if abs(fcf) >= 1e9 else f"${fcf/1e6:.1f}M"
                                    st.metric("Free Cash Flow", fcf_str)
                                else:
                                    st.metric("Free Cash Flow", "N/A")
                            
                            with h4:
                                td = health.get('total_debt')
                                if td:
                                    td_str = f"${td/1e9:.2f}B" if td >= 1e9 else f"${td/1e6:.1f}M"
                                    st.metric("Total Debt", td_str)
                                else:
                                    st.metric("Total Debt", "N/A")
                            
                            # Growth & Dividends
                            grow_col, div_col = st.columns(2)
                            
                            with grow_col:
                                st.markdown("#### Growth")
                                growth = fundamentals.get('growth', {})
                                rg = growth.get('revenue_growth')
                                eg = growth.get('earnings_growth')
                                
                                if rg is not None:
                                    st.metric("Revenue Growth (YoY)", f"{rg*100:+.1f}%")
                                if eg is not None:
                                    st.metric("Earnings Growth (YoY)", f"{eg*100:+.1f}%")
                            
                            with div_col:
                                st.markdown("#### Dividends")
                                div = fundamentals.get('dividends', {})
                                dy = div.get('dividend_yield')
                                dr = div.get('dividend_rate')
                                pr = div.get('payout_ratio')
                                
                                if dy:
                                    # Yahoo Finance format detection:
                                    # If multiplying by 100 would give >20%, value is likely already a percentage
                                    # (Most stocks have yields 0-6%, rarely above 15%)
                                    if dy * 100 > 20:
                                        # Already in percentage-like format (e.g., 0.44 = 0.44%)
                                        st.metric("Dividend Yield", f"{dy:.2f}%")
                                    else:
                                        # Decimal format (e.g., 0.0044 = 0.44%)
                                        st.metric("Dividend Yield", f"{dy*100:.2f}%")
                                elif dr:
                                    st.metric("Annual Dividend", f"${dr:.2f}")
                                else:
                                    st.write("No dividend")
                                
                                if pr:
                                    # Payout ratio: same logic - if *100 > 200%, likely already percentage
                                    if pr * 100 > 200:
                                        st.metric("Payout Ratio", f"{pr:.1f}%")
                                    else:
                                        st.metric("Payout Ratio", f"{pr*100:.1f}%")
                            
                            # Analyst Ratings
                            st.markdown("#### Analyst Recommendations")
                            analyst = fundamentals.get('analyst', {})
                            
                            if analyst.get('target_mean') or analyst.get('recommendation'):
                                a1, a2, a3, a4 = st.columns(4)
                                
                                with a1:
                                    rec = analyst.get('recommendation', '').upper()
                                    if rec:
                                        rec_colors = {'BUY': '🟢', 'STRONG_BUY': '🟢', 'OUTPERFORM': '🟢',
                                                     'HOLD': '🟡', 'NEUTRAL': '🟡', 
                                                     'SELL': '🔴', 'UNDERPERFORM': '🔴', 'STRONG_SELL': '🔴'}
                                        icon = rec_colors.get(rec, '⚪')
                                        st.metric("Rating", f"{icon} {rec.replace('_', ' ')}")
                                
                                with a2:
                                    tm = analyst.get('target_mean')
                                    if tm:
                                        upside = ((tm - current_price) / current_price) * 100
                                        st.metric("Price Target (Avg)", f"${tm:.2f}", f"{upside:+.1f}%")
                                
                                with a3:
                                    tl = analyst.get('target_low')
                                    if tl:
                                        st.metric("Target Low", f"${tl:.2f}")
                                
                                with a4:
                                    th = analyst.get('target_high')
                                    if th:
                                        st.metric("Target High", f"${th:.2f}")
                                
                                na = analyst.get('num_analysts')
                                if na:
                                    st.caption(f"Based on {na} analyst(s)")
                            else:
                                st.info("No analyst coverage available")
                            
                            # ============ PIOTROSKI F-SCORE ============
                            st.markdown("#### 📈 Piotroski F-Score")
                            st.caption("A 9-point scoring system measuring financial strength (higher is better)")
                            
                            with st.spinner("Calculating Piotroski F-Score..."):
                                f_score, score_details = calculate_piotroski_score(search_symbol)
                            
                            if f_score is not None:
                                # Score display with color coding
                                score_col, detail_col = st.columns([1, 2])
                                
                                with score_col:
                                    if f_score >= 7:
                                        st.success(f"### F-Score: {f_score}/9")
                                        st.write("**Strong** financial position")
                                    elif f_score >= 4:
                                        st.warning(f"### F-Score: {f_score}/9")
                                        st.write("**Average** financial position")
                                    else:
                                        st.error(f"### F-Score: {f_score}/9")
                                        st.write("**Weak** financial position")
                                    
                                    # Score interpretation
                                    st.markdown("""
                                    **Interpretation:**
                                    - 8-9: Very Strong
                                    - 6-7: Strong
                                    - 4-5: Average
                                    - 2-3: Weak
                                    - 0-1: Very Weak
                                    """)
                                
                                with detail_col:
                                    st.markdown("**Score Breakdown:**")
                                    
                                    # Group by category
                                    st.markdown("*Profitability (4 pts)*")
                                    for detail in score_details[:4]:
                                        st.write(detail)
                                    
                                    st.markdown("*Leverage & Liquidity (3 pts)*")
                                    for detail in score_details[4:7]:
                                        st.write(detail)
                                    
                                    st.markdown("*Operating Efficiency (2 pts)*")
                                    for detail in score_details[7:9]:
                                        st.write(detail)
                            else:
                                st.warning("Could not calculate Piotroski F-Score")
                                for detail in score_details:
                                    st.write(detail)
                            
                            # ============ EARNINGS CALENDAR ============
                            st.markdown("#### 📅 Earnings History & Upcoming")
                            
                            with st.spinner("Fetching earnings data..."):
                                earnings_data = get_earnings_calendar(search_symbol, api_keys)
                            
                            if earnings_data.get('earnings_history'):
                                earnings_df = pd.DataFrame(earnings_data['earnings_history'])
                                
                                # Format the dataframe
                                if 'date' in earnings_df.columns:
                                    earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.strftime('%Y-%m-%d')
                                
                                display_cols = ['date', 'eps_estimate', 'eps_actual', 'surprise']
                                display_cols = [c for c in display_cols if c in earnings_df.columns]
                                
                                st.dataframe(
                                    earnings_df[display_cols].rename(columns={
                                        'date': 'Date',
                                        'eps_estimate': 'EPS Estimate',
                                        'eps_actual': 'EPS Actual',
                                        'surprise': 'Surprise %'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                if earnings_data.get('source'):
                                    st.caption(f"Source: {earnings_data['source']}")
                            else:
                                st.info("No earnings history available")
                        
                        else:
                            st.warning("Could not fetch fundamental data for this symbol")
                else:
                    st.error(f"Could not find data for symbol: {search_symbol}. Please try again in a few moments (API rate limits may apply).")
    
    # ============ TAB 2: PORTFOLIO CALCULATOR ============
    with tab2:
        st.header("📈 Daily Profitability Calculator")
        
        # Sidebar parameters (now in main area)
        with st.expander("⚙️ Model Parameters", expanded=True):
            param_col1, param_col2, param_col3, param_col4 = st.columns(4)
            
            with param_col1:
                st.subheader("Risk-Free Rate")
                rfr_mode = st.radio("Input method", ["Slider", "Manual"], key="rfr_mode_calc", horizontal=True)
                if rfr_mode == "Slider":
                    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.15, 0.04, 0.005)
                else:
                    risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.04, step=0.001, format="%.4f")
            
            with param_col2:
                st.subheader("Liquidity Filters")
                min_volume = st.number_input("Minimum Volume", min_value=1, max_value=1000, value=20, step=1)
                max_spread_ratio = st.number_input("Max Spread Ratio", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            
            with param_col3:
                st.subheader("Capital")
                free_capital = st.number_input("Free Capital ($)", value=100.0, step=100.0)
            
            with param_col4:
                st.subheader("Risk Metrics")
                var_confidence = st.selectbox("VaR Confidence Level", [90, 95, 99], index=1)
                show_percentiles = st.multiselect(
                    "Show Percentiles",
                    options=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                    default=[5, 25, 50, 75, 95]
                )
        
        show_unleveraged = st.checkbox("Show unleveraged metrics", value=True)
        
        # Portfolio Input
        st.subheader("📁 Portfolio Input")
        
        input_method = st.radio(
            "How would you like to input your portfolio?",
            ["Upload Excel File", "Manual Entry"],
            horizontal=True,
            key="input_method_calc"
        )
        
        with st.expander("ℹ️ About Leverage"):
            st.markdown("""
            **Value**: The total position size (including any leverage/margin)
            
            **Unleveraged Value**: Your actual capital at risk (without leverage)
            
            **Example**: If you have $1,000 and use 2x leverage to buy $2,000 worth of stock:
            - Value = $2,000
            - Unleveraged Value = $1,000
            
            If not using leverage, set both values equal.
            """)
        
        stock_list = None
        
        if input_method == "Upload Excel File":
            uploaded_file = st.file_uploader("📤 Upload your Stock Distribution Excel file", type=['xlsx', 'xls'])
            
            if uploaded_file is not None:
                stock_list = pd.read_excel(uploaded_file)
                st.write("**📊 Uploaded Portfolio:**")
                st.dataframe(stock_list, use_container_width=True)
                
                required_cols = ['Stocks', 'Value']
                if not all(col in stock_list.columns for col in required_cols):
                    st.error("❌ Excel file must have 'Stocks' and 'Value' columns")
                    stock_list = None
                else:
                    if 'Unleveraged Value' not in stock_list.columns:
                        stock_list['Unleveraged Value'] = stock_list['Value']
                        st.info("ℹ️ 'Unleveraged Value' column not found - assuming no leverage")
            else:
                st.info("👆 Please upload an Excel file with columns: 'Stocks', 'Value', and optionally 'Unleveraged Value'")
                
        else:  # Manual Entry
            st.write("**✏️ Enter Your Portfolio:**")
            
            if 'manual_stocks_calc' not in st.session_state:
                st.session_state.manual_stocks_calc = pd.DataFrame({
                    'Stocks': ['AAPL', 'MSFT'],
                    'Value': [2000.0, 3000.0],
                    'Unleveraged Value': [1000.0, 1500.0]
                })
            
            edited_df = st.data_editor(
                st.session_state.manual_stocks_calc,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Stocks": st.column_config.TextColumn("Ticker Symbol", max_chars=10),
                    "Value": st.column_config.NumberColumn("Position Value ($)", min_value=0, format="$%.2f"),
                    "Unleveraged Value": st.column_config.NumberColumn("Unleveraged Value ($)", min_value=0, format="$%.2f")
                }
            )
            
            st.session_state.manual_stocks_calc = edited_df
            
            if len(edited_df) > 0 and edited_df['Stocks'].notna().any() and edited_df['Value'].notna().any():
                stock_list = edited_df.dropna(subset=['Stocks', 'Value'])
                stock_list = stock_list[stock_list['Stocks'].str.strip() != '']
                stock_list = stock_list[stock_list['Value'] > 0]
                stock_list = stock_list.reset_index(drop=True)
                
                if 'Unleveraged Value' not in stock_list.columns or stock_list['Unleveraged Value'].isna().any():
                    stock_list['Unleveraged Value'] = stock_list['Unleveraged Value'].fillna(stock_list['Value'])
                
                if len(stock_list) > 0:
                    leverage_ratio = stock_list['Value'].sum() / stock_list['Unleveraged Value'].sum()
                    st.success(f"✅ {len(stock_list)} stock(s) ready | Effective leverage: {leverage_ratio:.2f}x")
                else:
                    stock_list = None
        
        # Analysis
        if stock_list is not None and len(stock_list) > 0:
            tickers = list(stock_list['Stocks'])
            
            with st.spinner("Fetching expiration dates (this may take a moment)..."):
                import io
                import sys
                
                expiration_dates = []
                for ticker in tickers:
                    try:
                        time.sleep(0.3)  # Rate limiting
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = io.StringIO()
                        sys.stderr = io.StringIO()
                        
                        t = yf.Ticker(ticker)
                        expirations = t.options
                        
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        
                        expiration_dates.append(expirations)
                    except Exception as e:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        st.warning(f"Could not fetch options for {ticker}")
                        expiration_dates.append([])
                
                possible_expirations = union_of_lists(expiration_dates)
                possible_expirations = sorted(possible_expirations)
                
                if possible_expirations:
                    first_exp_days = (pd.to_datetime(possible_expirations[0]) - pd.Timestamp.today()).days
                    if first_exp_days <= 0:
                        possible_expirations.pop(0)
            
            if not possible_expirations:
                st.error("No valid expiration dates found for any ticker")
            else:
                st.subheader("📅 Select Expiration Date")
                
                selected_expiry_str = st.selectbox("Expiration Date", options=possible_expirations, index=0)
                target_expiration = possible_expirations.index(selected_expiry_str)
                
                selected_exp_date = pd.to_datetime(possible_expirations[target_expiration])
                days_to_expiry = (selected_exp_date - pd.Timestamp.today()).days
                st.info(f"📆 Selected: **{possible_expirations[target_expiration]}** ({days_to_expiry} days to expiry)")
                
                if st.button("🚀 Calculate Distribution", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    investment_grid_list = []
                    pdf_smooth_list = []
                    failed_tickers = []
                    
                    for i, ticker in enumerate(tickers):
                        status_text.text(f"Processing {ticker}... (please wait)")
                        progress_bar.progress((i + 1) / len(tickers))
                        
                        try:
                            investment_grid, pdf_smooth = ticker_prediction(
                                i, stock_list, possible_expirations, target_expiration,
                                risk_free_rate, min_volume, max_spread_ratio
                            )
                            if investment_grid is not None:
                                investment_grid_list.append(investment_grid)
                                pdf_smooth_list.append(pdf_smooth)
                            else:
                                failed_tickers.append(ticker)
                        except Exception as e:
                            failed_tickers.append(ticker)
                            st.warning(f"Error processing {ticker}: {e}")
                    
                    status_text.text("Convolving PDFs...")
                    
                    if len(investment_grid_list) < 1:
                        st.error("Could not process any tickers successfully")
                    else:
                        if failed_tickers:
                            st.warning(f"Skipped tickers (insufficient data): {', '.join(failed_tickers)}")
                        
                        if len(investment_grid_list) == 1:
                            investment_values_final = np.array([i + free_capital for i in investment_grid_list[0]])
                            pdf_values_final = np.array(pdf_smooth_list[0])
                        else:
                            investment_values_final, pdf_values_final = convolve_pdfs(
                                investment_grid_list, pdf_smooth_list
                            )
                            investment_values_final = investment_values_final + free_capital
                        
                        status_text.text("Done!")
                        progress_bar.progress(100)
                        
                        # Calculate statistics
                        expected_value = np.trapz(investment_values_final * pdf_values_final, investment_values_final)
                        current_value_leveraged = sum(stock_list['Value']) + free_capital
                        unleveraged_capital = sum(stock_list['Unleveraged Value']) + free_capital
                        expected_gain = expected_value - current_value_leveraged
                        expected_return_leveraged = (expected_gain / current_value_leveraged) * 100
                        expected_return_unleveraged = (expected_gain / unleveraged_capital) * 100
                        leverage_ratio = current_value_leveraged / unleveraged_capital
                        
                        var_percentile = 100 - var_confidence
                        var_value = calculate_percentile(investment_values_final, pdf_values_final, var_percentile)
                        var_loss = current_value_leveraged - var_value
                        var_loss_pct_leveraged = (var_loss / current_value_leveraged) * 100
                        var_loss_pct_unleveraged = (var_loss / unleveraged_capital) * 100
                        
                        percentile_values = {}
                        for p in show_percentiles:
                            percentile_values[p] = calculate_percentile(investment_values_final, pdf_values_final, p)
                        
                        prob_loss = calculate_probability_below(investment_values_final, pdf_values_final, current_value_leveraged)
                        prob_profit = 1 - prob_loss
                        
                        # Create figure
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=investment_values_final,
                            y=pdf_values_final,
                            mode='lines',
                            name='Implied PDF',
                            line=dict(color='cyan', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(0, 255, 255, 0.1)'
                        ))
                        
                        fig.add_vline(x=current_value_leveraged, line_dash="dash", line_color="red", line_width=2,
                                     annotation_text=f"Current: ${current_value_leveraged:,.2f}")
                        fig.add_vline(x=expected_value, line_dash="dash", line_color="lime", line_width=2,
                                     annotation_text=f"Expected: ${expected_value:,.2f}")
                        fig.add_vline(x=var_value, line_dash="dot", line_color="orange", line_width=2,
                                     annotation_text=f"VaR {var_confidence}%: ${var_value:,.2f}")
                        
                        fig.update_layout(
                            title=f"Implied Probability Density for {possible_expirations[target_expiration]}",
                            xaxis_title="Portfolio Value ($)",
                            yaxis_title="Probability Density",
                            template="plotly_dark",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary Statistics
                        st.subheader("📊 Summary Statistics")
                        
                        if leverage_ratio > 1.01:
                            st.info(f"⚡ **Leverage Ratio: {leverage_ratio:.2f}x**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Position Value", f"${current_value_leveraged:,.2f}")
                        with col2:
                            st.metric("Expected Value", f"${expected_value:,.2f}")
                        with col3:
                            st.metric("Expected Return", f"{expected_return_leveraged:+.2f}%")
                        with col4:
                            st.metric(f"VaR ({var_confidence}%)", f"${var_loss:,.2f}")
                        
                        # Probabilities
                        prob_col1, prob_col2 = st.columns(2)
                        with prob_col1:
                            st.metric("Probability of Profit", f"{prob_profit*100:.1f}%")
                        with prob_col2:
                            st.metric("Probability of Loss", f"{prob_loss*100:.1f}%")
    
    # ============ TAB 3: ECONOMIC CALENDAR ============
    with tab3:
        st.header("📅 Economic & Earnings Calendar")
        
        cal_tab1, cal_tab2 = st.tabs(["🌍 Economic Events", "📊 Earnings Calendar"])
        
        # Economic Events Tab
        with cal_tab1:
            st.subheader("Upcoming Economic Events")
            
            if not api_keys.get('finnhub'):
                st.warning("⚠️ Add a Finnhub API key in the sidebar to see live economic events")
            
            with st.spinner("Fetching economic calendar..."):
                events, events_source = get_economic_calendar(api_keys)
            
            if events:
                if events_source:
                    st.caption(f"📡 Data source: {events_source}")
                
                # Filter by impact
                impact_filter = st.multiselect(
                    "Filter by Impact",
                    options=['high', 'medium', 'low'],
                    default=['high', 'medium']
                )
                
                # Filter events
                if events_source == 'Finnhub':
                    filtered_events = [e for e in events if e.get('impact', 'low').lower() in impact_filter]
                else:
                    filtered_events = events
                
                if filtered_events:
                    for event in filtered_events:
                        impact = event.get('impact', 'medium').lower()
                        if impact == 'high':
                            impact_icon = "🔴"
                        elif impact == 'medium':
                            impact_icon = "🟡"
                        else:
                            impact_icon = "🟢"
                        
                        country = event.get('country', 'US')
                        country_flag = '🇺🇸' if country == 'US' else '🌍'
                        
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{impact_icon} {event.get('event', 'Unknown Event')}**")
                                
                                date_str = event.get('date', '')
                                if date_str:
                                    st.caption(f"{country_flag} {country} | {date_str}")
                                
                                # Show actual vs estimate if available
                                actual = event.get('actual')
                                estimate = event.get('estimate')
                                previous = event.get('previous')
                                
                                if actual is not None or estimate is not None or previous is not None:
                                    metrics = []
                                    if previous is not None:
                                        metrics.append(f"Previous: {previous}")
                                    if estimate is not None:
                                        metrics.append(f"Forecast: {estimate}")
                                    if actual is not None:
                                        metrics.append(f"**Actual: {actual}**")
                                    st.write(" | ".join(metrics))
                            
                            with col2:
                                recurring = event.get('recurring')
                                if recurring:
                                    st.caption(recurring)
                            
                            st.divider()
                else:
                    st.info("No events matching your filter")
            else:
                st.info("No economic events found. Add a Finnhub API key for live data.")
            
            # Key dates reference
            with st.expander("📋 Key Recurring Economic Events"):
                st.markdown("""
                | Event | Typical Release | Impact |
                |-------|-----------------|--------|
                | **Fed Interest Rate Decision** | 8x/year (FOMC meetings) | 🔴 High |
                | **Non-Farm Payrolls** | First Friday of month | 🔴 High |
                | **CPI (Inflation)** | ~10th-15th of month | 🔴 High |
                | **GDP Growth** | Quarterly | 🔴 High |
                | **Retail Sales** | ~15th of month | 🟡 Medium |
                | **Unemployment Claims** | Weekly (Thursday) | 🟡 Medium |
                | **ISM Manufacturing PMI** | First business day of month | 🟡 Medium |
                | **Consumer Confidence** | Last Tuesday of month | 🟡 Medium |
                """)
        
        # Earnings Calendar Tab
        with cal_tab2:
            st.subheader("Earnings Calendar Lookup")
            
            # Search for specific stock earnings
            earnings_symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="earnings_lookup").upper()
            
            if st.button("🔍 Get Earnings Info", key="earnings_btn"):
                if earnings_symbol:
                    with st.spinner(f"Fetching earnings for {earnings_symbol}..."):
                        earnings_data = get_earnings_calendar(earnings_symbol, api_keys)
                    
                    if earnings_data:
                        st.markdown(f"### {earnings_symbol} Earnings")
                        
                        # Next earnings date
                        next_earnings = earnings_data.get('next_earnings')
                        if next_earnings:
                            st.info(f"📅 **Next Earnings Date:** {next_earnings}")
                        
                        # Earnings history
                        if earnings_data.get('earnings_history'):
                            st.markdown("#### Earnings History")
                            
                            history = earnings_data['earnings_history']
                            
                            # Create DataFrame for display
                            df = pd.DataFrame(history)
                            
                            if not df.empty:
                                # Format columns
                                if 'date' in df.columns:
                                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                                
                                # Calculate beat/miss
                                if 'eps_estimate' in df.columns and 'eps_actual' in df.columns:
                                    def calc_status(row):
                                        if pd.isna(row['eps_actual']) or pd.isna(row['eps_estimate']):
                                            return '⏳ Pending'
                                        elif row['eps_actual'] > row['eps_estimate']:
                                            return '✅ Beat'
                                        elif row['eps_actual'] < row['eps_estimate']:
                                            return '❌ Miss'
                                        else:
                                            return '➖ Met'
                                    df['Status'] = df.apply(calc_status, axis=1)
                                
                                # Display
                                st.dataframe(
                                    df.rename(columns={
                                        'date': 'Date',
                                        'eps_estimate': 'EPS Est.',
                                        'eps_actual': 'EPS Actual',
                                        'surprise': 'Surprise %',
                                        'quarter': 'Quarter'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Visualization
                                if 'eps_estimate' in df.columns and 'eps_actual' in df.columns:
                                    chart_df = df.dropna(subset=['eps_actual'])
                                    if not chart_df.empty and len(chart_df) > 1:
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Bar(
                                            x=chart_df['date'],
                                            y=chart_df['eps_estimate'],
                                            name='Estimate',
                                            marker_color='gray'
                                        ))
                                        
                                        fig.add_trace(go.Bar(
                                            x=chart_df['date'],
                                            y=chart_df['eps_actual'],
                                            name='Actual',
                                            marker_color=['green' if a > e else 'red' 
                                                         for a, e in zip(chart_df['eps_actual'], chart_df['eps_estimate'])]
                                        ))
                                        
                                        fig.update_layout(
                                            title=f'{earnings_symbol} EPS: Estimate vs Actual',
                                            barmode='group',
                                            template='plotly_dark',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            if earnings_data.get('source'):
                                st.caption(f"Source: {earnings_data['source']}")
                        else:
                            st.info("No earnings history available")
                    else:
                        st.error("Could not fetch earnings data")
            
            # Popular stocks quick lookup
            st.markdown("---")
            st.markdown("#### Quick Lookup - Popular Stocks")
            
            popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM']
            
            cols = st.columns(4)
            for i, stock in enumerate(popular_stocks):
                with cols[i % 4]:
                    if st.button(stock, key=f"quick_{stock}"):
                        st.session_state['earnings_lookup'] = stock
                        st.rerun()
    
    # ============ TAB 4: ECONOMIC NEWS ============
    with tab4:
        st.header("📰 Economic & Market News")
        
        if st.button("🔄 Refresh News", key="refresh_econ"):
            st.cache_data.clear()
        
        with st.spinner("Loading news..."):
            news = get_economic_news()
        
        if news:
            for article in news:
                sentiment, polarity = analyze_sentiment(article['title'])
                
                if sentiment == 'Positive':
                    sentiment_icon = "🟢"
                elif sentiment == 'Negative':
                    sentiment_icon = "🔴"
                else:
                    sentiment_icon = "🟡"
                
                with st.container():
                    st.markdown(f"""
                    ### {sentiment_icon} {article['title']}
                    **Source:** {article['source']} | **Published:** {article.get('published', 'N/A')}
                    
                    {article.get('summary', '')}
                    
                    [Read full article]({article['link']})
                    """)
                    st.divider()
        else:
            st.info("Unable to fetch economic news at this time. Please try again later.")
    
    # ============ TAB 5: INSURANCE NEWS ============
    with tab5:
        st.header("🏢 Insurance Industry News")
        
        if st.button("🔄 Refresh News", key="refresh_insurance"):
            st.cache_data.clear()
        
        with st.spinner("Loading news..."):
            news = get_insurance_news()
        
        if news:
            for article in news:
                sentiment, polarity = analyze_sentiment(article['title'])
                
                if sentiment == 'Positive':
                    sentiment_icon = "🟢"
                elif sentiment == 'Negative':
                    sentiment_icon = "🔴"
                else:
                    sentiment_icon = "🟡"
                
                with st.container():
                    st.markdown(f"""
                    ### {sentiment_icon} {article['title']}
                    **Source:** {article['source']} | **Published:** {article.get('published', 'N/A')}
                    
                    {article.get('summary', '')}
                    
                    [Read full article]({article['link']})
                    """)
                    st.divider()
        else:
            st.info("Unable to fetch insurance news at this time. Please try again later.")
    
    # Footer with logo
    st.divider()
    footer_col1, footer_col2 = st.columns([1, 5])
    with footer_col1:
        if logo_path.exists():
            st.image(str(logo_path), width=50)
    with footer_col2:
        st.caption(f"Notrix Investment Fund | Financial Command Center | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
