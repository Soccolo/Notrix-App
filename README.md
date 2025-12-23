# 📊 Financial Command Center

A comprehensive desktop financial dashboard application built with Streamlit that combines portfolio analysis, market data, stock research, and news aggregation.

## Features

### 🎯 Live Market Ticker
- Real-time prices for Gold, Silver, Copper, SPY, NASDAQ, and Dow Jones
- Price change indicators with color coding
- Auto-refresh every 60 seconds

### 📈 Stock Research & Analysis
- **Stock Search**: Look up any stock by symbol
- **Interactive Charts**: Candlestick charts with technical indicators
  - 20 & 50-day Simple Moving Averages
  - Bollinger Bands
  - Volume analysis
  - RSI (Relative Strength Index)
- **Stock Prediction**: AI-based prediction using technical analysis
  - RSI analysis
  - MACD signals
  - Moving average crossovers
  - Trend analysis
- **News & Sentiment**: Real-time news with sentiment analysis for searched stocks

### 🧮 Portfolio Calculator
- Your original Daily Profitability Calculator with all features:
  - Black-Scholes options pricing
  - Implied probability distributions
  - PDF convolution for multiple positions
  - VaR (Value at Risk) calculations
  - Leverage analysis
  - Percentile analysis
  - Excel upload or manual portfolio entry

### 📰 Economic News
- Aggregated news from Reuters, MarketWatch, CNBC
- Sentiment analysis for each article
- Direct links to full articles

### 🏢 Insurance Industry News
- Dedicated feed for insurance sector news
- Sentiment indicators
- Industry-specific sources

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project files

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download TextBlob corpora (for sentiment analysis):
```bash
python -m textblob.download_corpora
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Creating a Desktop Shortcut

### Windows
Create a batch file `run_dashboard.bat`:
```batch
@echo off
cd /d "C:\path\to\financial_dashboard"
call venv\Scripts\activate
streamlit run app.py
```

### macOS/Linux
Create a shell script `run_dashboard.sh`:
```bash
#!/bin/bash
cd /path/to/financial_dashboard
source venv/bin/activate
streamlit run app.py
```

Make it executable: `chmod +x run_dashboard.sh`

## Usage Tips

### Stock Research
1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Select your desired time period
3. View the technical analysis chart
4. Check the prediction signals
5. Review recent news and sentiment

### Portfolio Calculator
1. Enter your stocks either via Excel upload or manual entry
2. Set your risk parameters in the Model Parameters section
3. Select an expiration date
4. Click "Calculate Distribution" to see the implied probability distribution

### Excel File Format
Your portfolio Excel file should have these columns:
- `Stocks`: Ticker symbols (e.g., AAPL, MSFT)
- `Value`: Position value including leverage
- `Unleveraged Value` (optional): Actual capital at risk

## Configuration

You can modify the following in the app:
- Risk-free rate
- Minimum volume for options
- Maximum spread ratio
- VaR confidence level
- Displayed percentiles

## Technical Details

### Data Sources
- **Stock Data**: Yahoo Finance (yfinance)
- **News**: RSS feeds from major financial news sources
- **Sentiment**: TextBlob NLP library

### Technical Indicators
- **RSI**: 14-period Relative Strength Index
- **MACD**: 12/26/9 Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period, 2 standard deviations
- **Moving Averages**: 20, 50, and 200-day SMAs

### Prediction Algorithm
The prediction score is based on:
- RSI levels (oversold/overbought)
- MACD signal line crossovers
- Price position relative to moving averages
- Short-term vs long-term trend alignment

## Disclaimer

⚠️ **This application is for informational and educational purposes only.**

- Stock predictions are based on technical analysis and should not be considered financial advice
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- Consult with a licensed financial advisor for personalized advice

## License

MIT License - Feel free to modify and use for personal purposes.
