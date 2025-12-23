# 🔑 API Keys Setup Guide for Notrix Investment Fund

## Quick Start (5 minutes)

Since Yahoo Finance frequently blocks automated requests, we recommend getting at least ONE free API key for reliable data.

---

## 🏆 RECOMMENDED: Finnhub (Best Free Option)

**Why:** 60 API calls per minute FREE - most generous!

### Steps:
1. Go to [finnhub.io](https://finnhub.io/)
2. Click "Get free API key"
3. Sign up with email
4. Copy your API key
5. Paste it in the app's sidebar under "Finnhub API Key"
6. Click "🧪 Test API Connection" to verify it works

---

## Alternative Options

### Twelve Data
- **Free tier:** 800 calls/day
- **Website:** [twelvedata.com](https://twelvedata.com/)
- Good for: stocks, forex, crypto

### Alpha Vantage
- **Free tier:** 25 calls/day
- **Website:** [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- Good for: US stocks, technical indicators

### Financial Modeling Prep (FMP)
- **Free tier:** 250 calls/day
- **Website:** [financialmodelingprep.com](https://site.financialmodelingprep.com/)
- Good for: fundamentals, company data

---

## 📊 How Commodities Are Tracked

Since most free APIs don't provide direct commodity futures data, we use **ETF proxies**:

| Asset | Direct Symbol | ETF Proxy Used |
|-------|---------------|----------------|
| Gold | GC=F | GLD (SPDR Gold Trust) |
| Silver | SI=F | SLV (iShares Silver Trust) |
| Copper | HG=F | CPER (Copper Index Fund) |
| NASDAQ | ^IXIC | QQQ (Invesco QQQ Trust) |
| Dow Jones | ^DJI | DIA (SPDR Dow Jones ETF) |

The ETF prices closely track the underlying commodities/indices.

---

## 🔧 Debugging

If data isn't loading:
1. Check the "🔧 Show API debug info" checkbox in the sidebar
2. Click "🧪 Test API Connection" to verify your API key works
3. Look at the debug log to see which APIs are being tried

---

## 💡 Pro Tips

### Option 1: Enter keys in the sidebar
Just paste your API keys in the sidebar when you run the app.

### Option 2: Use environment variables (persistent)
Set these before running the app:

**Windows (Command Prompt):**
```cmd
set FINNHUB_API_KEY=your_key_here
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
$env:FINNHUB_API_KEY="your_key_here"
streamlit run app.py
```

**Git Bash / Mac / Linux:**
```bash
export FINNHUB_API_KEY="your_key_here"
streamlit run app.py
```

---

## ❓ FAQ

**Q: Do I need all the API keys?**
A: No! Just ONE key (preferably Finnhub) is enough for most use cases.

**Q: Is the data free?**
A: Yes, all these APIs have free tiers. You only pay if you exceed limits.

**Q: Why is Yahoo Finance not working?**
A: Yahoo Finance frequently blocks automated requests and rate-limits users. The alternative APIs are more reliable.

**Q: Can I use the app without any API keys?**
A: Yes, but you'll see cached/placeholder data instead of live prices.

**Q: Why does it show ETF prices instead of futures?**
A: Free API tiers usually don't include commodity futures. ETFs track the same prices very closely.
