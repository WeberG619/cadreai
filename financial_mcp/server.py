#!/usr/bin/env python3
"""
Financial MCP Server - Complete Market Intelligence Suite
For educational purposes only - not financial advice

Data Sources:
- Yahoo Finance (real-time quotes, fundamentals)
- FRED (Federal Reserve economic data)
- Finnhub (news, sentiment, insider trades)
- Alpha Vantage (technical data)

Tools Categories:
1. Market Data - Real-time quotes, market overview
2. Technical Analysis - RSI, MACD, patterns, signals
3. Fundamental Analysis - Financials, ratios, valuations
4. Portfolio Management - Track holdings, performance
5. Economic Data - Fed rates, inflation, jobs, GDP
6. News & Sentiment - Headlines, sentiment scores
7. Insider Activity - Insider buys/sells
8. Screening - Find stocks by criteria
9. Alerts & Signals - Price moves, volume spikes
10. Risk Analysis - Beta, volatility, correlation
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import asyncio
import hashlib

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Financial libraries
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# Paths
SCRIPT_DIR = Path(__file__).parent
PORTFOLIO_FILE = SCRIPT_DIR / "portfolio.json"
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.json"
ALERTS_FILE = SCRIPT_DIR / "alerts.json"
CONFIG_FILE = SCRIPT_DIR / "config.json"

# API Keys (free tiers)
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Initialize MCP server
server = Server("financial-mcp")

# Cache for API calls (reduce rate limiting)
_cache = {}
CACHE_DURATION = 300  # 5 minutes


def get_cached(key, fetch_func, duration=CACHE_DURATION):
    """Simple cache to avoid hitting rate limits."""
    now = datetime.now().timestamp()
    if key in _cache:
        data, timestamp = _cache[key]
        if now - timestamp < duration:
            return data
    data = fetch_func()
    _cache[key] = (data, now)
    return data


def load_json(file_path, default):
    """Load JSON file or return default."""
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return default


def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_portfolio():
    return load_json(PORTFOLIO_FILE, {"holdings": [], "cash": 0, "transactions": []})


def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().isoformat()
    save_json(PORTFOLIO_FILE, portfolio)


def load_watchlist():
    return load_json(WATCHLIST_FILE, {"stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "TSLA"]})


def save_watchlist(watchlist):
    save_json(WATCHLIST_FILE, watchlist)


def load_alerts():
    return load_json(ALERTS_FILE, {"price_alerts": [], "volume_alerts": []})


def save_alerts(alerts):
    save_json(ALERTS_FILE, alerts)


# ============== TECHNICAL INDICATORS ==============

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


def detect_patterns(prices):
    """Detect common chart patterns."""
    patterns = []

    # Check for golden cross (50 MA crosses above 200 MA)
    if len(prices) >= 200:
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            patterns.append({"pattern": "Golden Cross", "signal": "bullish", "strength": "strong"})
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            patterns.append({"pattern": "Death Cross", "signal": "bearish", "strength": "strong"})

    # Check for double bottom/top (simplified)
    recent = prices.tail(20)
    if len(recent) >= 20:
        min_idx = recent.idxmin()
        max_idx = recent.idxmax()
        # More pattern detection can be added here

    return patterns


def get_support_resistance(prices, window=20):
    """Calculate support and resistance levels."""
    recent = prices.tail(window)
    return {
        "support_1": round(recent.min(), 2),
        "support_2": round(recent.quantile(0.25), 2),
        "resistance_1": round(recent.max(), 2),
        "resistance_2": round(recent.quantile(0.75), 2),
        "pivot": round(recent.mean(), 2)
    }


def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "0%": round(high, 2),
        "23.6%": round(high - (diff * 0.236), 2),
        "38.2%": round(high - (diff * 0.382), 2),
        "50%": round(high - (diff * 0.5), 2),
        "61.8%": round(high - (diff * 0.618), 2),
        "78.6%": round(high - (diff * 0.786), 2),
        "100%": round(low, 2)
    }


# ============== EXTERNAL API HELPERS ==============

def get_fred_data(series_id):
    """Get economic data from FRED."""
    if not FRED_API_KEY:
        return None
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "limit": 10,
            "sort_order": "desc"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("observations", [])
    except:
        pass
    return None


def get_finnhub_news(symbol):
    """Get news from Finnhub."""
    if not FINNHUB_API_KEY:
        return None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": week_ago,
            "to": today,
            "token": FINNHUB_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()[:10]
    except:
        pass
    return None


def get_finnhub_sentiment(symbol):
    """Get social sentiment from Finnhub."""
    if not FINNHUB_API_KEY:
        return None
    try:
        url = f"https://finnhub.io/api/v1/stock/social-sentiment"
        params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_insider_trades(symbol):
    """Get insider trading data from Finnhub."""
    if not FINNHUB_API_KEY:
        return None
    try:
        url = f"https://finnhub.io/api/v1/stock/insider-transactions"
        params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json().get("data", [])
            return data[:10]
    except:
        pass
    return None


def get_fear_greed_index():
    """Get CNN Fear & Greed Index approximation using VIX."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            vix_value = hist['Close'].iloc[-1]
            # Approximate fear/greed based on VIX
            if vix_value < 12:
                return {"value": 90, "label": "Extreme Greed", "vix": round(vix_value, 2)}
            elif vix_value < 17:
                return {"value": 70, "label": "Greed", "vix": round(vix_value, 2)}
            elif vix_value < 22:
                return {"value": 50, "label": "Neutral", "vix": round(vix_value, 2)}
            elif vix_value < 30:
                return {"value": 30, "label": "Fear", "vix": round(vix_value, 2)}
            else:
                return {"value": 10, "label": "Extreme Fear", "vix": round(vix_value, 2)}
    except:
        pass
    return {"value": 50, "label": "Unknown", "vix": None}


# ============== TOOL DEFINITIONS ==============

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available financial tools."""
    return [
        # === MARKET DATA ===
        Tool(
            name="get_stock_quote",
            description="Get real-time stock price, change, volume, and key metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker (e.g., AAPL, MSFT)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_market_overview",
            description="Get major indices, sector performance, and market mood (fear/greed)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_sector_performance",
            description="Get performance of all market sectors (Technology, Healthcare, etc.)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_market_movers",
            description="Get top gainers, losers, and most active stocks",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === TECHNICAL ANALYSIS ===
        Tool(
            name="get_technical_analysis",
            description="Complete technical analysis: RSI, MACD, Bollinger Bands, Stochastic, ATR, signals",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_support_resistance_levels",
            description="Get support, resistance, pivot points, and Fibonacci levels",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="detect_chart_patterns",
            description="Detect chart patterns (golden cross, death cross, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_moving_averages",
            description="Get all moving averages (5, 10, 20, 50, 100, 200 day) and crossover signals",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === FUNDAMENTAL ANALYSIS ===
        Tool(
            name="get_company_fundamentals",
            description="Get company financials: revenue, earnings, margins, growth rates, ratios",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_valuation_metrics",
            description="Get valuation metrics: P/E, P/S, P/B, PEG, EV/EBITDA, DCF estimate",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_earnings_info",
            description="Get earnings history, surprises, and upcoming earnings date",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_dividend_analysis",
            description="Get dividend yield, history, growth rate, payout ratio, ex-dates",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === PORTFOLIO ===
        Tool(
            name="get_portfolio_summary",
            description="Get complete portfolio: holdings, value, gains/losses, allocation",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="add_to_portfolio",
            description="Add a stock purchase to track",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "shares": {"type": "number"},
                    "purchase_price": {"type": "number"},
                    "purchase_date": {"type": "string", "description": "YYYY-MM-DD"}
                },
                "required": ["symbol", "shares", "purchase_price"]
            }
        ),
        Tool(
            name="remove_from_portfolio",
            description="Remove a holding from portfolio (sell)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "shares": {"type": "number", "description": "Shares to sell (all if not specified)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_portfolio_risk_analysis",
            description="Analyze portfolio risk: beta, volatility, correlation, diversification score",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === WATCHLIST ===
        Tool(
            name="get_watchlist",
            description="Get all watched stocks with current prices and changes",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="add_to_watchlist",
            description="Add stock to watchlist",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="remove_from_watchlist",
            description="Remove stock from watchlist",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),

        # === ECONOMIC DATA ===
        Tool(
            name="get_economic_calendar",
            description="Get upcoming economic events (Fed meetings, jobs report, CPI, etc.)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_interest_rates",
            description="Get current Fed funds rate, treasury yields, and rate history",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_inflation_data",
            description="Get current CPI, inflation rate, and trend",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_economic_indicators",
            description="Get key economic indicators: GDP, unemployment, consumer confidence",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === NEWS & SENTIMENT ===
        Tool(
            name="get_stock_news",
            description="Get recent news headlines for a stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_market_news",
            description="Get general market news and headlines",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_sentiment_analysis",
            description="Get social media and news sentiment for a stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),

        # === INSIDER & INSTITUTIONAL ===
        Tool(
            name="get_insider_trades",
            description="Get recent insider buying/selling activity",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_institutional_holders",
            description="Get major institutional holders and their positions",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),

        # === SCREENING ===
        Tool(
            name="screen_stocks",
            description="Screen stocks by criteria (P/E, market cap, dividend, sector, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_market_cap_b": {"type": "number", "description": "Min market cap in billions"},
                    "max_pe": {"type": "number"},
                    "min_dividend_yield": {"type": "number"},
                    "sector": {"type": "string"},
                    "min_volume": {"type": "number", "description": "Min average volume"}
                }
            }
        ),
        Tool(
            name="find_undervalued_stocks",
            description="Find potentially undervalued stocks based on fundamentals",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="find_momentum_stocks",
            description="Find stocks with strong momentum and volume",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="find_dividend_stocks",
            description="Find high-quality dividend stocks",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === COMPARISON ===
        Tool(
            name="compare_stocks",
            description="Compare multiple stocks: performance, fundamentals, technicals",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "period": {"type": "string", "default": "3mo"}
                },
                "required": ["symbols"]
            }
        ),

        # === HISTORICAL ===
        Tool(
            name="get_historical_data",
            description="Get historical price data with statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {"type": "string", "default": "1y", "description": "1d,5d,1mo,3mo,6mo,1y,2y,5y,max"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="backtest_strategy",
            description="Backtest a simple moving average crossover strategy",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "fast_ma": {"type": "integer", "default": 20},
                    "slow_ma": {"type": "integer", "default": 50},
                    "period": {"type": "string", "default": "2y"}
                },
                "required": ["symbol"]
            }
        ),

        # === RISK ===
        Tool(
            name="calculate_position_size",
            description="Calculate position size based on risk management rules",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_size": {"type": "number"},
                    "risk_percent": {"type": "number", "description": "% of account to risk"},
                    "entry_price": {"type": "number"},
                    "stop_loss": {"type": "number"}
                },
                "required": ["account_size", "risk_percent", "entry_price", "stop_loss"]
            }
        ),
        Tool(
            name="analyze_risk_reward",
            description="Analyze risk/reward for a trade setup",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_price": {"type": "number"},
                    "stop_loss": {"type": "number"},
                    "target_price": {"type": "number"}
                },
                "required": ["entry_price", "stop_loss", "target_price"]
            }
        ),

        # === ALERTS ===
        Tool(
            name="set_price_alert",
            description="Set a price alert for a stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "target_price": {"type": "number"},
                    "direction": {"type": "string", "enum": ["above", "below"]}
                },
                "required": ["symbol", "target_price", "direction"]
            }
        ),
        Tool(
            name="check_alerts",
            description="Check if any price alerts have been triggered",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === CRYPTO ===
        Tool(
            name="get_crypto_prices",
            description="Get prices for major cryptocurrencies",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === SEARCH ===
        Tool(
            name="search_stocks",
            description="Search for stocks by name or symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),

        # === ANALYST & RATINGS (NEW - Essential for investment decisions) ===
        Tool(
            name="get_analyst_ratings",
            description="Get Wall Street analyst ratings, price targets, and recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_price_targets",
            description="Get analyst price targets: low, average, high, and number of analysts",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_upgrades_downgrades",
            description="Get recent analyst upgrades and downgrades",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === SHORT INTEREST (NEW - Important for risk assessment) ===
        Tool(
            name="get_short_interest",
            description="Get short interest data: shares short, short ratio, days to cover",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === OPTIONS DATA (NEW - Understanding smart money) ===
        Tool(
            name="get_options_chain",
            description="Get options chain data: calls, puts, open interest, implied volatility",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "expiration": {"type": "string", "description": "Expiration date YYYY-MM-DD (optional, uses nearest)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_options_summary",
            description="Get options summary: put/call ratio, max pain, unusual activity",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === MARKET SENTIMENT (NEW - Timing decisions) ===
        Tool(
            name="get_fear_greed_index",
            description="Get market Fear & Greed index based on VIX and market indicators",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_market_breadth",
            description="Get market breadth: advance/decline ratio, new highs/lows",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === CORRELATION & DIVERSIFICATION (NEW - Portfolio construction) ===
        Tool(
            name="get_correlation_matrix",
            description="Get correlation matrix between stocks for diversification analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "List of tickers"}
                },
                "required": ["symbols"]
            }
        ),

        # === SEC FILINGS (NEW - Due diligence) ===
        Tool(
            name="get_sec_filings",
            description="Get recent SEC filings: 10-K, 10-Q, 8-K, insider Form 4",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === PAPER TRADING (NEW - Practice before real money) ===
        Tool(
            name="paper_trade",
            description="Execute a simulated paper trade to practice without real money",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["buy", "sell"]},
                    "shares": {"type": "number"},
                    "order_type": {"type": "string", "enum": ["market", "limit"], "default": "market"},
                    "limit_price": {"type": "number", "description": "Required for limit orders"}
                },
                "required": ["symbol", "action", "shares"]
            }
        ),
        Tool(
            name="get_paper_portfolio",
            description="Get paper trading portfolio status and P&L",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="reset_paper_portfolio",
            description="Reset paper trading account to starting cash",
            inputSchema={
                "type": "object",
                "properties": {
                    "starting_cash": {"type": "number", "default": 100000}
                }
            }
        ),

        # === EARNINGS ESTIMATES (NEW - Fundamental timing) ===
        Tool(
            name="get_earnings_estimates",
            description="Get earnings estimates: EPS estimates, revenue estimates, surprise history",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_upcoming_earnings",
            description="Get stocks with earnings coming up in next 7 days",
            inputSchema={"type": "object", "properties": {}}
        ),

        # === COMPANY EVENTS (NEW - Event-driven decisions) ===
        Tool(
            name="get_company_events",
            description="Get upcoming company events: earnings, ex-dividend dates, splits",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === ETF ANALYSIS (NEW - Sector exposure) ===
        Tool(
            name="get_etf_holdings",
            description="Get top holdings and sector breakdown for an ETF",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "ETF ticker (SPY, QQQ, etc.)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="find_etfs_holding_stock",
            description="Find ETFs that hold a specific stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),

        # === INVESTMENT CHECKLIST (NEW - Decision framework) ===
        Tool(
            name="get_investment_checklist",
            description="Get comprehensive investment checklist for a stock before buying",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        )
    ]


# ============== TOOL IMPLEMENTATIONS ==============

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute financial tools."""

    try:
        # === MARKET DATA ===
        if name == "get_stock_quote":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")

            if hist.empty:
                return [TextContent(type="text", text=f"Could not find: {symbol}")]

            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev_close
            change_pct = (change / prev_close) * 100

            # Volume analysis
            avg_vol = hist['Volume'].mean()
            today_vol = hist['Volume'].iloc[-1]
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1

            result = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "price": round(current, 2),
                "change": round(change, 2),
                "change_percent": round(change_pct, 2),
                "volume": int(today_vol),
                "avg_volume": int(avg_vol),
                "volume_ratio": round(vol_ratio, 2),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_ma": info.get("fiftyDayAverage"),
                "200_day_ma": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry")
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_market_overview":
            indices = {
                "^GSPC": "S&P 500",
                "^IXIC": "NASDAQ",
                "^DJI": "Dow Jones",
                "^RUT": "Russell 2000",
                "^VIX": "VIX"
            }

            results = []
            for symbol, label in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change_pct = ((current - prev) / prev) * 100
                        results.append({
                            "index": label,
                            "symbol": symbol,
                            "price": round(current, 2),
                            "change_percent": round(change_pct, 2)
                        })
                except:
                    pass

            fear_greed = get_fear_greed_index()

            return [TextContent(type="text", text=json.dumps({
                "indices": results,
                "fear_greed": fear_greed,
                "timestamp": datetime.now().isoformat()
            }, indent=2))]

        elif name == "get_sector_performance":
            sectors = {
                "XLK": "Technology",
                "XLV": "Healthcare",
                "XLF": "Financials",
                "XLE": "Energy",
                "XLY": "Consumer Discretionary",
                "XLP": "Consumer Staples",
                "XLI": "Industrials",
                "XLB": "Materials",
                "XLU": "Utilities",
                "XLRE": "Real Estate",
                "XLC": "Communication Services"
            }

            results = []
            for symbol, sector in sectors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        day_ago = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        week_ago = hist['Close'].iloc[0]

                        results.append({
                            "sector": sector,
                            "etf": symbol,
                            "price": round(current, 2),
                            "day_change": round(((current - day_ago) / day_ago) * 100, 2),
                            "week_change": round(((current - week_ago) / week_ago) * 100, 2)
                        })
                except:
                    pass

            results.sort(key=lambda x: x["day_change"], reverse=True)
            return [TextContent(type="text", text=json.dumps({"sectors": results}, indent=2))]

        elif name == "get_market_movers":
            # Use some popular stocks to find movers
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX",
                      "JPM", "BAC", "WFC", "XOM", "CVX", "JNJ", "PFE", "UNH", "WMT", "HD", "DIS"]

            movers = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty and len(hist) > 1:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change_pct = ((current - prev) / prev) * 100
                        volume = hist['Volume'].iloc[-1]

                        movers.append({
                            "symbol": symbol,
                            "price": round(current, 2),
                            "change_percent": round(change_pct, 2),
                            "volume": int(volume)
                        })
                except:
                    pass

            gainers = sorted(movers, key=lambda x: x["change_percent"], reverse=True)[:5]
            losers = sorted(movers, key=lambda x: x["change_percent"])[:5]
            most_active = sorted(movers, key=lambda x: x["volume"], reverse=True)[:5]

            return [TextContent(type="text", text=json.dumps({
                "top_gainers": gainers,
                "top_losers": losers,
                "most_active": most_active
            }, indent=2))]

        # === TECHNICAL ANALYSIS ===
        elif name == "get_technical_analysis":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")

            if hist.empty:
                return [TextContent(type="text", text=f"Could not find: {symbol}")]

            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            current = close.iloc[-1]

            # RSI
            rsi = calculate_rsi(close)
            rsi_value = rsi.iloc[-1]
            rsi_signal = "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral"

            # MACD
            macd, signal, histogram = calculate_macd(close)
            macd_signal = "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
            bb_position = "near_upper" if current > bb_upper.iloc[-1] else "near_lower" if current < bb_lower.iloc[-1] else "middle"

            # Stochastic
            k, d = calculate_stochastic(high, low, close)
            stoch_signal = "oversold" if k.iloc[-1] < 20 else "overbought" if k.iloc[-1] > 80 else "neutral"

            # ATR (volatility)
            atr = calculate_atr(high, low, close)
            atr_pct = (atr.iloc[-1] / current) * 100

            # Moving averages
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # Trend
            trend = "bullish" if current > ma20 > ma50 else "bearish" if current < ma20 < ma50 else "mixed"

            result = {
                "symbol": symbol,
                "price": round(current, 2),
                "trend": trend,
                "indicators": {
                    "rsi": {"value": round(rsi_value, 2), "signal": rsi_signal},
                    "macd": {
                        "macd": round(macd.iloc[-1], 4),
                        "signal": round(signal.iloc[-1], 4),
                        "histogram": round(histogram.iloc[-1], 4),
                        "trend": macd_signal
                    },
                    "bollinger": {
                        "upper": round(bb_upper.iloc[-1], 2),
                        "middle": round(bb_middle.iloc[-1], 2),
                        "lower": round(bb_lower.iloc[-1], 2),
                        "position": bb_position
                    },
                    "stochastic": {
                        "k": round(k.iloc[-1], 2),
                        "d": round(d.iloc[-1], 2),
                        "signal": stoch_signal
                    },
                    "atr": {
                        "value": round(atr.iloc[-1], 2),
                        "percent": round(atr_pct, 2)
                    }
                },
                "moving_averages": {
                    "ma20": round(ma20, 2),
                    "ma50": round(ma50, 2),
                    "ma200": round(ma200, 2) if ma200 else None,
                    "price_vs_ma20": f"{round((current/ma20 - 1) * 100, 2)}%",
                    "price_vs_ma50": f"{round((current/ma50 - 1) * 100, 2)}%"
                },
                "summary": f"Trend is {trend}. RSI is {rsi_signal} at {round(rsi_value, 1)}. MACD is {macd_signal}. Stochastic is {stoch_signal}."
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_support_resistance_levels":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")

            if hist.empty:
                return [TextContent(type="text", text=f"Could not find: {symbol}")]

            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            current = close.iloc[-1]

            # Support/Resistance
            sr = get_support_resistance(close)

            # Fibonacci levels
            period_high = high.max()
            period_low = low.min()
            fib = calculate_fibonacci_levels(period_high, period_low)

            result = {
                "symbol": symbol,
                "current_price": round(current, 2),
                "support_resistance": sr,
                "fibonacci_levels": fib,
                "period_high": round(period_high, 2),
                "period_low": round(period_low, 2)
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_moving_averages":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                return [TextContent(type="text", text=f"Could not find: {symbol}")]

            close = hist['Close']
            current = close.iloc[-1]

            mas = {}
            for period in [5, 10, 20, 50, 100, 200]:
                if len(close) >= period:
                    ma = close.rolling(period).mean().iloc[-1]
                    mas[f"ma{period}"] = {
                        "value": round(ma, 2),
                        "vs_price": f"{round((current/ma - 1) * 100, 2)}%",
                        "price_above": current > ma
                    }

            # Crossover signals
            signals = []
            if len(close) >= 50:
                ma20 = close.rolling(20).mean()
                ma50 = close.rolling(50).mean()
                if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-2] <= ma50.iloc[-2]:
                    signals.append("20/50 Golden Cross (bullish)")
                elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-2] >= ma50.iloc[-2]:
                    signals.append("20/50 Death Cross (bearish)")

            if len(close) >= 200:
                ma50 = close.rolling(50).mean()
                ma200 = close.rolling(200).mean()
                if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
                    signals.append("50/200 Golden Cross (bullish)")
                elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
                    signals.append("50/200 Death Cross (bearish)")

            result = {
                "symbol": symbol,
                "price": round(current, 2),
                "moving_averages": mas,
                "signals": signals if signals else ["No crossover signals detected"]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "detect_chart_patterns":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                return [TextContent(type="text", text=f"Could not find: {symbol}")]

            patterns = detect_patterns(hist['Close'])

            return [TextContent(type="text", text=json.dumps({
                "symbol": symbol,
                "patterns_detected": patterns if patterns else [{"pattern": "No clear patterns", "signal": "neutral"}]
            }, indent=2))]

        # === FUNDAMENTAL ANALYSIS ===
        elif name == "get_company_fundamentals":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            result = {
                "symbol": symbol,
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "employees": info.get("fullTimeEmployees"),
                "financials": {
                    "revenue": info.get("totalRevenue"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "gross_profit": info.get("grossProfits"),
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "profit_margin": info.get("profitMargins"),
                    "ebitda": info.get("ebitda"),
                    "net_income": info.get("netIncomeToCommon"),
                    "eps": info.get("trailingEps"),
                    "eps_growth": info.get("earningsGrowth")
                },
                "balance_sheet": {
                    "total_cash": info.get("totalCash"),
                    "total_debt": info.get("totalDebt"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "book_value": info.get("bookValue")
                },
                "cash_flow": {
                    "operating_cash_flow": info.get("operatingCashflow"),
                    "free_cash_flow": info.get("freeCashflow")
                }
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_valuation_metrics":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            result = {
                "symbol": symbol,
                "name": info.get("longName"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "valuation": {
                    "pe_trailing": info.get("trailingPE"),
                    "pe_forward": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    "price_to_book": info.get("priceToBook"),
                    "ev_to_revenue": info.get("enterpriseToRevenue"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda")
                },
                "comparison": {
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "50_day_avg": info.get("fiftyDayAverage"),
                    "200_day_avg": info.get("twoHundredDayAverage")
                }
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_earnings_info":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            # Get earnings history
            try:
                earnings_hist = stock.earnings_history
            except:
                earnings_hist = None

            # Get calendar
            try:
                calendar = stock.calendar
            except:
                calendar = None

            result = {
                "symbol": symbol,
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "earnings_growth": info.get("earningsGrowth"),
                "next_earnings_date": str(calendar.get("Earnings Date", [None])[0]) if calendar and isinstance(calendar, dict) else None
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_dividend_analysis":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info
            dividends = stock.dividends

            div_yield = info.get("dividendYield", 0) or 0
            div_rate = info.get("dividendRate", 0) or 0
            payout = info.get("payoutRatio", 0) or 0

            # Calculate dividend growth
            div_growth = None
            if len(dividends) >= 8:
                recent_year = dividends.tail(4).sum()
                prev_year = dividends.iloc[-8:-4].sum()
                if prev_year > 0:
                    div_growth = ((recent_year - prev_year) / prev_year) * 100

            result = {
                "symbol": symbol,
                "dividend_yield": round(div_yield * 100, 2),
                "annual_dividend": div_rate,
                "payout_ratio": round(payout * 100, 2),
                "ex_dividend_date": info.get("exDividendDate"),
                "5_year_avg_yield": info.get("fiveYearAvgDividendYield"),
                "dividend_growth_rate": round(div_growth, 2) if div_growth else None,
                "recent_dividends": [{"date": str(d.date()), "amount": round(v, 4)} for d, v in list(dividends.tail(4).items())] if len(dividends) > 0 else []
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === PORTFOLIO ===
        elif name == "get_portfolio_summary":
            portfolio = load_portfolio()

            if not portfolio["holdings"]:
                return [TextContent(type="text", text="Portfolio is empty. Use add_to_portfolio to track holdings.")]

            total_value = 0
            total_cost = 0
            holdings_detail = []
            sector_allocation = {}

            for holding in portfolio["holdings"]:
                try:
                    stock = yf.Ticker(holding["symbol"])
                    info = stock.info
                    hist = stock.history(period="1d")

                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        value = current * holding["shares"]
                        cost = holding["purchase_price"] * holding["shares"]
                        gain = value - cost
                        gain_pct = (gain / cost) * 100

                        total_value += value
                        total_cost += cost

                        sector = info.get("sector", "Unknown")
                        sector_allocation[sector] = sector_allocation.get(sector, 0) + value

                        holdings_detail.append({
                            "symbol": holding["symbol"],
                            "shares": holding["shares"],
                            "cost_basis": round(holding["purchase_price"], 2),
                            "current_price": round(current, 2),
                            "value": round(value, 2),
                            "gain_loss": round(gain, 2),
                            "gain_loss_pct": round(gain_pct, 2),
                            "sector": sector
                        })
                except:
                    pass

            # Calculate sector percentages
            sector_pct = {k: round(v/total_value*100, 1) for k, v in sector_allocation.items()} if total_value > 0 else {}

            total_gain = total_value - total_cost
            total_gain_pct = (total_gain / total_cost) * 100 if total_cost > 0 else 0

            result = {
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_gain_loss": round(total_gain, 2),
                "total_gain_loss_pct": round(total_gain_pct, 2),
                "cash": portfolio.get("cash", 0),
                "holdings": holdings_detail,
                "sector_allocation": sector_pct
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "add_to_portfolio":
            portfolio = load_portfolio()

            holding = {
                "symbol": arguments["symbol"].upper(),
                "shares": arguments["shares"],
                "purchase_price": arguments["purchase_price"],
                "purchase_date": arguments.get("purchase_date", datetime.now().strftime("%Y-%m-%d"))
            }

            portfolio["holdings"].append(holding)
            portfolio["transactions"].append({
                "type": "buy",
                "symbol": holding["symbol"],
                "shares": holding["shares"],
                "price": holding["purchase_price"],
                "date": holding["purchase_date"]
            })

            save_portfolio(portfolio)

            return [TextContent(type="text", text=f"Added {holding['shares']} shares of {holding['symbol']} at ${holding['purchase_price']}")]

        elif name == "remove_from_portfolio":
            portfolio = load_portfolio()
            symbol = arguments["symbol"].upper()
            shares_to_sell = arguments.get("shares")

            for i, holding in enumerate(portfolio["holdings"]):
                if holding["symbol"] == symbol:
                    if shares_to_sell is None or shares_to_sell >= holding["shares"]:
                        portfolio["holdings"].pop(i)
                        return [TextContent(type="text", text=f"Removed all {holding['shares']} shares of {symbol}")]
                    else:
                        portfolio["holdings"][i]["shares"] -= shares_to_sell
                        save_portfolio(portfolio)
                        return [TextContent(type="text", text=f"Sold {shares_to_sell} shares of {symbol}")]

            return [TextContent(type="text", text=f"{symbol} not found in portfolio")]

        elif name == "get_portfolio_risk_analysis":
            portfolio = load_portfolio()

            if not portfolio["holdings"]:
                return [TextContent(type="text", text="Portfolio is empty")]

            symbols = [h["symbol"] for h in portfolio["holdings"]]
            weights = []
            betas = []
            returns_data = []

            total_value = 0
            for holding in portfolio["holdings"]:
                try:
                    stock = yf.Ticker(holding["symbol"])
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        value = hist['Close'].iloc[-1] * holding["shares"]
                        total_value += value
                except:
                    pass

            for holding in portfolio["holdings"]:
                try:
                    stock = yf.Ticker(holding["symbol"])
                    info = stock.info
                    hist = stock.history(period="1y")

                    if not hist.empty:
                        value = hist['Close'].iloc[-1] * holding["shares"]
                        weight = value / total_value if total_value > 0 else 0
                        weights.append(weight)
                        betas.append(info.get("beta", 1))

                        returns = hist['Close'].pct_change().dropna()
                        returns_data.append(returns)
                except:
                    pass

            # Portfolio beta
            portfolio_beta = sum(w * b for w, b in zip(weights, betas)) if weights else 1

            # Portfolio volatility
            if returns_data:
                combined = pd.concat(returns_data, axis=1)
                portfolio_returns = combined.mean(axis=1)
                volatility = portfolio_returns.std() * np.sqrt(252) * 100
            else:
                volatility = 0

            # Diversification score (number of sectors)
            sectors = set()
            for holding in portfolio["holdings"]:
                try:
                    info = yf.Ticker(holding["symbol"]).info
                    sectors.add(info.get("sector", "Unknown"))
                except:
                    pass

            div_score = min(len(sectors) / 5 * 100, 100)  # Max score at 5+ sectors

            result = {
                "portfolio_beta": round(portfolio_beta, 2),
                "annualized_volatility": round(volatility, 2),
                "diversification_score": round(div_score, 1),
                "sector_count": len(sectors),
                "risk_level": "high" if portfolio_beta > 1.2 or volatility > 25 else "medium" if portfolio_beta > 0.8 else "low"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === WATCHLIST ===
        elif name == "get_watchlist":
            watchlist = load_watchlist()
            results = []

            for symbol in watchlist["stocks"]:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="2d")
                    info = stock.info

                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change_pct = ((current - prev) / prev) * 100

                        results.append({
                            "symbol": symbol,
                            "name": info.get("shortName", symbol),
                            "price": round(current, 2),
                            "change_percent": round(change_pct, 2),
                            "volume": int(hist['Volume'].iloc[-1])
                        })
                except:
                    pass

            results.sort(key=lambda x: x["change_percent"], reverse=True)
            return [TextContent(type="text", text=json.dumps({"watchlist": results}, indent=2))]

        elif name == "add_to_watchlist":
            watchlist = load_watchlist()
            symbol = arguments["symbol"].upper()

            if symbol not in watchlist["stocks"]:
                watchlist["stocks"].append(symbol)
                save_watchlist(watchlist)
                return [TextContent(type="text", text=f"Added {symbol} to watchlist")]
            return [TextContent(type="text", text=f"{symbol} already in watchlist")]

        elif name == "remove_from_watchlist":
            watchlist = load_watchlist()
            symbol = arguments["symbol"].upper()

            if symbol in watchlist["stocks"]:
                watchlist["stocks"].remove(symbol)
                save_watchlist(watchlist)
                return [TextContent(type="text", text=f"Removed {symbol} from watchlist")]
            return [TextContent(type="text", text=f"{symbol} not in watchlist")]

        # === ECONOMIC DATA ===
        elif name == "get_interest_rates":
            rates = {}

            # Treasury yields from Yahoo Finance
            treasuries = {
                "^IRX": "3-Month T-Bill",
                "^FVX": "5-Year Treasury",
                "^TNX": "10-Year Treasury",
                "^TYX": "30-Year Treasury"
            }

            for symbol, name in treasuries.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        rates[name] = round(hist['Close'].iloc[-1], 2)
                except:
                    pass

            # Fed funds rate approximation
            try:
                fed_fund_etf = yf.Ticker("USFR")
                info = fed_fund_etf.info
                rates["Fed Funds Rate (approx)"] = info.get("yield", 0)
            except:
                pass

            return [TextContent(type="text", text=json.dumps({
                "interest_rates": rates,
                "yield_curve": "normal" if rates.get("10-Year Treasury", 0) > rates.get("3-Month T-Bill", 0) else "inverted",
                "timestamp": datetime.now().isoformat()
            }, indent=2))]

        elif name == "get_economic_calendar":
            # Simplified economic calendar based on known patterns
            now = datetime.now()

            events = [
                {"event": "FOMC Meeting", "typical_schedule": "Every 6 weeks", "impact": "high"},
                {"event": "Non-Farm Payrolls", "typical_schedule": "First Friday of month", "impact": "high"},
                {"event": "CPI Report", "typical_schedule": "Mid-month", "impact": "high"},
                {"event": "GDP Report", "typical_schedule": "End of month/quarter", "impact": "high"},
                {"event": "Retail Sales", "typical_schedule": "Mid-month", "impact": "medium"},
                {"event": "Consumer Confidence", "typical_schedule": "End of month", "impact": "medium"}
            ]

            return [TextContent(type="text", text=json.dumps({
                "note": "Economic calendar - check financial news for exact dates",
                "key_events": events
            }, indent=2))]

        elif name == "get_inflation_data":
            # Get TIP ETF as inflation proxy
            try:
                tip = yf.Ticker("TIP")
                hist = tip.history(period="1y")

                # Approximate inflation from breakeven rates
                tips_10y = yf.Ticker("^TNX")
                tips_hist = tips_10y.history(period="1d")

                result = {
                    "note": "Inflation proxies - check BLS for official CPI",
                    "10y_treasury_yield": round(tips_hist['Close'].iloc[-1], 2) if not tips_hist.empty else None,
                    "tip_etf_ytd_return": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2) if len(hist) > 0 else None,
                    "interpretation": "Treasury yields reflect market inflation expectations"
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except:
                return [TextContent(type="text", text="Could not fetch inflation data")]

        elif name == "get_economic_indicators":
            indicators = {}

            # Use ETFs as proxies
            proxies = {
                "SPY": "S&P 500 (economy health)",
                "XLI": "Industrials (manufacturing)",
                "XLY": "Consumer Discretionary (spending)",
                "XLF": "Financials (credit conditions)"
            }

            for symbol, desc in proxies.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        start = hist['Close'].iloc[0]
                        end = hist['Close'].iloc[-1]
                        change = ((end - start) / start) * 100
                        indicators[desc] = {
                            "3mo_change": round(change, 2),
                            "trend": "improving" if change > 0 else "declining"
                        }
                except:
                    pass

            return [TextContent(type="text", text=json.dumps({
                "economic_proxies": indicators,
                "note": "These ETFs serve as proxies for economic conditions"
            }, indent=2))]

        # === NEWS & SENTIMENT ===
        elif name == "get_stock_news":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            news = stock.news

            if not news:
                return [TextContent(type="text", text=f"No news found for {symbol}")]

            news_items = []
            for item in news[:10]:
                news_items.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else None
                })

            return [TextContent(type="text", text=json.dumps({"symbol": symbol, "news": news_items}, indent=2))]

        elif name == "get_market_news":
            # Get news from major indices
            spy = yf.Ticker("SPY")
            news = spy.news[:10] if spy.news else []

            news_items = []
            for item in news:
                news_items.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link")
                })

            return [TextContent(type="text", text=json.dumps({"market_news": news_items}, indent=2))]

        elif name == "get_sentiment_analysis":
            symbol = arguments["symbol"].upper()

            # Use yfinance recommendations as sentiment proxy
            stock = yf.Ticker(symbol)

            try:
                recs = stock.recommendations
                if recs is not None and len(recs) > 0:
                    recent = recs.tail(10)
                    buy_count = len(recent[recent['To Grade'].str.contains('Buy|Outperform|Overweight', case=False, na=False)])
                    sell_count = len(recent[recent['To Grade'].str.contains('Sell|Underperform|Underweight', case=False, na=False)])
                    hold_count = len(recent) - buy_count - sell_count

                    sentiment = "bullish" if buy_count > sell_count + hold_count else "bearish" if sell_count > buy_count else "neutral"

                    result = {
                        "symbol": symbol,
                        "analyst_sentiment": sentiment,
                        "recent_ratings": {
                            "buy": buy_count,
                            "hold": hold_count,
                            "sell": sell_count
                        }
                    }
                else:
                    result = {"symbol": symbol, "sentiment": "no data available"}
            except:
                result = {"symbol": symbol, "sentiment": "no data available"}

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === INSIDER & INSTITUTIONAL ===
        elif name == "get_insider_trades":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            try:
                insiders = stock.insider_transactions
                if insiders is not None and len(insiders) > 0:
                    trades = []
                    for _, row in insiders.head(10).iterrows():
                        trades.append({
                            "insider": row.get("Insider Trading"),
                            "position": row.get("Position"),
                            "transaction": row.get("Transaction"),
                            "shares": row.get("Shares"),
                            "value": row.get("Value")
                        })

                    return [TextContent(type="text", text=json.dumps({
                        "symbol": symbol,
                        "recent_insider_trades": trades
                    }, indent=2))]
            except:
                pass

            return [TextContent(type="text", text=f"No insider trade data for {symbol}")]

        elif name == "get_institutional_holders":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            try:
                holders = stock.institutional_holders
                if holders is not None and len(holders) > 0:
                    inst_list = []
                    for _, row in holders.head(10).iterrows():
                        inst_list.append({
                            "holder": row.get("Holder"),
                            "shares": int(row.get("Shares", 0)),
                            "value": int(row.get("Value", 0)),
                            "percent_out": round(row.get("% Out", 0) * 100, 2) if row.get("% Out") else None
                        })

                    return [TextContent(type="text", text=json.dumps({
                        "symbol": symbol,
                        "top_institutional_holders": inst_list
                    }, indent=2))]
            except:
                pass

            return [TextContent(type="text", text=f"No institutional holder data for {symbol}")]

        # === SCREENING ===
        elif name == "screen_stocks":
            # Expanded stock universe
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
                "UNH", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PFE",
                "KO", "PEP", "COST", "TMO", "AVGO", "MCD", "DIS", "CSCO", "ACN", "ABT",
                "NKE", "INTC", "AMD", "CRM", "NFLX", "ADBE", "PYPL", "QCOM", "TXN", "ORCL"
            ]

            results = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info

                    mcap = info.get("marketCap", 0)
                    pe = info.get("trailingPE")
                    div_yield = (info.get("dividendYield") or 0) * 100
                    volume = info.get("averageVolume", 0)
                    sector = info.get("sector")

                    # Apply filters
                    if arguments.get("min_market_cap_b") and mcap / 1e9 < arguments["min_market_cap_b"]:
                        continue
                    if arguments.get("max_pe") and pe and pe > arguments["max_pe"]:
                        continue
                    if arguments.get("min_dividend_yield") and div_yield < arguments["min_dividend_yield"]:
                        continue
                    if arguments.get("sector") and sector != arguments["sector"]:
                        continue
                    if arguments.get("min_volume") and volume < arguments["min_volume"]:
                        continue

                    results.append({
                        "symbol": symbol,
                        "name": info.get("shortName"),
                        "market_cap_b": round(mcap / 1e9, 2),
                        "pe_ratio": round(pe, 2) if pe else None,
                        "dividend_yield": round(div_yield, 2),
                        "sector": sector
                    })
                except:
                    pass

            return [TextContent(type="text", text=json.dumps({"matches": results[:20]}, indent=2))]

        elif name == "find_undervalued_stocks":
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "WFC", "JNJ", "PFE",
                      "VZ", "T", "INTC", "IBM", "GM", "F", "CVS", "WBA", "KR", "TGT"]

            candidates = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info

                    pe = info.get("trailingPE")
                    pb = info.get("priceToBook")
                    peg = info.get("pegRatio")

                    # Simple value criteria
                    if pe and pe < 15 and pb and pb < 2:
                        candidates.append({
                            "symbol": symbol,
                            "name": info.get("shortName"),
                            "pe_ratio": round(pe, 2),
                            "price_to_book": round(pb, 2),
                            "peg_ratio": round(peg, 2) if peg else None,
                            "reason": "Low P/E and P/B ratios"
                        })
                except:
                    pass

            candidates.sort(key=lambda x: x["pe_ratio"])
            return [TextContent(type="text", text=json.dumps({
                "potentially_undervalued": candidates[:10],
                "note": "These stocks have low valuation metrics - requires further research"
            }, indent=2))]

        elif name == "find_momentum_stocks":
            symbols = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "GOOGL", "AMZN", "NFLX", "CRM",
                      "AVGO", "ADBE", "NOW", "SNOW", "PANW", "CRWD", "DDOG", "ZS", "NET", "PLTR"]

            candidates = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="3mo")

                    if not hist.empty:
                        start = hist['Close'].iloc[0]
                        end = hist['Close'].iloc[-1]
                        change_3mo = ((end - start) / start) * 100

                        # Volume surge
                        avg_vol = hist['Volume'].mean()
                        recent_vol = hist['Volume'].tail(5).mean()
                        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

                        if change_3mo > 10 and vol_ratio > 1:
                            candidates.append({
                                "symbol": symbol,
                                "3mo_return": round(change_3mo, 2),
                                "volume_ratio": round(vol_ratio, 2),
                                "price": round(end, 2)
                            })
                except:
                    pass

            candidates.sort(key=lambda x: x["3mo_return"], reverse=True)
            return [TextContent(type="text", text=json.dumps({
                "momentum_stocks": candidates[:10],
                "criteria": "3-month return >10% with above-average volume"
            }, indent=2))]

        elif name == "find_dividend_stocks":
            symbols = ["JNJ", "PG", "KO", "PEP", "MCD", "VZ", "T", "XOM", "CVX", "ABBV",
                      "MRK", "PFE", "IBM", "MMM", "CAT", "HD", "LOW", "TGT", "WMT", "O"]

            candidates = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info

                    div_yield = (info.get("dividendYield") or 0) * 100
                    payout = (info.get("payoutRatio") or 0) * 100

                    if div_yield > 2 and payout < 80:
                        candidates.append({
                            "symbol": symbol,
                            "name": info.get("shortName"),
                            "dividend_yield": round(div_yield, 2),
                            "payout_ratio": round(payout, 2),
                            "5yr_avg_yield": info.get("fiveYearAvgDividendYield")
                        })
                except:
                    pass

            candidates.sort(key=lambda x: x["dividend_yield"], reverse=True)
            return [TextContent(type="text", text=json.dumps({
                "dividend_stocks": candidates[:10],
                "criteria": "Yield >2%, Payout <80%"
            }, indent=2))]

        # === COMPARISON ===
        elif name == "compare_stocks":
            symbols = [s.upper() for s in arguments["symbols"]]
            period = arguments.get("period", "3mo")

            results = []
            for symbol in symbols[:5]:  # Limit to 5
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period=period)

                    if not hist.empty:
                        start = hist['Close'].iloc[0]
                        end = hist['Close'].iloc[-1]
                        returns = ((end - start) / start) * 100
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100

                        results.append({
                            "symbol": symbol,
                            "name": info.get("shortName"),
                            "price": round(end, 2),
                            "period_return": round(returns, 2),
                            "volatility": round(volatility, 2),
                            "pe_ratio": info.get("trailingPE"),
                            "market_cap_b": round(info.get("marketCap", 0) / 1e9, 2),
                            "dividend_yield": round((info.get("dividendYield") or 0) * 100, 2)
                        })
                except:
                    pass

            results.sort(key=lambda x: x["period_return"], reverse=True)
            return [TextContent(type="text", text=json.dumps({
                "comparison": results,
                "period": period,
                "best_performer": results[0]["symbol"] if results else None
            }, indent=2))]

        # === HISTORICAL ===
        elif name == "get_historical_data":
            symbol = arguments["symbol"].upper()
            period = arguments.get("period", "1y")

            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            if hist.empty:
                return [TextContent(type="text", text=f"No data for {symbol}")]

            returns = hist['Close'].pct_change().dropna()

            result = {
                "symbol": symbol,
                "period": period,
                "start_date": str(hist.index[0].date()),
                "end_date": str(hist.index[-1].date()),
                "start_price": round(hist['Close'].iloc[0], 2),
                "end_price": round(hist['Close'].iloc[-1], 2),
                "high": round(hist['High'].max(), 2),
                "low": round(hist['Low'].min(), 2),
                "total_return": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2),
                "annualized_volatility": round(returns.std() * np.sqrt(252) * 100, 2),
                "max_drawdown": round((hist['Close'] / hist['Close'].cummax() - 1).min() * 100, 2),
                "avg_daily_volume": int(hist['Volume'].mean())
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "backtest_strategy":
            symbol = arguments["symbol"].upper()
            fast_ma = arguments.get("fast_ma", 20)
            slow_ma = arguments.get("slow_ma", 50)
            period = arguments.get("period", "2y")

            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            if hist.empty or len(hist) < slow_ma:
                return [TextContent(type="text", text=f"Insufficient data for {symbol}")]

            # Calculate moving averages
            hist['fast_ma'] = hist['Close'].rolling(fast_ma).mean()
            hist['slow_ma'] = hist['Close'].rolling(slow_ma).mean()

            # Generate signals
            hist['signal'] = 0
            hist.loc[hist['fast_ma'] > hist['slow_ma'], 'signal'] = 1
            hist.loc[hist['fast_ma'] <= hist['slow_ma'], 'signal'] = -1

            # Calculate strategy returns
            hist['daily_return'] = hist['Close'].pct_change()
            hist['strategy_return'] = hist['signal'].shift(1) * hist['daily_return']

            # Performance metrics
            total_return = (hist['strategy_return'].sum()) * 100
            buy_hold_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100

            # Count trades
            hist['trade'] = hist['signal'].diff().abs()
            num_trades = int(hist['trade'].sum() / 2)

            result = {
                "symbol": symbol,
                "strategy": f"{fast_ma}/{slow_ma} MA Crossover",
                "period": period,
                "strategy_return": round(total_return, 2),
                "buy_hold_return": round(buy_hold_return, 2),
                "outperformance": round(total_return - buy_hold_return, 2),
                "num_trades": num_trades,
                "current_signal": "buy" if hist['signal'].iloc[-1] == 1 else "sell"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === RISK ===
        elif name == "calculate_position_size":
            account = arguments["account_size"]
            risk_pct = arguments["risk_percent"]
            entry = arguments["entry_price"]
            stop = arguments["stop_loss"]

            risk_amount = account * (risk_pct / 100)
            risk_per_share = abs(entry - stop)

            if risk_per_share == 0:
                return [TextContent(type="text", text="Entry and stop loss cannot be same")]

            shares = int(risk_amount / risk_per_share)
            position_value = shares * entry

            result = {
                "recommended_shares": shares,
                "position_value": round(position_value, 2),
                "position_percent": round((position_value / account) * 100, 2),
                "risk_amount": round(risk_amount, 2),
                "risk_per_share": round(risk_per_share, 2),
                "max_loss": round(shares * risk_per_share, 2)
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "analyze_risk_reward":
            entry = arguments["entry_price"]
            stop = arguments["stop_loss"]
            target = arguments["target_price"]

            risk = abs(entry - stop)
            reward = abs(target - entry)
            ratio = reward / risk if risk > 0 else 0

            result = {
                "entry": entry,
                "stop_loss": stop,
                "target": target,
                "risk": round(risk, 2),
                "reward": round(reward, 2),
                "risk_reward_ratio": round(ratio, 2),
                "assessment": "favorable" if ratio >= 2 else "marginal" if ratio >= 1 else "unfavorable",
                "win_rate_needed": f"{round(100 / (1 + ratio), 1)}%" if ratio > 0 else "N/A"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === ALERTS ===
        elif name == "set_price_alert":
            alerts = load_alerts()

            alert = {
                "symbol": arguments["symbol"].upper(),
                "target_price": arguments["target_price"],
                "direction": arguments["direction"],
                "created": datetime.now().isoformat()
            }

            alerts["price_alerts"].append(alert)
            save_alerts(alerts)

            return [TextContent(type="text", text=f"Alert set: {alert['symbol']} {alert['direction']} ${alert['target_price']}")]

        elif name == "check_alerts":
            alerts = load_alerts()
            triggered = []
            remaining = []

            for alert in alerts["price_alerts"]:
                try:
                    stock = yf.Ticker(alert["symbol"])
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]

                        if alert["direction"] == "above" and current >= alert["target_price"]:
                            triggered.append({**alert, "current_price": round(current, 2)})
                        elif alert["direction"] == "below" and current <= alert["target_price"]:
                            triggered.append({**alert, "current_price": round(current, 2)})
                        else:
                            remaining.append(alert)
                except:
                    remaining.append(alert)

            alerts["price_alerts"] = remaining
            save_alerts(alerts)

            return [TextContent(type="text", text=json.dumps({
                "triggered_alerts": triggered,
                "active_alerts": remaining
            }, indent=2))]

        # === CRYPTO ===
        elif name == "get_crypto_prices":
            cryptos = {
                "BTC-USD": "Bitcoin",
                "ETH-USD": "Ethereum",
                "SOL-USD": "Solana",
                "XRP-USD": "Ripple",
                "ADA-USD": "Cardano",
                "DOGE-USD": "Dogecoin"
            }

            results = []
            for symbol, name in cryptos.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change = ((current - prev) / prev) * 100

                        results.append({
                            "name": name,
                            "symbol": symbol.replace("-USD", ""),
                            "price": round(current, 2),
                            "change_24h": round(change, 2)
                        })
                except:
                    pass

            return [TextContent(type="text", text=json.dumps({"crypto": results}, indent=2))]

        # === SEARCH ===
        elif name == "search_stocks":
            query = arguments["query"].upper()

            # Common stocks database
            stocks = {
                "AAPL": "Apple Inc", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc",
                "AMZN": "Amazon.com Inc", "NVDA": "NVIDIA Corporation", "META": "Meta Platforms Inc",
                "TSLA": "Tesla Inc", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
                "V": "Visa Inc", "UNH": "UnitedHealth Group", "JNJ": "Johnson & Johnson",
                "WMT": "Walmart Inc", "PG": "Procter & Gamble", "MA": "Mastercard Inc",
                "HD": "Home Depot", "CVX": "Chevron Corporation", "MRK": "Merck & Co",
                "ABBV": "AbbVie Inc", "PFE": "Pfizer Inc", "KO": "Coca-Cola Company",
                "PEP": "PepsiCo Inc", "COST": "Costco Wholesale", "TMO": "Thermo Fisher Scientific",
                "AVGO": "Broadcom Inc", "MCD": "McDonald's Corporation", "DIS": "Walt Disney Company",
                "CSCO": "Cisco Systems", "ACN": "Accenture", "ABT": "Abbott Laboratories",
                "NKE": "Nike Inc", "INTC": "Intel Corporation", "AMD": "Advanced Micro Devices",
                "CRM": "Salesforce Inc", "NFLX": "Netflix Inc", "ADBE": "Adobe Inc",
                "PYPL": "PayPal Holdings", "QCOM": "Qualcomm Inc", "TXN": "Texas Instruments",
                "ORCL": "Oracle Corporation", "IBM": "IBM", "GE": "General Electric"
            }

            results = []
            for symbol, name in stocks.items():
                if query in symbol or query.lower() in name.lower():
                    results.append({"symbol": symbol, "name": name})

            return [TextContent(type="text", text=json.dumps({"results": results[:15]}, indent=2))]

        # ============== NEW INVESTMENT DECISION TOOLS ==============

        # === ANALYST RATINGS ===
        elif name == "get_analyst_ratings":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            # Get recommendations
            try:
                recs = stock.recommendations
                recent_recs = []
                if recs is not None and not recs.empty:
                    recent = recs.tail(10)
                    for idx, row in recent.iterrows():
                        recent_recs.append({
                            "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                            "firm": row.get("Firm", "N/A"),
                            "to_grade": row.get("To Grade", "N/A"),
                            "from_grade": row.get("From Grade", "N/A"),
                            "action": row.get("Action", "N/A")
                        })
            except:
                recent_recs = []

            # Count ratings
            rating_counts = {"buy": 0, "hold": 0, "sell": 0}
            for rec in recent_recs:
                grade = rec.get("to_grade", "").lower()
                if any(b in grade for b in ["buy", "outperform", "overweight", "accumulate"]):
                    rating_counts["buy"] += 1
                elif any(s in grade for s in ["sell", "underperform", "underweight", "reduce"]):
                    rating_counts["sell"] += 1
                else:
                    rating_counts["hold"] += 1

            result = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "recommendation_mean": info.get("recommendationMean"),  # 1=Strong Buy, 5=Strong Sell
                "recommendation_key": info.get("recommendationKey"),
                "number_of_analysts": info.get("numberOfAnalystOpinions"),
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "current_price": info.get("currentPrice"),
                "upside_potential": round(((info.get("targetMeanPrice", 0) / info.get("currentPrice", 1)) - 1) * 100, 1) if info.get("currentPrice") and info.get("targetMeanPrice") else None,
                "rating_summary": rating_counts,
                "recent_recommendations": recent_recs[:5]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_price_targets":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            current = info.get("currentPrice", info.get("regularMarketPrice", 0))

            result = {
                "symbol": symbol,
                "current_price": current,
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "number_of_analysts": info.get("numberOfAnalystOpinions"),
                "upside_to_low": round(((info.get("targetLowPrice", 0) / current) - 1) * 100, 1) if current else None,
                "upside_to_mean": round(((info.get("targetMeanPrice", 0) / current) - 1) * 100, 1) if current else None,
                "upside_to_high": round(((info.get("targetHighPrice", 0) / current) - 1) * 100, 1) if current else None
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_upgrades_downgrades":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            try:
                recs = stock.recommendations
                if recs is not None and not recs.empty:
                    upgrades = []
                    downgrades = []
                    for idx, row in recs.tail(20).iterrows():
                        action = row.get("Action", "").lower()
                        entry = {
                            "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                            "firm": row.get("Firm", "N/A"),
                            "from": row.get("From Grade", "N/A"),
                            "to": row.get("To Grade", "N/A")
                        }
                        if "upgrade" in action or "up" in action:
                            upgrades.append(entry)
                        elif "downgrade" in action or "down" in action:
                            downgrades.append(entry)

                    return [TextContent(type="text", text=json.dumps({
                        "symbol": symbol,
                        "recent_upgrades": upgrades[:5],
                        "recent_downgrades": downgrades[:5]
                    }, indent=2))]
            except:
                pass

            return [TextContent(type="text", text=json.dumps({"symbol": symbol, "message": "No upgrade/downgrade data available"}, indent=2))]

        # === SHORT INTEREST ===
        elif name == "get_short_interest":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            shares_short = info.get("sharesShort", 0)
            shares_outstanding = info.get("sharesOutstanding", 1)
            avg_volume = info.get("averageVolume", 1)

            short_percent = (shares_short / shares_outstanding * 100) if shares_outstanding else 0
            days_to_cover = (shares_short / avg_volume) if avg_volume else 0

            result = {
                "symbol": symbol,
                "shares_short": shares_short,
                "shares_outstanding": shares_outstanding,
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "short_percent_of_shares": round(short_percent, 2),
                "short_ratio": info.get("shortRatio"),
                "days_to_cover": round(days_to_cover, 1),
                "prior_short_shares": info.get("sharesShortPriorMonth"),
                "short_change": round(((shares_short / info.get("sharesShortPriorMonth", shares_short)) - 1) * 100, 1) if info.get("sharesShortPriorMonth") else 0,
                "squeeze_risk": "HIGH" if short_percent > 20 else "MODERATE" if short_percent > 10 else "LOW"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === OPTIONS DATA ===
        elif name == "get_options_chain":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            try:
                expirations = stock.options
                if not expirations:
                    return [TextContent(type="text", text=json.dumps({"error": "No options available"}, indent=2))]

                # Use specified expiration or nearest
                exp = arguments.get("expiration")
                if exp and exp in expirations:
                    target_exp = exp
                else:
                    target_exp = expirations[0]  # Nearest expiration

                chain = stock.option_chain(target_exp)
                calls = chain.calls
                puts = chain.puts

                # Get current price
                info = stock.info
                current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))

                # Find ATM options
                atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:5]]
                atm_puts = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:5]]

                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "expiration": target_exp,
                    "available_expirations": list(expirations[:5]),
                    "total_call_volume": int(calls['volume'].sum()) if 'volume' in calls else 0,
                    "total_put_volume": int(puts['volume'].sum()) if 'volume' in puts else 0,
                    "total_call_oi": int(calls['openInterest'].sum()) if 'openInterest' in calls else 0,
                    "total_put_oi": int(puts['openInterest'].sum()) if 'openInterest' in puts else 0,
                    "atm_calls": atm_calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records') if not atm_calls.empty else [],
                    "atm_puts": atm_puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records') if not atm_puts.empty else []
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_options_summary":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            try:
                expirations = stock.options
                if not expirations:
                    return [TextContent(type="text", text=json.dumps({"error": "No options available"}, indent=2))]

                # Analyze first few expirations
                total_call_oi = 0
                total_put_oi = 0
                total_call_vol = 0
                total_put_vol = 0
                all_strikes = []

                for exp in expirations[:3]:
                    try:
                        chain = stock.option_chain(exp)
                        total_call_oi += chain.calls['openInterest'].sum() if 'openInterest' in chain.calls else 0
                        total_put_oi += chain.puts['openInterest'].sum() if 'openInterest' in chain.puts else 0
                        total_call_vol += chain.calls['volume'].sum() if 'volume' in chain.calls else 0
                        total_put_vol += chain.puts['volume'].sum() if 'volume' in chain.puts else 0
                        all_strikes.extend(chain.calls['strike'].tolist())
                    except:
                        pass

                put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

                # Calculate max pain (strike with most open interest)
                max_pain = max(set(all_strikes), key=all_strikes.count) if all_strikes else 0

                info = stock.info
                current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))

                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "put_call_ratio": round(put_call_ratio, 2),
                    "put_call_sentiment": "BEARISH" if put_call_ratio > 1.2 else "BULLISH" if put_call_ratio < 0.8 else "NEUTRAL",
                    "total_call_open_interest": int(total_call_oi),
                    "total_put_open_interest": int(total_put_oi),
                    "total_call_volume": int(total_call_vol),
                    "total_put_volume": int(total_put_vol),
                    "estimated_max_pain": max_pain,
                    "expirations_analyzed": expirations[:3]
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        # === MARKET SENTIMENT ===
        elif name == "get_fear_greed_index":
            # Calculate Fear & Greed using multiple indicators
            indicators = {}

            # 1. VIX (Fear indicator)
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="1mo")
                if not vix_hist.empty:
                    vix_current = vix_hist['Close'].iloc[-1]
                    vix_avg = vix_hist['Close'].mean()
                    # VIX > 30 = Extreme Fear, VIX < 15 = Extreme Greed
                    vix_score = max(0, min(100, 100 - ((vix_current - 10) * 3.5)))
                    indicators["vix"] = {"value": round(vix_current, 1), "score": round(vix_score)}
            except:
                pass

            # 2. S&P 500 vs 125-day MA
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="6mo")
                if len(spy_hist) >= 125:
                    current = spy_hist['Close'].iloc[-1]
                    ma125 = spy_hist['Close'].rolling(125).mean().iloc[-1]
                    diff_pct = ((current / ma125) - 1) * 100
                    ma_score = max(0, min(100, 50 + (diff_pct * 5)))
                    indicators["sp500_momentum"] = {"vs_125ma": round(diff_pct, 1), "score": round(ma_score)}
            except:
                pass

            # 3. Put/Call ratio (market-wide)
            try:
                # Use SPY options as proxy
                spy = yf.Ticker("SPY")
                if spy.options:
                    chain = spy.option_chain(spy.options[0])
                    put_oi = chain.puts['openInterest'].sum()
                    call_oi = chain.calls['openInterest'].sum()
                    pc_ratio = put_oi / call_oi if call_oi > 0 else 1
                    # High PC ratio = fear, Low = greed
                    pc_score = max(0, min(100, 100 - (pc_ratio * 50)))
                    indicators["put_call_ratio"] = {"value": round(pc_ratio, 2), "score": round(pc_score)}
            except:
                pass

            # Calculate overall score
            if indicators:
                avg_score = sum(ind.get("score", 50) for ind in indicators.values()) / len(indicators)
            else:
                avg_score = 50

            # Determine sentiment
            if avg_score >= 75:
                sentiment = "EXTREME GREED"
            elif avg_score >= 55:
                sentiment = "GREED"
            elif avg_score >= 45:
                sentiment = "NEUTRAL"
            elif avg_score >= 25:
                sentiment = "FEAR"
            else:
                sentiment = "EXTREME FEAR"

            result = {
                "fear_greed_score": round(avg_score),
                "sentiment": sentiment,
                "interpretation": "Time to be cautious - others are greedy" if avg_score > 70 else "Time to look for opportunities - others are fearful" if avg_score < 30 else "Market sentiment is balanced",
                "indicators": indicators
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_market_breadth":
            # Market breadth using sector ETFs
            sectors = {
                "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
                "XLY": "Consumer Disc", "XLP": "Consumer Staples", "XLE": "Energy",
                "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate"
            }

            advancing = 0
            declining = 0
            unchanged = 0
            sector_data = []

            for symbol, name in sectors.items():
                try:
                    etf = yf.Ticker(symbol)
                    hist = etf.history(period="5d")
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change_pct = ((current / prev) - 1) * 100

                        if change_pct > 0.1:
                            advancing += 1
                        elif change_pct < -0.1:
                            declining += 1
                        else:
                            unchanged += 1

                        sector_data.append({
                            "sector": name,
                            "symbol": symbol,
                            "change_pct": round(change_pct, 2)
                        })
                except:
                    pass

            # Sort by performance
            sector_data.sort(key=lambda x: x["change_pct"], reverse=True)

            ad_ratio = advancing / declining if declining > 0 else advancing
            breadth = "BULLISH" if ad_ratio > 1.5 else "BEARISH" if ad_ratio < 0.67 else "MIXED"

            result = {
                "advancing_sectors": advancing,
                "declining_sectors": declining,
                "unchanged_sectors": unchanged,
                "advance_decline_ratio": round(ad_ratio, 2),
                "market_breadth": breadth,
                "sector_leaders": sector_data[:3],
                "sector_laggards": sector_data[-3:]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === CORRELATION ===
        elif name == "get_correlation_matrix":
            symbols = [s.upper() for s in arguments["symbols"]]

            if len(symbols) < 2:
                return [TextContent(type="text", text=json.dumps({"error": "Need at least 2 symbols"}, indent=2))]

            # Get historical prices
            prices = {}
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1y")
                    if not hist.empty:
                        prices[symbol] = hist['Close']
                except:
                    pass

            if len(prices) < 2:
                return [TextContent(type="text", text=json.dumps({"error": "Could not get data for enough symbols"}, indent=2))]

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(prices)
            correlation = df.corr()

            # Convert to readable format
            corr_data = {}
            for s1 in correlation.index:
                corr_data[s1] = {}
                for s2 in correlation.columns:
                    corr_data[s1][s2] = round(correlation.loc[s1, s2], 3)

            # Find highly correlated pairs
            high_corr = []
            low_corr = []
            for i, s1 in enumerate(correlation.index):
                for s2 in correlation.columns[i+1:]:
                    c = correlation.loc[s1, s2]
                    if c > 0.7:
                        high_corr.append({"pair": f"{s1}-{s2}", "correlation": round(c, 3)})
                    elif c < 0.3:
                        low_corr.append({"pair": f"{s1}-{s2}", "correlation": round(c, 3)})

            result = {
                "correlation_matrix": corr_data,
                "highly_correlated": high_corr,
                "diversifying_pairs": low_corr,
                "diversification_tip": "Low correlation pairs provide better diversification"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # === SEC FILINGS ===
        elif name == "get_sec_filings":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)

            # Get SEC filings via Yahoo Finance
            try:
                filings = stock.sec_filings
                if filings is not None and not filings.empty:
                    recent_filings = []
                    for idx, row in filings.head(10).iterrows():
                        recent_filings.append({
                            "date": str(row.get("date", "")),
                            "type": row.get("type", ""),
                            "title": row.get("title", ""),
                            "link": row.get("edgarUrl", "")
                        })
                    return [TextContent(type="text", text=json.dumps({
                        "symbol": symbol,
                        "recent_filings": recent_filings
                    }, indent=2))]
            except:
                pass

            # Fallback: Just provide info about where to find filings
            info = stock.info
            return [TextContent(type="text", text=json.dumps({
                "symbol": symbol,
                "cik": info.get("companyOfficers", [{}])[0].get("fiscalYear") if info.get("companyOfficers") else None,
                "sec_website": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type=&dateb=&owner=include&count=40",
                "note": "Visit SEC EDGAR for complete filings"
            }, indent=2))]

        # === PAPER TRADING ===
        elif name == "paper_trade":
            paper_file = SCRIPT_DIR / "paper_portfolio.json"
            portfolio = load_json(paper_file, {"cash": 100000, "holdings": [], "transactions": []})

            symbol = arguments["symbol"].upper()
            action = arguments["action"]
            shares = arguments["shares"]
            order_type = arguments.get("order_type", "market")

            # Get current price
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            if hist.empty:
                return [TextContent(type="text", text=json.dumps({"error": f"Could not get price for {symbol}"}, indent=2))]

            price = hist['Close'].iloc[-1]
            if order_type == "limit":
                limit_price = arguments.get("limit_price")
                if not limit_price:
                    return [TextContent(type="text", text=json.dumps({"error": "Limit price required for limit orders"}, indent=2))]
                if action == "buy" and limit_price < price:
                    return [TextContent(type="text", text=json.dumps({"status": "pending", "message": f"Limit buy at ${limit_price} - current price ${round(price, 2)} is higher"}, indent=2))]
                elif action == "sell" and limit_price > price:
                    return [TextContent(type="text", text=json.dumps({"status": "pending", "message": f"Limit sell at ${limit_price} - current price ${round(price, 2)} is lower"}, indent=2))]
                price = limit_price

            total_cost = price * shares

            if action == "buy":
                if total_cost > portfolio["cash"]:
                    return [TextContent(type="text", text=json.dumps({"error": f"Insufficient funds. Need ${round(total_cost, 2)}, have ${round(portfolio['cash'], 2)}"}, indent=2))]

                portfolio["cash"] -= total_cost

                # Check if already holding
                existing = next((h for h in portfolio["holdings"] if h["symbol"] == symbol), None)
                if existing:
                    # Average in
                    total_shares = existing["shares"] + shares
                    avg_cost = ((existing["shares"] * existing["avg_cost"]) + total_cost) / total_shares
                    existing["shares"] = total_shares
                    existing["avg_cost"] = round(avg_cost, 2)
                else:
                    portfolio["holdings"].append({
                        "symbol": symbol,
                        "shares": shares,
                        "avg_cost": round(price, 2),
                        "purchase_date": datetime.now().isoformat()
                    })

            elif action == "sell":
                existing = next((h for h in portfolio["holdings"] if h["symbol"] == symbol), None)
                if not existing:
                    return [TextContent(type="text", text=json.dumps({"error": f"No {symbol} shares to sell"}, indent=2))]
                if shares > existing["shares"]:
                    return [TextContent(type="text", text=json.dumps({"error": f"Only have {existing['shares']} shares"}, indent=2))]

                portfolio["cash"] += total_cost
                existing["shares"] -= shares

                if existing["shares"] == 0:
                    portfolio["holdings"].remove(existing)

            # Record transaction
            portfolio["transactions"].append({
                "date": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "price": round(price, 2),
                "total": round(total_cost, 2)
            })

            portfolio["last_updated"] = datetime.now().isoformat()
            save_json(paper_file, portfolio)

            return [TextContent(type="text", text=json.dumps({
                "status": "executed",
                "action": action,
                "symbol": symbol,
                "shares": shares,
                "price": round(price, 2),
                "total": round(total_cost, 2),
                "remaining_cash": round(portfolio["cash"], 2)
            }, indent=2))]

        elif name == "get_paper_portfolio":
            paper_file = SCRIPT_DIR / "paper_portfolio.json"
            portfolio = load_json(paper_file, {"cash": 100000, "holdings": [], "transactions": []})

            total_value = portfolio["cash"]
            holdings_detail = []

            for holding in portfolio["holdings"]:
                try:
                    stock = yf.Ticker(holding["symbol"])
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        market_value = current_price * holding["shares"]
                        cost_basis = holding["avg_cost"] * holding["shares"]
                        gain_loss = market_value - cost_basis
                        gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0

                        total_value += market_value

                        holdings_detail.append({
                            "symbol": holding["symbol"],
                            "shares": holding["shares"],
                            "avg_cost": holding["avg_cost"],
                            "current_price": round(current_price, 2),
                            "market_value": round(market_value, 2),
                            "gain_loss": round(gain_loss, 2),
                            "gain_loss_pct": round(gain_loss_pct, 2)
                        })
                except:
                    pass

            starting_cash = 100000  # Assumed starting amount
            total_return = ((total_value / starting_cash) - 1) * 100

            result = {
                "cash": round(portfolio["cash"], 2),
                "holdings_value": round(total_value - portfolio["cash"], 2),
                "total_value": round(total_value, 2),
                "total_return_pct": round(total_return, 2),
                "holdings": holdings_detail,
                "recent_transactions": portfolio.get("transactions", [])[-5:]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "reset_paper_portfolio":
            starting_cash = arguments.get("starting_cash", 100000)
            paper_file = SCRIPT_DIR / "paper_portfolio.json"

            portfolio = {
                "cash": starting_cash,
                "holdings": [],
                "transactions": [],
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }

            save_json(paper_file, portfolio)

            return [TextContent(type="text", text=json.dumps({
                "status": "reset",
                "starting_cash": starting_cash,
                "message": "Paper trading portfolio reset successfully"
            }, indent=2))]

        # === EARNINGS ESTIMATES ===
        elif name == "get_earnings_estimates":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            # Get earnings calendar
            try:
                calendar = stock.calendar
                earnings_date = calendar.get("Earnings Date") if calendar else None
            except:
                earnings_date = None

            # Get earnings history
            try:
                earnings = stock.earnings_history
                earnings_hist = []
                if earnings is not None and not earnings.empty:
                    for idx, row in earnings.tail(8).iterrows():
                        earnings_hist.append({
                            "date": str(idx) if not hasattr(idx, 'strftime') else idx.strftime("%Y-%m-%d"),
                            "eps_estimate": row.get("epsEstimate"),
                            "eps_actual": row.get("epsActual"),
                            "surprise": row.get("epsDifference"),
                            "surprise_pct": round(row.get("surprisePercent", 0) * 100, 1) if row.get("surprisePercent") else None
                        })
            except:
                earnings_hist = []

            result = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "next_earnings_date": str(earnings_date[0]) if earnings_date and len(earnings_date) > 0 else None,
                "forward_eps": info.get("forwardEps"),
                "trailing_eps": info.get("trailingEps"),
                "peg_ratio": info.get("pegRatio"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_history": earnings_hist
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_upcoming_earnings":
            # Check popular stocks for upcoming earnings
            watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
            upcoming = []

            today = datetime.now()
            next_week = today + timedelta(days=7)

            for symbol in watchlist:
                try:
                    stock = yf.Ticker(symbol)
                    calendar = stock.calendar
                    if calendar and "Earnings Date" in calendar:
                        earnings_date = calendar["Earnings Date"]
                        if earnings_date and len(earnings_date) > 0:
                            ed = earnings_date[0]
                            if hasattr(ed, 'date'):
                                ed = ed.date()
                            if today.date() <= ed <= next_week.date():
                                upcoming.append({
                                    "symbol": symbol,
                                    "earnings_date": str(ed)
                                })
                except:
                    pass

            return [TextContent(type="text", text=json.dumps({
                "upcoming_earnings_this_week": upcoming,
                "note": "Add stocks to watchlist for more complete coverage"
            }, indent=2))]

        # === COMPANY EVENTS ===
        elif name == "get_company_events":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info

            events = []

            # Earnings
            try:
                calendar = stock.calendar
                if calendar and "Earnings Date" in calendar:
                    ed = calendar["Earnings Date"]
                    if ed and len(ed) > 0:
                        events.append({"type": "earnings", "date": str(ed[0])})
            except:
                pass

            # Ex-Dividend
            if info.get("exDividendDate"):
                ex_date = datetime.fromtimestamp(info["exDividendDate"])
                events.append({"type": "ex_dividend", "date": ex_date.strftime("%Y-%m-%d")})

            # Dividend Date
            if info.get("dividendDate"):
                div_date = datetime.fromtimestamp(info["dividendDate"])
                events.append({"type": "dividend_payment", "date": div_date.strftime("%Y-%m-%d")})

            result = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "upcoming_events": events,
                "fiscal_year_end": info.get("lastFiscalYearEnd"),
                "most_recent_quarter": info.get("mostRecentQuarter")
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # === ETF ANALYSIS ===
        elif name == "get_etf_holdings":
            symbol = arguments["symbol"].upper()
            etf = yf.Ticker(symbol)
            info = etf.info

            # Check if it's an ETF
            if info.get("quoteType") != "ETF":
                return [TextContent(type="text", text=json.dumps({"error": f"{symbol} is not an ETF"}, indent=2))]

            holdings = []
            try:
                # Get top holdings if available
                fund_holdings = etf.funds_data
                if fund_holdings:
                    top_holdings = fund_holdings.top_holdings
                    if top_holdings is not None and not top_holdings.empty:
                        for idx, row in top_holdings.head(10).iterrows():
                            holdings.append({
                                "symbol": idx,
                                "name": row.get("holdingName", idx),
                                "weight": round(row.get("holdingPercent", 0) * 100, 2) if row.get("holdingPercent") else None
                            })
            except:
                pass

            result = {
                "etf": symbol,
                "name": info.get("longName", symbol),
                "category": info.get("category"),
                "total_assets": info.get("totalAssets"),
                "expense_ratio": info.get("expenseRatio"),
                "yield": info.get("yield"),
                "top_holdings": holdings if holdings else "Holdings data not available via API"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "find_etfs_holding_stock":
            symbol = arguments["symbol"].upper()

            # Common ETFs and their major holdings
            etf_data = {
                "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ"],
                "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "PEP"],
                "DIA": ["UNH", "GS", "MSFT", "HD", "MCD", "AMGN", "CAT", "V", "BA", "JPM"],
                "IWM": ["SMCI", "MSTR", "MRNA", "PLUG", "RKT"],  # Small caps
                "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "ADBE", "ORCL", "CSCO", "ACN"],
                "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "SPGI", "GS", "AXP", "MS"],
                "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "DHR"],
                "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "PXD", "VLO", "OXY"],
                "ARKK": ["TSLA", "ROKU", "COIN", "SQ", "SPOT", "PATH", "ZM", "TWLO", "DKNG", "U"]
            }

            found_in = []
            for etf, holdings in etf_data.items():
                if symbol in holdings:
                    found_in.append(etf)

            return [TextContent(type="text", text=json.dumps({
                "symbol": symbol,
                "found_in_etfs": found_in,
                "note": "This is a simplified list. Use ETF research tools for complete data."
            }, indent=2))]

        # === INVESTMENT CHECKLIST ===
        elif name == "get_investment_checklist":
            symbol = arguments["symbol"].upper()
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")

            checklist = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "current_price": info.get("currentPrice"),
                "checks": []
            }

            # Valuation checks
            pe = info.get("trailingPE", 0)
            forward_pe = info.get("forwardPE", 0)
            peg = info.get("pegRatio", 0)

            checklist["checks"].append({
                "category": "VALUATION",
                "pe_ratio": pe,
                "forward_pe": forward_pe,
                "peg_ratio": peg,
                "assessment": "PASS" if (pe and pe < 25 and forward_pe and forward_pe < pe) else "REVIEW"
            })

            # Growth checks
            revenue_growth = info.get("revenueGrowth", 0)
            earnings_growth = info.get("earningsGrowth", 0)

            checklist["checks"].append({
                "category": "GROWTH",
                "revenue_growth": round(revenue_growth * 100, 1) if revenue_growth else None,
                "earnings_growth": round(earnings_growth * 100, 1) if earnings_growth else None,
                "assessment": "PASS" if (revenue_growth and revenue_growth > 0.1) else "REVIEW"
            })

            # Financial health
            debt_equity = info.get("debtToEquity", 0)
            current_ratio = info.get("currentRatio", 0)

            checklist["checks"].append({
                "category": "FINANCIAL_HEALTH",
                "debt_to_equity": debt_equity,
                "current_ratio": current_ratio,
                "assessment": "PASS" if (debt_equity and debt_equity < 100 and current_ratio and current_ratio > 1) else "REVIEW"
            })

            # Profitability
            profit_margin = info.get("profitMargins", 0)
            roe = info.get("returnOnEquity", 0)

            checklist["checks"].append({
                "category": "PROFITABILITY",
                "profit_margin": round(profit_margin * 100, 1) if profit_margin else None,
                "return_on_equity": round(roe * 100, 1) if roe else None,
                "assessment": "PASS" if (profit_margin and profit_margin > 0.1) else "REVIEW"
            })

            # Technical check
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                ma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else ma50

                checklist["checks"].append({
                    "category": "TECHNICAL",
                    "above_50ma": current > ma50,
                    "above_200ma": current > ma200,
                    "assessment": "PASS" if current > ma50 and current > ma200 else "CAUTION" if current > ma200 else "REVIEW"
                })

            # Analyst sentiment
            rec_mean = info.get("recommendationMean")
            num_analysts = info.get("numberOfAnalystOpinions", 0)

            checklist["checks"].append({
                "category": "ANALYST_SENTIMENT",
                "recommendation": info.get("recommendationKey"),
                "recommendation_score": rec_mean,
                "number_of_analysts": num_analysts,
                "assessment": "PASS" if (rec_mean and rec_mean <= 2.5) else "REVIEW"
            })

            # Overall score
            passes = sum(1 for c in checklist["checks"] if c["assessment"] == "PASS")
            total = len(checklist["checks"])

            checklist["overall"] = {
                "passes": passes,
                "total": total,
                "score_pct": round((passes / total) * 100),
                "verdict": "STRONG BUY" if passes >= 5 else "CONSIDER" if passes >= 3 else "MORE RESEARCH NEEDED"
            }

            return [TextContent(type="text", text=json.dumps(checklist, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
