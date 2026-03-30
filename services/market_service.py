import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from core.config import settings


class MarketService:
    """
    Uses Yahoo Finance (free, no key needed) as primary.
    Falls back to Alpha Vantage if ALPHA_VANTAGE_KEY is set.
    """

    YF_BASE = "https://query1.finance.yahoo.com/v8/finance"
    YF_V7 = "https://query1.finance.yahoo.com/v7/finance"

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                url = f"{self.YF_BASE}/chart/{ticker}?interval=1d&range=1d"
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                data = r.json()
                meta = data["chart"]["result"][0]["meta"]
                return {
                    "ticker": ticker,
                    "price": round(meta.get("regularMarketPrice", 0), 2),
                    "prev_close": round(meta.get("previousClose", 0), 2),
                    "change": round(meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0), 2),
                    "change_pct": round(
                        (meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0))
                        / max(meta.get("previousClose", 1), 1) * 100, 2
                    ),
                    "volume": meta.get("regularMarketVolume", 0),
                    "market_cap": meta.get("marketCap"),
                    "currency": meta.get("currency", "USD"),
                    "exchange": meta.get("exchangeName", ""),
                }
            except Exception as e:
                return {"ticker": ticker, "error": str(e), "price": None}

    async def get_history(self, ticker: str, period: str = "1mo") -> Dict[str, Any]:
        period_map = {"1d": "1d", "1w": "5d", "1mo": "1mo", "3mo": "3mo", "1y": "1y", "5y": "5y"}
        yf_range = period_map.get(period, "1mo")
        interval = "1d" if yf_range not in ("1d",) else "5m"

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                url = f"{self.YF_BASE}/chart/{ticker}?interval={interval}&range={yf_range}"
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                data = r.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                closes = result["indicators"]["quote"][0]["close"]
                opens = result["indicators"]["quote"][0].get("open", [])
                highs = result["indicators"]["quote"][0].get("high", [])
                lows = result["indicators"]["quote"][0].get("low", [])
                volumes = result["indicators"]["quote"][0].get("volume", [])

                candles = []
                for i, ts in enumerate(timestamps):
                    if closes[i] is None:
                        continue
                    candles.append({
                        "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                        "open": round(opens[i] or 0, 2),
                        "high": round(highs[i] or 0, 2),
                        "low": round(lows[i] or 0, 2),
                        "close": round(closes[i] or 0, 2),
                        "volume": volumes[i] or 0,
                    })
                return {"ticker": ticker, "period": period, "candles": candles}
            except Exception as e:
                return {"ticker": ticker, "error": str(e), "candles": []}

    async def search(self, query: str) -> List[Dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                url = f"{self.YF_V7}/finance/search?q={query}&quotesCount=6&newsCount=0"
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                data = r.json()
                quotes = data.get("quotes", [])
                return [
                    {
                        "ticker": q.get("symbol"),
                        "name": q.get("longname") or q.get("shortname"),
                        "exchange": q.get("exchange"),
                        "type": q.get("quoteType"),
                    }
                    for q in quotes[:6]
                    if q.get("symbol")
                ]
            except Exception as e:
                return []

    async def get_news(self, ticker: str) -> List[Dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                url = f"{self.YF_V7}/finance/search?q={ticker}&quotesCount=0&newsCount=8"
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                data = r.json()
                news = data.get("news", [])
                return [
                    {
                        "title": n.get("title"),
                        "publisher": n.get("publisher"),
                        "link": n.get("link"),
                        "published": datetime.fromtimestamp(n.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                    }
                    for n in news[:8]
                ]
            except Exception as e:
                return []

    async def get_multiple_quotes(self, tickers: List[str]) -> List[Dict]:
        tasks = [self.get_quote(t) for t in tickers]
        return await asyncio.gather(*tasks)
