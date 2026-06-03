"""
tools.py — CrewAI tools exposed to the analysis agents.

Each tool is a sync wrapper. Async services (MarketService) are bridged with
asyncio.run since the crew runs inside asyncio.to_thread (no active loop).

Tools:
  - RAGDocumentTool  : semantic search over uploaded documents
  - MarketQuoteTool  : live price / volume / market cap for a ticker
  - MarketHistoryTool: OHLC history summary for a ticker
  - MarketNewsTool   : recent headlines for a ticker
  - PeerCompTool     : sector peer comparables for a ticker
  - WebSearchTool    : SEC EDGAR filings search beyond Yahoo
  - SQLQueryTool     : cross-document lookups against the analyses/documents DB
  - CalculatorTool   : safe numeric expression evaluator
"""

from __future__ import annotations

import ast
import asyncio
import json
import operator as _op
from typing import Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from services.market_service import MarketService
from services.rag_service import RAGService


# ── Safe expression evaluator ─────────────────────────────────────────────────

_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {ast.dump(node)}")


def safe_eval(expr: str) -> float:
    return _eval_node(ast.parse(expr, mode="eval").body)


# ── RAG document tool ─────────────────────────────────────────────────────────

class RAGDocumentTool(BaseTool):
    name: str = "search_documents"
    description: str = (
        "Search the uploaded financial documents for specific data. "
        "Input: short keyword phrase (e.g. 'revenue 2023', 'operating margin', 'debt'). "
        "Returns the most relevant passages."
    )
    rag: RAGService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        if self.rag is None:
            return "RAG not initialised."
        return self.rag.retrieve(query.strip(), top_k=2) or "No data found."


# ── Calculator tool ───────────────────────────────────────────────────────────

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "Evaluate a numeric expression. Use for ALL ratios, percentages, and arithmetic — "
        "never compute math in your head. "
        "Input: a math expression using + - * / ** % and parentheses. "
        "Example: '(450.2 - 380.5) / 380.5 * 100' for percent change."
    )

    def _run(self, expression: str) -> str:
        try:
            result = safe_eval(expression.strip())
            return f"{result:.6f}".rstrip("0").rstrip(".") if isinstance(result, float) else str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"


# ── Market tools ──────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a sync context — safe inside to_thread."""
    return asyncio.run(coro)


class MarketQuoteTool(BaseTool):
    name: str = "market_quote"
    description: str = (
        "Get the LIVE current market price, volume, market cap, and daily change for a stock ticker. "
        "Input: a single ticker symbol (e.g. 'AAPL', 'TSLA', 'MSFT'). "
        "Use this to compare current price against document figures and compute valuation ratios."
    )
    market: MarketService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, ticker: str) -> str:
        data = _run_async(self.market.get_quote(ticker.strip().upper()))
        if data.get("error"):
            return f"Error fetching {ticker}: {data['error']}"
        return json.dumps(data, default=str)


class HistoryInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    period: str = Field(default="1mo", description="One of: 1d, 1w, 1mo, 3mo, 1y, 5y")


class MarketHistoryTool(BaseTool):
    name: str = "market_history"
    description: str = (
        "Get historical price summary for a ticker — start/end prices, high, low, and % change. "
        "Use this to assess volatility, momentum, and price trends. "
        "Periods: 1d, 1w, 1mo, 3mo, 1y, 5y."
    )
    args_schema: Type[BaseModel] = HistoryInput
    market: MarketService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, ticker: str, period: str = "1mo") -> str:
        data = _run_async(self.market.get_history(ticker.strip().upper(), period.strip()))
        if data.get("error"):
            return f"Error fetching history for {ticker}: {data['error']}"
        candles = data.get("candles", [])
        if not candles:
            return f"No history data for {ticker}"
        prices = [c["close"] for c in candles if c.get("close")]
        if not prices:
            return f"No valid prices for {ticker}"
        first, last = candles[0], candles[-1]
        change_pct = (
            round((last["close"] - first["close"]) / first["close"] * 100, 2)
            if first["close"] else 0
        )
        return json.dumps({
            "ticker": ticker.upper(),
            "period": period,
            "start_date": first["date"],
            "end_date": last["date"],
            "start_close": first["close"],
            "end_close": last["close"],
            "high": max(prices),
            "low": min(prices),
            "change_pct": change_pct,
            "data_points": len(candles),
        })


class MarketNewsTool(BaseTool):
    name: str = "market_news"
    description: str = (
        "Get recent news headlines for a stock ticker. "
        "Input: a single ticker symbol (e.g. 'AAPL'). "
        "Returns up to 8 recent headlines with publisher and date. "
        "Use this to identify material events, catalysts, or risk signals."
    )
    market: MarketService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, ticker: str) -> str:
        news = _run_async(self.market.get_news(ticker.strip().upper()))
        if not news:
            return f"No news found for {ticker}"
        return "\n".join(
            f"- {n.get('title', 'untitled')} ({n.get('publisher', 'unknown')}, {n.get('published', '')})"
            for n in news[:8]
        )


class PeerCompTool(BaseTool):
    name: str = "peer_comparison"
    description: str = (
        "Fetch sector peer comparables for a ticker — peer tickers, market caps, "
        "and daily % change. Input: a single ticker symbol (e.g. 'AAPL'). "
        "Use this to assess relative size and momentum versus sector peers when "
        "computing valuation multiples."
    )
    market: MarketService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, ticker: str) -> str:
        data = _run_async(self.market.get_peer_comp(ticker.strip().upper()))
        if data.get("error") and not data.get("peers"):
            return f"No peer comparables for {ticker}: {data['error']}"
        return json.dumps(data, default=str)


# ── Web search (SEC EDGAR) ────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search SEC EDGAR full-text index for recent filings (10-K, 10-Q, 8-K) "
        "and material disclosures beyond Yahoo News. "
        "Input: free-text query (e.g. 'Apple revenue guidance', 'Tesla margin risk'). "
        "Returns up to 5 recent filings with form type, company, filing date."
    )

    def _run(self, query: str) -> str:
        try:
            with httpx.Client(timeout=12) as client:
                r = client.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": query.strip(), "forms": "10-K,10-Q,8-K"},
                    headers={"User-Agent": "FinAnalyzer research@finanalyzer.app"},
                )
                r.raise_for_status()
                data = r.json()
            hits = data.get("hits", {}).get("hits", [])[:5]
            if not hits:
                return f"No EDGAR filings found for '{query}'."
            lines = []
            for h in hits:
                src = h.get("_source", {})
                names = src.get("display_names") or ["?"]
                lines.append(
                    f"- [{src.get('form', '?')}] {names[0]} "
                    f"filed {src.get('file_date', '?')} "
                    f"(adsh {src.get('adsh', '?')})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"EDGAR search error: {e}"


# ── SQL query tool (cross-document DB lookups) ────────────────────────────────

class SQLQueryInput(BaseModel):
    operation: str = Field(
        ...,
        description=(
            "One of: 'recent_analyses' (last N analyses with recommendation+risk), "
            "'high_risk_analyses' (analyses with risk_score >= threshold), "
            "'list_documents' (recently uploaded documents)."
        ),
    )
    user_id: str = Field(default="", description="Optional user_id filter")
    risk_threshold: float = Field(default=7.0, description="Used by high_risk_analyses")
    limit: int = Field(default=5, description="Max rows to return (capped at 20)")


class SQLQueryTool(BaseTool):
    name: str = "sql_query"
    description: str = (
        "Query prior analyses and uploaded documents in the database for cross-document "
        "context (e.g. has this company been analysed before? what was the prior "
        "recommendation? which docs are on file?). "
        "Operations: recent_analyses, high_risk_analyses, list_documents."
    )
    args_schema: Type[BaseModel] = SQLQueryInput

    def _run(
        self,
        operation: str,
        user_id: str = "",
        risk_threshold: float = 7.0,
        limit: int = 5,
    ) -> str:
        try:
            return _run_async(
                _execute_sql_op(
                    operation.strip().lower(),
                    user_id.strip(),
                    float(risk_threshold),
                    min(max(int(limit), 1), 20),
                )
            )
        except Exception as e:
            return f"SQL query error: {e}"


async def _execute_sql_op(op: str, user_id: str, risk_threshold: float, limit: int) -> str:
    from core.database import Analysis, AsyncSessionLocal, Document

    async with AsyncSessionLocal() as session:
        if op == "recent_analyses":
            stmt = select(Analysis).order_by(desc(Analysis.created_at)).limit(limit)
            if user_id:
                stmt = (
                    select(Analysis)
                    .where(Analysis.user_id == user_id)
                    .order_by(desc(Analysis.created_at))
                    .limit(limit)
                )
            rows = (await session.execute(stmt)).scalars().all()
            if not rows:
                return "No prior analyses found."
            return json.dumps(
                [_summarise_analysis(r) for r in rows], default=str
            )

        if op == "high_risk_analyses":
            stmt = (
                select(Analysis)
                .where(Analysis.risk_score >= risk_threshold)
                .order_by(desc(Analysis.created_at))
                .limit(limit)
            )
            if user_id:
                stmt = (
                    select(Analysis)
                    .where(
                        Analysis.user_id == user_id,
                        Analysis.risk_score >= risk_threshold,
                    )
                    .order_by(desc(Analysis.created_at))
                    .limit(limit)
                )
            rows = (await session.execute(stmt)).scalars().all()
            if not rows:
                return f"No analyses with risk_score >= {risk_threshold}."
            return json.dumps(
                [_summarise_analysis(r) for r in rows], default=str
            )

        if op == "list_documents":
            stmt = select(Document).order_by(desc(Document.created_at)).limit(limit)
            if user_id:
                stmt = (
                    select(Document)
                    .where(Document.user_id == user_id)
                    .order_by(desc(Document.created_at))
                    .limit(limit)
                )
            rows = (await session.execute(stmt)).scalars().all()
            if not rows:
                return "No documents on file."
            return json.dumps(
                [
                    {
                        "id": r.id,
                        "filename": r.original_name or r.filename,
                        "status": r.status,
                        "created_at": str(r.created_at),
                    }
                    for r in rows
                ],
                default=str,
            )

        return (
            f"Unknown operation '{op}'. "
            "Valid: recent_analyses, high_risk_analyses, list_documents."
        )


def _summarise_analysis(row) -> dict:
    result = row.result if isinstance(row.result, dict) else {}
    return {
        "id": row.id,
        "query": (row.query or "")[:120],
        "risk_score": row.risk_score,
        "recommendation": result.get("recommendation"),
        "valuation": result.get("valuation"),
        "confidence_pct": result.get("confidence_pct"),
        "created_at": str(row.created_at),
    }
