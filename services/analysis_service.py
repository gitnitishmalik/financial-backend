"""
analysis_service.py  –  5-Agent high-quality financial analysis pipeline

Agents (sequential):
  1. Document Verifier    – validates doc is financial, extracts structure
  2. Financial Analyst    – extracts exact metrics, flags trends (+calculator)
  3. Valuation Analyst    – P/E, EV/EBITDA, comparables (+live market data)
  4. Risk Specialist      – data-backed risks (+market history + news)
  5. Investment Advisor   – decisive JSON recommendation (synthesis only)

Tool access by agent — see services/tools.py for definitions.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from crewai import Agent, Crew, LLM, Process, Task

from core.config import settings
from services.market_service import MarketService
from services.rag_service import RAGService
from services.tools import (
    CalculatorTool,
    MarketHistoryTool,
    MarketNewsTool,
    MarketQuoteTool,
    PeerCompTool,
    RAGDocumentTool,
    SQLQueryTool,
    WebSearchTool,
)


# ── Document text extraction ──────────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:8]:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:8000]
        if suffix in (".csv", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")[:8000]
        if suffix in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows: List[str] = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join("" if c is None else str(c) for c in row))
            return "\n".join(rows)[:8000]
    except Exception as e:
        return f"[Error reading file: {e}]"
    return ""


# ── Groq LLM ─────────────────────────────────────────────────────────────────

def get_groq_llm() -> LLM:
    return LLM(
        model=f"groq/{settings.GROQ_MODEL}",
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=500,     # tight cap per agent to stay under TPM
    )


# ── chart_data normaliser ────────────────────────────────────────────────────

def _coerce_number(v: Any) -> float | None:
    """Pull a float out of '1,234', '$1.2B', '12.5%', etc. Returns None on failure."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.upper() in ("N/A", "NOT REPORTED", "NA", "NULL"):
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    mult = 1.0
    if s and s[-1] in "BbTtMmKk":
        suffix = s[-1].lower()
        s = s[:-1].strip()
        mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[suffix]
    try:
        return float(s) * mult
    except Exception:
        return None


_QUARTER_RE = re.compile(r"Q([1-4])\s*[/\- ]?\s*(\d{2,4})", re.IGNORECASE)


def _period_sort_key(period: str) -> tuple:
    """Sort 'Q1 2023', 'FY 2024', '2023' chronologically. Unknown → end."""
    if not isinstance(period, str):
        return (9999, 9)
    m = _QUARTER_RE.search(period)
    if m:
        y = int(m.group(2))
        if y < 100:
            y += 2000
        return (y, int(m.group(1)))
    m = re.search(r"(\d{4})", period)
    if m:
        return (int(m.group(1)), 0)
    return (9999, 9)


def _clean_chart_data(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        period = r.get("period") or r.get("date") or r.get("label")
        if not period:
            continue
        revenue = _coerce_number(r.get("revenue"))
        net_income = _coerce_number(r.get("net_income") or r.get("netIncome") or r.get("net_profit"))
        op_margin = _coerce_number(r.get("operating_margin") or r.get("op_margin"))
        fcf = _coerce_number(r.get("fcf") or r.get("free_cash_flow"))
        # Need at least one numeric to be useful
        if all(v is None for v in (revenue, net_income, op_margin, fcf)):
            continue
        out.append({
            "period": str(period).strip(),
            "revenue": revenue,
            "net_income": net_income,
            "operating_margin": op_margin,
            "fcf": fcf,
        })
    out.sort(key=lambda x: _period_sort_key(x["period"]))
    return out[:20]  # cap at 20 periods so the chart stays readable


# ── Analysis Service ──────────────────────────────────────────────────────────

class AnalysisService:

    def __init__(self) -> None:
        self._rag = RAGService()
        self._market = MarketService()

    async def analyze(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._run_crew, file_paths, query)

    def _run_crew(self, file_paths: List[str], query: str) -> Dict[str, Any]:

        # ── RAG ingest ────────────────────────────────────────────────────────
        raw_texts = [extract_text_from_file(fp) for fp in file_paths]
        self._rag.ingest([t for t in raw_texts if t])
        context = self._rag.retrieve(query, top_k=2)   # ~800 words

        llm           = get_groq_llm()
        rag_tool      = RAGDocumentTool(rag=self._rag)
        calc_tool     = CalculatorTool()
        quote_tool    = MarketQuoteTool(market=self._market)
        history_tool  = MarketHistoryTool(market=self._market)
        news_tool     = MarketNewsTool(market=self._market)
        peer_tool     = PeerCompTool(market=self._market)
        web_tool      = WebSearchTool()
        sql_tool      = SQLQueryTool()

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 1 — Document Verifier
        # Confirms this is a real financial document and maps its structure
        # ══════════════════════════════════════════════════════════════════════
        verifier = Agent(
            role="Financial Document Verifier",
            goal=(
                "Confirm the document is a legitimate financial report and "
                "identify what data is available for analysis."
            ),
            backstory=(
                "You are a Big Four audit senior with 15 years verifying financial statements. "
                "You instantly identify document type (10-K, 10-Q, earnings release, balance sheet). "
                "You confirm which financial statements are present: "
                "income statement, balance sheet, cash flow statement. "
                "You flag missing data that will limit analysis quality. "
                "You are precise and factual — you never assume data exists if it's not there."
            ),
            llm=llm,
            tools=[rag_tool],
            max_iter=1,
            verbose=False,
            allow_delegation=False,
        )

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 2 — Senior Financial Analyst
        # Extracts all hard metrics and labels each IMPROVING or DETERIORATING
        # ══════════════════════════════════════════════════════════════════════
        analyst = Agent(
            role="Senior Financial Analyst",
            goal="Extract every financial metric with exact numbers and label each trend.",
            backstory=(
                "You are a CFA charterholder with 20 years at Goldman Sachs equity research. "
                "You extract EXACT figures — never approximate or estimate. "
                "For every metric you state: the number, the YoY change, and IMPROVING or DETERIORATING. "
                "You ALWAYS use the calculator tool for percentage and ratio computations — never compute in your head. "
                "You never use hedging words like 'may', 'could', 'might', 'potentially'. "
                "If a metric is missing from the document, you say 'NOT REPORTED' — never guess. "
                "You focus on: revenue, gross/operating/net margins, EPS, FCF, debt levels, "
                "working capital, and any sector-specific KPIs present in the document."
            ),
            llm=llm,
            tools=[rag_tool, calc_tool, sql_tool],
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 3 — Valuation Analyst
        # Applies valuation frameworks: P/E, EV/EBITDA, DCF signals, comparables
        # ══════════════════════════════════════════════════════════════════════
        valuation_analyst = Agent(
            role="Valuation Analyst",
            goal="Determine if the stock is overvalued, fairly valued, or undervalued using hard data.",
            backstory=(
                "You are a valuation specialist from Morgan Stanley's M&A division. "
                "You ALWAYS pull the LIVE market price using market_quote before computing ratios — "
                "document figures from past quarters are stale. "
                "You use market_history to assess price trend and momentum. "
                "You use peer_comparison to ground sector multiples in real peer data instead of guessing benchmarks. "
                "You use the calculator tool for every P/E, EV/EBITDA, PEG, and FCF-yield computation. "
                "You apply standard valuation frameworks to every analysis:\n"
                "- P/E ratio vs sector average (state the sector benchmark)\n"
                "- EV/EBITDA vs sector average\n"
                "- Price-to-FCF if FCF is reported\n"
                "- DCF signal: is FCF growing or shrinking? Does that justify current price?\n"
                "- Revenue multiple vs growth rate (PEG ratio if possible)\n"
                "If valuation data is not in the document, you state which metrics are missing "
                "and what that means for the valuation confidence. "
                "You give a clear verdict: OVERVALUED / FAIRLY VALUED / UNDERVALUED with a reason."
            ),
            llm=llm,
            tools=[rag_tool, quote_tool, history_tool, peer_tool, calc_tool, sql_tool],
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 4 — Risk Specialist
        # Identifies specific, data-backed risks — no generic statements allowed
        # ══════════════════════════════════════════════════════════════════════
        risk_specialist = Agent(
            role="Risk Assessment Specialist",
            goal="Identify the top 4 material risks, each backed by a specific number.",
            backstory=(
                "You are a former Chief Risk Officer at JP Morgan with 25 years experience. "
                "Every risk you identify must reference a specific metric from the documents OR live market data. "
                "You use market_news to check for recent material events that signal risk. "
                "You use web_search to pull recent SEC filings (10-K/10-Q/8-K) for regulatory and disclosure risk. "
                "You use market_history to assess price volatility and momentum trends. "
                "You use sql_query (high_risk_analyses) to learn from prior high-risk calls in the database. "
                "You use the calculator for ratio-based risk thresholds (D/E, current ratio, etc.). "
                "You categorise risks across four dimensions:\n"
                "  FINANCIAL: leverage, liquidity, margin compression\n"
                "  OPERATIONAL: volume, capacity, supply chain signals\n"
                "  MARKET: demand trends, competitive position, pricing power, volatility\n"
                "  REGULATORY: compliance, legal, policy exposure\n"
                "Each risk gets a severity: LOW / MEDIUM / HIGH / CRITICAL\n"
                "BAD example: 'Market conditions may affect revenue'\n"
                "GOOD example: 'Operating margin compressed 580bps YoY to 4.1% — "
                "if this trend continues one more quarter, FCF turns negative — HIGH'\n"
                "You give an overall portfolio risk score 1-10."
            ),
            llm=llm,
            tools=[rag_tool, quote_tool, history_tool, news_tool, web_tool, calc_tool, sql_tool],
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 5 — Investment Advisor
        # Synthesises all findings into a single decisive JSON recommendation
        # ══════════════════════════════════════════════════════════════════════
        advisor = Agent(
            role="Portfolio Manager & Investment Advisor",
            goal="Synthesise all findings into one decisive, data-backed investment call.",
            backstory=(
                "You manage a $2B long/short equity fund with a 12-year track record. "
                "You make DECISIVE calls with clear conviction levels:\n"
                "  BUY:  fundamentals improving AND valuation is attractive\n"
                "  SELL: fundamentals deteriorating OR significantly overvalued\n"
                "  HOLD: mixed signals with a specific catalyst to watch (you name it)\n"
                "Your confidence_pct is determined by data quality and signal strength:\n"
                "  90-100: all metrics point same direction, strong data\n"
                "  70-89:  majority of metrics align, minor conflicts\n"
                "  50-69:  mixed signals, catalyst-dependent\n"
                "  Below 50: insufficient data — you say so explicitly\n"
                "Every reason must start with a specific number from the documents. "
                "Output ONLY valid JSON — no markdown, no prose, no explanation."
            ),
            llm=llm,
            tools=[],
            max_iter=1,
            verbose=False,
            allow_delegation=False,
        )

        # ══════════════════════════════════════════════════════════════════════
        # TASKS
        # ══════════════════════════════════════════════════════════════════════

        task_verify = Task(
            description=(
                f"Document context:\n{context[:800]}\n\n"
                "Identify:\n"
                "1. Document type (10-K / 10-Q / earnings release / other)\n"
                "2. Company name and reporting period\n"
                "3. Which financial statements are present\n"
                "4. Any obvious data gaps that will limit analysis\n"
                "Max 100 words. Be factual."
            ),
            agent=verifier,
            expected_output=(
                "Document type, company, period, available statements, data gaps. Under 100 words."
            ),
        )

        task_analyze = Task(
            description=(
                f"User question: {query[:150]}\n\n"
                f"Document data:\n{context[:1200]}\n\n"
                "Extract EXACT numbers for every available metric:\n"
                "• Revenue: absolute figure + YoY %\n"
                "• Gross margin, Operating margin, Net margin\n"
                "• EPS (basic + diluted)\n"
                "• Free cash flow\n"
                "• Total debt, D/E ratio, current ratio\n"
                "• Sector KPIs (units, deliveries, subscribers, etc.)\n\n"
                "Use the calculator tool for EVERY percentage change and ratio — never compute in your head.\n"
                "Use search_documents to find numbers not visible in the context above.\n"
                "Label each: IMPROVING ↑ or DETERIORATING ↓ with the % change.\n"
                "If not reported: say NOT REPORTED.\n"
                "No hedging. Max 250 words."
            ),
            agent=analyst,
            expected_output=(
                "Exact metrics with IMPROVING/DETERIORATING labels and % changes. "
                "NOT REPORTED for missing data. Under 250 words."
            ),
            context=[task_verify],
        )

        task_value = Task(
            description=(
                "Using the analyst's metrics, apply valuation frameworks:\n"
                "FIRST: identify the ticker mentioned in the document and call market_quote to get the LIVE price.\n"
                "THEN: call market_history with period='1y' to assess price trend.\n"
                "THEN: use the calculator tool for every ratio computation:\n"
                "1. P/E ratio (live_price / EPS) — compare to sector average (state the benchmark used)\n"
                "2. EV/EBITDA — compare to sector average\n"
                "3. FCF yield or Price/FCF if FCF is available\n"
                "4. PEG ratio if EPS growth is reported\n"
                "5. Overall verdict: OVERVALUED / FAIRLY VALUED / UNDERVALUED\n\n"
                "If no ticker is identifiable, state that and base verdict on document-only data.\n"
                "If a metric is missing, state it and explain impact on confidence.\n"
                "Max 150 words."
            ),
            agent=valuation_analyst,
            expected_output=(
                "Valuation multiples with sector benchmarks. "
                "Clear OVERVALUED/FAIRLY VALUED/UNDERVALUED verdict. Under 150 words."
            ),
            context=[task_analyze],
        )

        task_risk = Task(
            description=(
                "Identify exactly 4 material risks using the analyst and valuation findings.\n\n"
                "If a ticker is identifiable, call market_news to check for recent risk signals "
                "and market_history (period='3mo') to assess volatility.\n"
                "Use the calculator for ratio-based thresholds (D/E, current ratio, interest coverage).\n\n"
                "For each risk:\n"
                "• Category: FINANCIAL / OPERATIONAL / MARKET / REGULATORY\n"
                "• Specific metric that signals this risk (exact number required)\n"
                "• Consequence: what happens if this risk materialises\n"
                "• Severity: LOW / MEDIUM / HIGH / CRITICAL\n\n"
                "End with: Overall Risk Score: X/10\n"
                "Max 200 words. No generic statements."
            ),
            agent=risk_specialist,
            expected_output=(
                "4 categorised risks with exact metrics, consequences, severity. "
                "Overall risk score X/10. Under 200 words."
            ),
            context=[task_analyze, task_value],
        )

        task_recommend = Task(
            description=(
                "Synthesise the verification, analysis, valuation, and risk assessment.\n\n"
                "Decision rules:\n"
                "• BUY:  metrics improving + valuation attractive + risk score ≤ 6\n"
                "• SELL: metrics deteriorating + overvalued OR risk score ≥ 8\n"
                "• HOLD: mixed signals — name the specific catalyst to watch\n\n"
                "confidence_pct rules:\n"
                "• 90-100: all signals aligned, strong data quality\n"
                "• 70-89:  majority aligned, minor conflicts\n"
                "• 50-69:  mixed, catalyst-dependent\n"
                "• <50:    weak data — say so in summary\n\n"
                "Output ONLY this JSON object:\n"
                "{\n"
                '  "recommendation": "BUY" or "HOLD" or "SELL",\n'
                '  "confidence_pct": <integer 0-100>,\n'
                '  "price_target": <number or null>,\n'
                '  "risk_score": <number 1-10>,\n'
                '  "valuation": "OVERVALUED" or "FAIRLY VALUED" or "UNDERVALUED",\n'
                '  "reasons": ["<number> + reason", "<number> + reason", "<number> + reason"],\n'
                '  "risks": ["<category>: <specific risk with number>", "..."],\n'
                '  "metrics": {\n'
                '    "revenue": "<exact figure>",\n'
                '    "revenue_growth": "<YoY %>",\n'
                '    "operating_margin": "<exact %>",\n'
                '    "net_margin": "<exact %>",\n'
                '    "eps": "<exact figure>",\n'
                '    "fcf": "<exact figure or NOT REPORTED>"\n'
                '  },\n'
                '  "chart_data": [\n'
                '    /* Extract EVERY reporting period the document shows — quarters or years.\n'
                '       Use search_documents to find the financial-highlights table.\n'
                '       Numbers MUST be in the same unit (e.g. millions). Drop entries with no data.\n'
                '       Each entry: */\n'
                '    {"period": "Q1 2023", "revenue": <number>, "net_income": <number>,\n'
                '     "operating_margin": <number>, "fcf": <number or null>}\n'
                '  ],\n'
                '  "catalyst": "<specific event to watch or N/A>",\n'
                '  "summary": "<2 direct sentences: key deciding factor + action>"\n'
                "}"
            ),
            agent=advisor,
            expected_output="Valid JSON only. No markdown. No prose outside the JSON.",
            context=[task_verify, task_analyze, task_value, task_risk],
        )

        # ── Planner: pick which agents are actually relevant ──────────────────
        plan = self._plan(query)
        agents_active = {
            "verifier":  True,                      # always confirm the doc
            "analyst":   "analyst"   in plan,
            "valuation": "valuation" in plan,
            "risk":      "risk"      in plan,
            "advisor":   True,                      # always produce JSON
        }
        # Drop the dependency edges of skipped tasks so CrewAI doesn't choke
        analyze_ctx = [task_verify]
        value_ctx   = [task_analyze] if agents_active["analyst"] else [task_verify]
        risk_ctx    = ([task_analyze] if agents_active["analyst"] else []) \
                    + ([task_value]    if agents_active["valuation"] else [])
        if not risk_ctx:
            risk_ctx = [task_verify]
        recommend_ctx = [task_verify]
        if agents_active["analyst"]:   recommend_ctx.append(task_analyze)
        if agents_active["valuation"]: recommend_ctx.append(task_value)
        if agents_active["risk"]:      recommend_ctx.append(task_risk)

        # Rebind contexts on the existing Task objects
        task_analyze.context = analyze_ctx
        task_value.context   = value_ctx
        task_risk.context    = risk_ctx
        task_recommend.context = recommend_ctx

        active_agents: list[Agent] = [verifier]
        active_tasks:  list[Task]  = [task_verify]
        if agents_active["analyst"]:
            active_agents.append(analyst);            active_tasks.append(task_analyze)
        if agents_active["valuation"]:
            active_agents.append(valuation_analyst);  active_tasks.append(task_value)
        if agents_active["risk"]:
            active_agents.append(risk_specialist);    active_tasks.append(task_risk)
        active_agents.append(advisor)
        active_tasks.append(task_recommend)

        # ── Assemble crew ─────────────────────────────────────────────────────
        crew = Crew(
            agents=active_agents,
            tasks=active_tasks,
            process=Process.sequential,
            verbose=False,
        )

        raw_result = self._kickoff_with_retry(crew)
        result = self._parse_result(raw_result)

        # ── Critic: validate JSON internal consistency, retry once if needed ─
        result = self._critique_and_refine(result, query)
        result["plan"] = sorted([k for k, v in agents_active.items() if v])
        return result

    # ── Planner ────────────────────────────────────────────────────────────────

    def _plan(self, query: str) -> list[str]:
        """Ask Groq which heavy agents this query actually needs. Returns a
        list of agent keys: 'analyst', 'valuation', 'risk'. Falls back to
        running all three on any error."""
        all_agents = ["analyst", "valuation", "risk"]
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are a planner for a financial-analysis crew. Given a "
                        "user query, decide which of these heavy agents are needed:\n"
                        "  - analyst  : extracts numeric metrics from documents\n"
                        "  - valuation: pulls live price/peers, computes multiples\n"
                        "  - risk     : assesses risks, runs news/EDGAR lookups\n"
                        "Output ONLY a JSON object: {\"agents\": [\"analyst\", ...]} "
                        "Verifier and advisor always run, don't list them."
                    )},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=80,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            picked = [a for a in data.get("agents", []) if a in all_agents]
            return picked or all_agents
        except Exception:
            return all_agents

    # ── Critic ─────────────────────────────────────────────────────────────────

    def _critic_check(self, result: Dict[str, Any]) -> list[str]:
        """Return a list of contradictions in the advisor's JSON. Empty = OK."""
        issues: list[str] = []
        rec = (result.get("recommendation") or "").upper()
        val = (result.get("valuation") or "").upper()
        try:
            risk = float(result.get("risk_score") or 0)
        except Exception:
            risk = 0.0
        try:
            conf = int(result.get("confidence_pct") or 0)
        except Exception:
            conf = 0

        if rec == "BUY" and risk >= 8:
            issues.append(f"BUY recommendation with risk_score={risk} (≥8) — high-risk BUY needs a stated catalyst justifying it.")
        if rec == "BUY" and val == "OVERVALUED":
            issues.append("BUY recommendation while valuation=OVERVALUED — these contradict.")
        if rec == "SELL" and val == "UNDERVALUED":
            issues.append("SELL recommendation while valuation=UNDERVALUED — these contradict.")
        if rec == "SELL" and risk <= 3:
            issues.append(f"SELL recommendation with risk_score={risk} (≤3) — low-risk SELL is unusual without a thesis.")
        if conf >= 85 and not (result.get("reasons") and len(result["reasons"]) >= 3):
            issues.append(f"confidence_pct={conf} is high but reasons[] has <3 entries.")
        return issues

    def _critique_and_refine(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """If the critic finds contradictions, re-ask the advisor with the
        critique inline. Single retry. Never raises."""
        issues = self._critic_check(result)
        if not issues:
            result.setdefault("critic_status", "ok")
            return result

        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are a critic for a financial-analysis advisor. The "
                        "user asks: " + query + "\n"
                        "The advisor produced JSON that has internal contradictions. "
                        "Resolve them and return ONLY the corrected JSON (same shape). "
                        "Keep all metrics. Adjust ONLY the recommendation / valuation / "
                        "confidence_pct / risk_score / summary / reasons to be self-consistent."
                    )},
                    {"role": "user", "content": (
                        "Issues found:\n- " + "\n- ".join(issues)
                        + "\n\nOriginal advisor JSON:\n" + json.dumps(result, default=str)
                    )},
                ],
                response_format={"type": "json_object"},
                max_tokens=1200,
                temperature=0.1,
            )
            refined_raw = resp.choices[0].message.content or "{}"
            refined = json.loads(refined_raw)
            # Sanity: only accept the refinement if it kept the shape
            if isinstance(refined, dict) and "recommendation" in refined:
                refined["critic_status"] = "refined"
                refined["critic_issues"] = issues
                return refined
        except Exception:
            pass

        result["critic_status"] = "issues_flagged_but_not_refined"
        result["critic_issues"] = issues
        return result

    # ── Retry on rate limit ───────────────────────────────────────────────────

    def _kickoff_with_retry(self, crew: Crew, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return str(crew.kickoff())
            except Exception as e:
                err = str(e)
                if "RateLimitError" in err and attempt < max_retries - 1:
                    wait = self._parse_wait_time(err) or (30 * (attempt + 1))
                    print(f"Rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    return json.dumps(self._error_response(err))
        return json.dumps(self._error_response("Max retries exceeded"))

    def _parse_wait_time(self, error_msg: str) -> int | None:
        match = re.search(r"try again in (\d+\.?\d*)s", error_msg, re.IGNORECASE)
        return int(float(match.group(1))) + 2 if match else None

    # ── Result parsing ────────────────────────────────────────────────────────

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] | None = None
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
            parsed = json.loads(clean)
        except Exception:
            try:
                match = re.search(r"\{[\s\S]*\}", raw)
                if match:
                    parsed = json.loads(match.group())
            except Exception:
                parsed = None

        if parsed is None:
            return self._fallback_parse(raw)

        # Normalise chart_data — sanitize numbers, drop bad rows, sort if possible.
        parsed["chart_data"] = _clean_chart_data(parsed.get("chart_data"))
        return parsed

    def _fallback_parse(self, raw: str) -> Dict[str, Any]:
        rec = "HOLD"
        for word in ("BUY", "SELL", "HOLD"):
            if word in raw.upper():
                rec = word
                break
        risk_score = 5.0
        m = re.search(r"risk[^\d]*(\d+\.?\d*)", raw, re.IGNORECASE)
        if m:
            risk_score = min(10.0, max(1.0, float(m.group(1))))
        return {
            "recommendation": rec,
            "confidence_pct": 65,
            "price_target": None,
            "risk_score": risk_score,
            "valuation": "UNKNOWN",
            "reasons": ["See summary for details"],
            "risks": ["Review full analysis for risk factors"],
            "metrics": {},
            "chart_data": [],
            "catalyst": "N/A",
            "summary": raw[:500],
        }

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "recommendation": "HOLD",
            "confidence_pct": 0,
            "price_target": None,
            "risk_score": 5.0,
            "valuation": "UNKNOWN",
            "reasons": ["Analysis could not be completed"],
            "risks": [f"Error: {error[:200]}"],
            "metrics": {},
            "chart_data": [],
            "catalyst": "N/A",
            "summary": f"Analysis failed: {error[:300]}",
        }