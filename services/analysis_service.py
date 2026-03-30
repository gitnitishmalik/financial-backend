"""
analysis_service.py  –  5-Agent high-quality financial analysis pipeline

Agents (sequential):
  1. Document Verifier    – validates doc is financial, extracts structure
  2. Financial Analyst    – extracts exact metrics, flags trends
  3. Valuation Analyst    – DCF, P/E, EV/EBITDA, sector comparables
  4. Risk Specialist      – data-backed risks with severity scores
  5. Investment Advisor   – decisive JSON recommendation

Token budget (~12k TPM safe):
  Each agent: ~800 in / ~400 out = ~1200 per agent × 5 = ~6000 total
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
from crewai.tools import BaseTool

from core.config import settings
from services.rag_service import RAGService


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


# ── RAG Tool ─────────────────────────────────────────────────────────────────

class RAGDocumentTool(BaseTool):
    name: str = "search_documents"
    description: str = (
        "Search financial documents for specific data. "
        "Input: short keyword (e.g. 'revenue 2023', 'net income', 'P/E ratio'). "
        "Returns the most relevant passages from uploaded documents."
    )
    rag: RAGService = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        if self.rag is None:
            return "RAG not initialised."
        return self.rag.retrieve(query.strip(), top_k=2) or "No data found."


# ── Analysis Service ──────────────────────────────────────────────────────────

class AnalysisService:

    def __init__(self) -> None:
        self._rag = RAGService()

    async def analyze(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._run_crew, file_paths, query)

    def _run_crew(self, file_paths: List[str], query: str) -> Dict[str, Any]:

        # ── RAG ingest ────────────────────────────────────────────────────────
        raw_texts = [extract_text_from_file(fp) for fp in file_paths]
        self._rag.ingest([t for t in raw_texts if t])
        context = self._rag.retrieve(query, top_k=2)   # ~800 words

        llm      = get_groq_llm()
        rag_tool = RAGDocumentTool(rag=self._rag)

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
                "You never use hedging words like 'may', 'could', 'might', 'potentially'. "
                "If a metric is missing from the document, you say 'NOT REPORTED' — never guess. "
                "You focus on: revenue, gross/operating/net margins, EPS, FCF, debt levels, "
                "working capital, and any sector-specific KPIs present in the document."
            ),
            llm=llm,
            tools=[rag_tool],
            max_iter=2,
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
            tools=[rag_tool],
            max_iter=1,
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
                "Every risk you identify must reference a specific metric from the documents. "
                "You categorise risks across four dimensions:\n"
                "  FINANCIAL: leverage, liquidity, margin compression\n"
                "  OPERATIONAL: volume, capacity, supply chain signals\n"
                "  MARKET: demand trends, competitive position, pricing power\n"
                "  REGULATORY: compliance, legal, policy exposure\n"
                "Each risk gets a severity: LOW / MEDIUM / HIGH / CRITICAL\n"
                "BAD example: 'Market conditions may affect revenue'\n"
                "GOOD example: 'Operating margin compressed 580bps YoY to 4.1% — "
                "if this trend continues one more quarter, FCF turns negative — HIGH'\n"
                "You give an overall portfolio risk score 1-10."
            ),
            llm=llm,
            tools=[],
            max_iter=1,
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
                "1. P/E ratio — compare to sector average (state the benchmark used)\n"
                "2. EV/EBITDA — compare to sector average\n"
                "3. FCF yield or Price/FCF if FCF is available\n"
                "4. PEG ratio if EPS growth is reported\n"
                "5. Overall verdict: OVERVALUED / FAIRLY VALUED / UNDERVALUED\n\n"
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
                '  "catalyst": "<specific event to watch or N/A>",\n'
                '  "summary": "<2 direct sentences: key deciding factor + action>"\n'
                "}"
            ),
            agent=advisor,
            expected_output="Valid JSON only. No markdown. No prose outside the JSON.",
            context=[task_verify, task_analyze, task_value, task_risk],
        )

        # ── Assemble crew ─────────────────────────────────────────────────────
        crew = Crew(
            agents=[verifier, analyst, valuation_analyst, risk_specialist, advisor],
            tasks=[task_verify, task_analyze, task_value, task_risk, task_recommend],
            process=Process.sequential,
            verbose=False,
        )

        raw_result = self._kickoff_with_retry(crew)
        return self._parse_result(raw_result)

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
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
            return json.loads(clean)
        except Exception:
            pass
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return self._fallback_parse(raw)

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
            "catalyst": "N/A",
            "summary": f"Analysis failed: {error[:300]}",
        }