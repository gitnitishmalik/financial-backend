"""
analysis_service.py  –  Multi-agent financial analysis using CrewAI + RAG

Architecture
------------
RAG pipeline (rag_service.py)
  ├── Chunks & embeds all uploaded documents on ingest
  └── Returns only the top-K relevant passages per query  →  saves ~70% tokens

CrewAI agents (sequential)
  ├── Agent 1: Senior Financial Analyst   (task1 → financial metrics)
  ├── Agent 2: Risk Assessment Specialist (task2 → risk report, uses task1 output)
  └── Agent 3: Investment Advisor         (task3 → JSON recommendation)
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool

from core.config import settings
from services.rag_service import RAGService


# ── Document text extraction ──────────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from PDF, CSV, TXT, XLSX files (max 15 000 chars)."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:15_000]
        if suffix in (".csv", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")[:15_000]
        if suffix in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows: List[str] = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join("" if c is None else str(c) for c in row))
            return "\n".join(rows)[:15_000]
    except Exception as e:
        return f"[Error reading file: {e}]"
    return ""


# ── Groq LLM factory ──────────────────────────────────────────────────────────

def get_groq_llm() -> LLM:
    """Return a crewai LLM pointing at Groq via litellm."""
    return LLM(
        model=f"groq/{settings.GROQ_MODEL}",
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=4096,
    )


# ── RAG-aware document tool ───────────────────────────────────────────────────

class RAGDocumentTool(BaseTool):
    """
    Tool given to the Analyst agent.
    Instead of dumping the full document, it retrieves only the most
    relevant passages for the current query — saving significant tokens.
    """

    name: str = "search_documents"
    description: str = (
        "Search the financial documents for information relevant to a specific question. "
        "Input: a plain-English question or keyword (e.g. 'revenue growth 2023'). "
        "Returns the most relevant passages from all uploaded documents."
    )

    # Pydantic field — set after construction via model_post_init or direct assignment
    rag: RAGService = None   # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        if self.rag is None:
            return "RAG service not initialised."
        result = self.rag.retrieve(query.strip())
        return result or "No relevant passages found."


# ── Analysis Service ──────────────────────────────────────────────────────────

class AnalysisService:

    def __init__(self) -> None:
        # One shared RAG service (re-ingested per request)
        self._rag = RAGService()

    # -------------------------------------------------------------------------
    # Public async entry-point
    # -------------------------------------------------------------------------
    async def analyze(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        """Run multi-agent analysis asynchronously (runs crew in a thread)."""
        return await asyncio.to_thread(self._run_crew, file_paths, query)

    # -------------------------------------------------------------------------
    # Synchronous crew execution
    # -------------------------------------------------------------------------
    def _run_crew(self, file_paths: List[str], query: str) -> Dict[str, Any]:

        # ── Step 1: Ingest documents into RAG ────────────────────────────────
        raw_texts = [extract_text_from_file(fp) for fp in file_paths]
        self._rag.ingest([t for t in raw_texts if t])

        # ── Step 2: Build a compact summary for agents that don't use the tool
        #    (risk assessor + advisor receive task context, not raw docs)
        #    We retrieve a broad overview so they have baseline awareness.
        overview_context = self._rag.retrieve(query, top_k=4)

        # ── LLM ──────────────────────────────────────────────────────────────
        llm = get_groq_llm()

        # ── RAG tool (only Analyst needs it) ─────────────────────────────────
        rag_tool = RAGDocumentTool(rag=self._rag)

        # ── Agent 1: Senior Financial Analyst ────────────────────────────────
        analyst = Agent(
            role="Senior Financial Analyst",
            goal=(
                "Extract and interpret all key financial metrics, trends, and signals "
                "from the provided documents to answer the user's query precisely."
            ),
            backstory=(
                "CFA charterholder with 15 years at top investment banks. "
                "You excel at reading financial statements, spotting revenue trends, "
                "margin compression, debt levels, and earnings quality. "
                "You ALWAYS use the search_documents tool to fetch evidence before drawing conclusions."
            ),
            llm=llm,
            tools=[rag_tool],
            max_iter=5,
            verbose=True,
            allow_delegation=False,
        )

        # ── Agent 2: Risk Assessment Specialist ──────────────────────────────
        risk_assessor = Agent(
            role="Risk Assessment Specialist",
            goal=(
                "Identify all material risks — market, credit, liquidity, regulatory, "
                "operational — and assign a numeric risk score 1 (very low) to 10 (critical)."
            ),
            backstory=(
                "Former Chief Risk Officer at Goldman Sachs, 20 years experience. "
                "You identify risks others miss and quantify their severity precisely. "
                "You build on the Analyst's findings and use the provided document context."
            ),
            llm=llm,
            tools=[],           # uses task context, not the tool (saves tokens)
            max_iter=3,
            verbose=True,
            allow_delegation=False,
        )

        # ── Agent 3: Investment Advisor ───────────────────────────────────────
        advisor = Agent(
            role="Investment Advisor",
            goal=(
                "Synthesise the analyst's findings and risk assessment into a clear, "
                "actionable investment recommendation backed by specific data."
            ),
            backstory=(
                "Manages a $500M portfolio with 18% annualised returns. "
                "Gives clear BUY / HOLD / SELL recommendations backed by evidence. "
                "Always outputs strictly valid JSON — no markdown, no preamble."
            ),
            llm=llm,
            tools=[],
            max_iter=3,
            verbose=True,
            allow_delegation=False,
        )

        # ── Task 1: Financial Analysis ────────────────────────────────────────
        task1 = Task(
            description=(
                f"User query: {query}\n\n"
                f"Relevant document context (retrieved via RAG):\n{overview_context}\n\n"
                "Use the `search_documents` tool to fetch additional details as needed.\n\n"
                "Analyse the documents thoroughly. Extract:\n"
                "• Revenue, profit, EPS, EBITDA figures and YoY trends\n"
                "• Gross margin, operating margin, net margin\n"
                "• Debt-to-equity, current ratio, quick ratio\n"
                "• Year-over-year growth rates\n"
                "• Any notable changes, anomalies, or red flags\n\n"
                "Be specific — cite exact numbers."
            ),
            agent=analyst,
            expected_output=(
                "A detailed financial metrics report with specific numbers, percentages, "
                "and year-over-year comparisons extracted from the documents."
            ),
        )

        # ── Task 2: Risk Assessment ───────────────────────────────────────────
        task2 = Task(
            description=(
                "Using the Analyst's financial findings above, produce a comprehensive risk assessment.\n"
                "• List 3–5 specific risks with severity: LOW / MEDIUM / HIGH / CRITICAL\n"
                "• Assign an overall risk score: integer 1 (very safe) to 10 (very risky)\n"
                "• Explain each risk with evidence from the documents\n\n"
                "Be concise but precise."
            ),
            agent=risk_assessor,
            expected_output=(
                "A risk report listing specific risks with severity labels and an overall "
                "numeric risk score from 1–10."
            ),
            context=[task1],
        )

        # ── Task 3: Investment Recommendation (JSON) ──────────────────────────
        task3 = Task(
            description=(
                "Synthesise the financial analysis and risk assessment.\n"
                "Output ONLY a single valid JSON object with exactly these keys:\n\n"
                "{\n"
                '  "recommendation": "BUY" | "HOLD" | "SELL",\n'
                '  "confidence_pct": <integer 0-100>,\n'
                '  "price_target": <number or null>,\n'
                '  "risk_score": <number 1-10>,\n'
                '  "reasons": ["reason1", "reason2", "reason3"],\n'
                '  "risks": ["risk1", "risk2"],\n'
                '  "metrics": {"revenue": "...", "margin": "...", "growth": "..."},\n'
                '  "summary": "<2-3 sentence plain-English summary>"\n'
                "}\n\n"
                "IMPORTANT: Output ONLY the JSON. No markdown fences, no explanation."
            ),
            agent=advisor,
            expected_output="A single valid JSON object with the investment recommendation.",
            context=[task1, task2],
        )

        # ── Assemble & run crew ───────────────────────────────────────────────
        crew = Crew(
            agents=[analyst, risk_assessor, advisor],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=True,
        )

        try:
            raw_result = str(crew.kickoff())
        except Exception as exc:
            return self._error_response(str(exc))

        return self._parse_result(raw_result)

    # -------------------------------------------------------------------------
    # Result parsing helpers
    # -------------------------------------------------------------------------

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        """Extract JSON from crew output with graceful fallback."""
        # 1. Direct parse
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
            return json.loads(clean)
        except Exception:
            pass

        # 2. Find JSON object inside prose
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # 3. Best-effort fallback
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
            "reasons": ["Analysis completed — see summary for details"],
            "risks": ["Review full analysis for risk factors"],
            "metrics": {},
            "summary": raw[:800],
        }

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "recommendation": "HOLD",
            "confidence_pct": 0,
            "price_target": None,
            "risk_score": 5.0,
            "reasons": ["Analysis could not be completed"],
            "risks": [f"Error: {error[:200]}"],
            "metrics": {},
            "summary": f"Analysis failed: {error[:300]}",
        }