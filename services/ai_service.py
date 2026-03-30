import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import re

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import pdfplumber

from core.config import settings


# ── Document text extraction ──────────────────────────────────────

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from PDF, CSV, or TXT files."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        if path.suffix.lower() == ".pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:15000]
        elif path.suffix.lower() in (".csv", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")[:15000]
        elif path.suffix.lower() in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)[:15000]
    except Exception as e:
        return f"[Error reading file: {e}]"
    return ""


# ── Groq LLM factory ──────────────────────────────────────────────
# crewai 1.x uses its own LLM class backed by litellm.
# The model string must be prefixed with "groq/" so litellm
# routes the call to the Groq API automatically.

def get_groq_llm() -> LLM:
    """Return a crewai LLM instance pointing at Groq."""
    return LLM(
        model=f"groq/{settings.GROQ_MODEL}",   # e.g. groq/llama-3.3-70b-versatile
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=4096,
    )


# ── CrewAI document read tool ─────────────────────────────────────

class ReadDocumentTool(BaseTool):
    name: str = "read_document"
    description: str = (
        "Read and extract the full text content from a financial document. "
        "Input should be the file path string."
    )

    def _run(self, file_path: str) -> str:
        content = extract_text_from_file(file_path.strip())
        if not content:
            return "No content could be extracted from this file."
        return content


# ── Main analysis service ─────────────────────────────────────────

class AnalysisService:

    def __init__(self):
        self.read_tool = ReadDocumentTool()

    async def analyze(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        """Run multi-agent analysis asynchronously."""
        return await asyncio.to_thread(self._run_crew, file_paths, query)

    def _run_crew(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        llm = get_groq_llm()

        # Pre-extract all document text so agents share the same context
        docs_text = []
        for fp in file_paths:
            text = extract_text_from_file(fp)
            if text:
                docs_text.append(f"[File: {Path(fp).name}]\n{text}")
        combined_docs = "\n\n---\n\n".join(docs_text)[:8000]

        # ── Agent 1: Senior Financial Analyst ─────────────────────
        analyst = Agent(
            role="Senior Financial Analyst",
            goal=(
                "Extract and interpret all key financial metrics, trends, and signals "
                "from the provided documents to answer the user's query."
            ),
            backstory=(
                "You are a CFA charterholder with 15 years at top investment banks. "
                "You excel at reading financial statements, spotting revenue trends, "
                "margin compression, debt levels, and earnings quality."
            ),
            llm=llm,
            tools=[self.read_tool],
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ── Agent 2: Risk Assessment Specialist ───────────────────
        risk_assessor = Agent(
            role="Risk Assessment Specialist",
            goal=(
                "Identify all material risks — market, credit, liquidity, regulatory, "
                "operational — and assign a numeric risk score from 1 (very low) to 10 (critical)."
            ),
            backstory=(
                "Former Chief Risk Officer at Goldman Sachs with 20 years experience. "
                "You identify risks others miss and quantify their severity precisely."
            ),
            llm=llm,
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ── Agent 3: Investment Advisor ───────────────────────────
        advisor = Agent(
            role="Investment Advisor",
            goal=(
                "Synthesize the analyst's findings and risk assessment into a clear, "
                "actionable investment recommendation with supporting evidence."
            ),
            backstory=(
                "You manage a $500M portfolio and have a track record of 18% annualized returns. "
                "You give clear BUY / HOLD / SELL recommendations backed by data."
            ),
            llm=llm,
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        # ── Task 1: Financial Analysis ────────────────────────────
        task1 = Task(
            description=(
                f"User query: {query}\n\n"
                f"Document content:\n{combined_docs}\n\n"
                "Analyze the documents thoroughly. Extract:\n"
                "- Revenue, profit, EPS, EBITDA figures and trends\n"
                "- Gross margin, operating margin, net margin\n"
                "- Debt-to-equity, current ratio, quick ratio\n"
                "- Year-over-year growth rates\n"
                "- Any notable changes or anomalies\n"
                "Be specific — cite exact numbers from the documents."
            ),
            agent=analyst,
            expected_output=(
                "A detailed financial metrics report with specific numbers, percentages, "
                "and year-over-year comparisons extracted from the documents."
            ),
        )

        # ── Task 2: Risk Assessment ───────────────────────────────
        task2 = Task(
            description=(
                "Using the analyst's financial findings, produce a comprehensive risk assessment.\n"
                "- List 3-5 specific risks with severity: LOW / MEDIUM / HIGH / CRITICAL\n"
                "- Assign an overall risk score: integer from 1 (very safe) to 10 (very risky)\n"
                "- Explain each risk with evidence from the documents"
            ),
            agent=risk_assessor,
            expected_output=(
                "A risk report listing specific risks with severity labels and an overall "
                "numeric risk score from 1-10."
            ),
            context=[task1],
        )

        # ── Task 3: Investment Recommendation (JSON output) ───────
        task3 = Task(
            description=(
                "Synthesize the analysis and risk assessment. "
                "Output ONLY a valid JSON object with exactly these keys:\n\n"
                '{\n'
                '  "recommendation": "BUY" or "HOLD" or "SELL",\n'
                '  "confidence_pct": integer 0-100,\n'
                '  "price_target": number or null,\n'
                '  "risk_score": number 1-10,\n'
                '  "reasons": ["reason 1", "reason 2", "reason 3"],\n'
                '  "risks": ["risk 1", "risk 2"],\n'
                '  "metrics": {"revenue": "...", "margin": "...", "growth": "..."},\n'
                '  "summary": "2-3 sentence plain-English summary"\n'
                '}\n\n'
                "Output ONLY the JSON. No markdown, no explanation, no preamble."
            ),
            agent=advisor,
            expected_output="A single valid JSON object with the investment recommendation.",
            context=[task1, task2],
        )

        # ── Run the crew ──────────────────────────────────────────
        crew = Crew(
            agents=[analyst, risk_assessor, advisor],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=False,
        )

        try:
            raw_result = str(crew.kickoff())
        except Exception as e:
            return self._error_response(str(e))

        return self._parse_result(raw_result)

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        """Extract JSON from crew output, with graceful fallback."""
        # Try strict JSON parse first
        try:
            clean = raw.strip()
            # Strip markdown code fences if present
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
            return json.loads(clean)
        except Exception:
            pass

        # Try finding JSON object inside the text
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Fallback: extract what we can
        return self._fallback_parse(raw)

    def _fallback_parse(self, raw: str) -> Dict[str, Any]:
        """Best-effort parse when JSON extraction fails."""
        rec = "HOLD"
        for word in ["BUY", "SELL", "HOLD"]:
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

