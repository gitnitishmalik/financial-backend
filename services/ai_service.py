"""
ai_service.py  –  Entry-point service used by routes.py

This is the file that FastAPI routes call.  It now delegates all heavy
lifting to AnalysisService (multi-agent CrewAI + RAG) instead of running
a single bare LLM call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.analysis_service import AnalysisService

# One instance per process — AnalysisService is stateless between calls
_analysis_service = AnalysisService()


async def analyze_documents(file_paths: List[str], query: str) -> Dict[str, Any]:
    """
    Analyse financial documents with the 3-agent CrewAI pipeline.

    Parameters
    ----------
    file_paths : list of absolute paths to uploaded files
    query      : the user's question / analysis request

    Returns
    -------
    dict with keys: recommendation, confidence_pct, price_target,
                    risk_score, reasons, risks, metrics, summary
    """
    return await _analysis_service.analyze(file_paths, query)


async def chat_with_documents(file_paths: List[str], message: str) -> str:
    """
    Lightweight chat mode: retrieve relevant context via RAG then answer
    with a single LLM call (no crew overhead for simple Q&A).
    """
    from services.rag_service import RAGService
    from services.analysis_service import extract_text_from_file, get_groq_llm

    rag = RAGService()
    texts = [extract_text_from_file(fp) for fp in file_paths]
    rag.ingest([t for t in texts if t])
    context = rag.retrieve(message, top_k=5)

    llm = get_groq_llm()

    prompt = (
        "You are FinAnalyzer AI, an expert financial analyst.\n"
        "Answer the user's question using ONLY the document context provided.\n"
        "If the answer is not in the context, say so clearly.\n\n"
        f"Document context:\n{context}\n\n"
        f"User question: {message}"
    )

    # crewai LLM exposes .call() for single-shot completions
    response = llm.call([{"role": "user", "content": prompt}])
    return response if isinstance(response, str) else str(response)