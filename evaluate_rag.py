"""
Evaluation Harness for Multimodal RAG.

Tests:
1. Summary-Based Retrieval vs Source-Based Retrieval vs Hybrid (Summary + Source).
2. Category-specific performance on Numerical, Terminology/Math, and Visual Architecture questions.
3. Quantifies summary distortion where summaries omit exact numbers, parameters, or equations.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def load_eval_dataset(path: str = "multimodal/eval_dataset.json") -> List[Dict[str, Any]]:
    """Loads the golden evaluation benchmark."""
    with open(path, "r") as f:
        return json.load(f)


def check_facts_present(retrieved_text: str, required_facts: List[str]) -> bool:
    """Checks if all required ground-truth facts/numbers are present in retrieved text."""
    retrieved_lower = retrieved_text.lower()
    return all(fact.lower() in retrieved_lower for fact in required_facts)


def run_benchmark(retriever_fn, dataset: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
    """
    Evaluates a retriever function against the golden benchmark.
    
    retriever_fn(query: str, k: int) -> List[Dict[str, Any]]
        Returns list of retrieved documents with 'content', 'summary', 'page_number', 'type'.
    """
    results_by_category = {
        "numerical": {"total": 0, "hits": 0},
        "terminology": {"total": 0, "hits": 0},
        "visual": {"total": 0, "hits": 0},
        "overall": {"total": 0, "hits": 0},
    }
    
    detailed_logs = []

    for item in dataset:
        cat = item["category"]
        results_by_category[cat]["total"] += 1
        results_by_category["overall"]["total"] += 1

        # Execute retrieval
        retrieved_docs = retriever_fn(item["query"], k=k)
        
        # Combine retrieved content into a single searchable buffer
        combined_text = ""
        for d in retrieved_docs:
            if isinstance(d, dict):
                combined_text += " " + str(d.get("content", "")) + " " + str(d.get("summary", ""))
            else:
                combined_text += " " + str(d)

        # Check fact retention
        is_hit = check_facts_present(combined_text, item["required_facts"])

        if is_hit:
            results_by_category[cat]["hits"] += 1
            results_by_category["overall"]["hits"] += 1

        detailed_logs.append({
            "id": item["id"],
            "query": item["query"],
            "category": cat,
            "hit": is_hit,
            "required_facts": item["required_facts"]
        })

    # Compute percentages
    summary_report = {}
    for cat, stats in results_by_category.items():
        total = stats["total"]
        hits = stats["hits"]
        summary_report[cat] = {
            "hits": hits,
            "total": total,
            "accuracy": round((hits / total) * 100, 1) if total > 0 else 0.0,
        }

    return {
        "summary": summary_report,
        "details": detailed_logs
    }


def format_evaluation_table(strategies_results: Dict[str, Dict[str, Any]]) -> str:
    """Formats comparative scorecard as a Markdown table."""
    headers = ["Strategy", "Overall Recall", "Numerical Recall", "Terminology Recall", "Visual Recall"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    
    for strategy_name, res in strategies_results.items():
        s = res["summary"]
        row = [
            strategy_name,
            f"{s['overall']['accuracy']}% ({s['overall']['hits']}/{s['overall']['total']})",
            f"{s['numerical']['accuracy']}% ({s['numerical']['hits']}/{s['numerical']['total']})",
            f"{s['terminology']['accuracy']}% ({s['terminology']['hits']}/{s['terminology']['total']})",
            f"{s['visual']['accuracy']}% ({s['visual']['hits']}/{s['visual']['total']})",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

