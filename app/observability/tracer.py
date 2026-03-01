"""Simple span-based tracer for pipeline observability."""
from __future__ import annotations

from typing import Any

from app.graph.state import Span


def summarize_trace(spans: list[Span]) -> dict[str, Any]:
    """Convert raw spans into a human-readable trace summary."""
    if not spans:
        return {}

    total_ms = sum(s["end_ms"] - s["start_ms"] for s in spans)
    node_timings = {
        s["node"]: round(s["end_ms"] - s["start_ms"], 1)
        for s in spans
    }

    return {
        "total_latency_ms": round(total_ms, 1),
        "node_timings_ms":  node_timings,
        "nodes_executed":   [s["node"] for s in spans],
        "span_details":     [
            {
                "node":       s["node"],
                "duration_ms": round(s["end_ms"] - s["start_ms"], 1),
                **s["metadata"],
            }
            for s in spans
        ],
    }
