"""Cost tracking and display — per-model token counting and USD estimation.

KEY CONCEPTS:
  1. Cost is tracked per model with tiered pricing
  2. Each API response's usage (input/output tokens) is accumulated
  3. Total cost persists across sessions via project config
  4. Cost display shows per-model breakdown with token counts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# MODEL PRICING
# ─────────────────────────────────────────────────────────
# For OpenAI models, we simplify to input/output pricing only.
#
# Prices from https://platform.openai.com/pricing (as of 2025)


@dataclass(frozen=True)
class ModelPricing:
    """Per-model pricing in USD per million tokens.

    For dazi with OpenAI, we use simplified input/output pricing.
    """
    model: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float


# Prices for OpenAI models (USD per million tokens)
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.00, 30.00),
    "gpt-4": ModelPricing("gpt-4", 30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.50, 1.50),
    "o1": ModelPricing("o1", 15.00, 60.00),
    "o1-mini": ModelPricing("o1-mini", 1.10, 4.40),
    "o3-mini": ModelPricing("o3-mini", 1.10, 4.40),
    "default": ModelPricing("default", 3.00, 15.00),
}

DEFAULT_PRICING = MODEL_PRICING["default"]


# ─────────────────────────────────────────────────────────
# COST CALCULATION
# ─────────────────────────────────────────────────────────
# Formula:
#   cost = (input_tokens / 1M) * input_price
#        + (output_tokens / 1M) * output_price


def _get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fallback to default.

    Tries exact match first, then prefix match, then default.
    """
    # Exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
    for key, pricing in MODEL_PRICING.items():
        if key != "default" and model.startswith(key):
            return pricing

    return DEFAULT_PRICING


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD."""
    pricing = _get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_mtok
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_mtok
    return input_cost + output_cost


# ─────────────────────────────────────────────────────────
# COST RECORD
# ─────────────────────────────────────────────────────────


@dataclass
class CostRecord:
    """Accumulated cost data for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    request_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "request_count": self.request_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostRecord:
        """Deserialize from dict."""
        return cls(
            model=data.get("model", "unknown"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            request_count=data.get("request_count", 0),
        )


# ─────────────────────────────────────────────────────────
# COST TRACKER
# ─────────────────────────────────────────────────────────


class CostTracker:
    """Tracks token usage and estimated cost per model across a session.

    The tracker accumulates costs in-memory during a session and
    persists to .dazi/last_session.json on save().

    Cost calculation:
        cost = (input_tokens / 1_000_000) * input_price
             + (output_tokens / 1_000_000) * output_price
    """

    def __init__(self, persistence_dir: Path) -> None:
        """Initialize cost tracker.

        Args:
            persistence_dir: Directory for last_session.json persistence.
        """
        self._records: dict[str, CostRecord] = {}
        self._persistence_path = persistence_dir / "last_session.json"

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Record a single LLM request's token usage and accumulate cost.

        Called after each LLM response to accumulate tokens and cost.

        Args:
            model: Model identifier string.
            input_tokens: Number of input tokens (prompt_tokens from OpenAI).
            output_tokens: Number of output tokens (completion_tokens from OpenAI).

        Returns:
            Updated CostRecord for this model.
        """
        if model not in self._records:
            self._records[model] = CostRecord(model=model)

        record = self._records[model]
        record.input_tokens += input_tokens
        record.output_tokens += output_tokens
        record.request_count += 1
        record.estimated_cost_usd = calculate_cost(model, record.input_tokens, record.output_tokens)

        return record

    def get_total_cost(self) -> float:
        """Sum of estimated_cost_usd across all models."""
        return sum(r.estimated_cost_usd for r in self._records.values())

    def get_total_tokens(self) -> tuple[int, int]:
        """Total (input_tokens, output_tokens) across all models."""
        total_input = sum(r.input_tokens for r in self._records.values())
        total_output = sum(r.output_tokens for r in self._records.values())
        return total_input, total_output

    def get_total_request_count(self) -> int:
        """Total number of LLM requests across all models."""
        return sum(r.request_count for r in self._records.values())

    def get_model_summary(self) -> dict[str, CostRecord]:
        """Per-model breakdown."""
        return dict(self._records)

    def format_cost(self) -> str:
        """Format total cost for display.

        Example: "$0.0234"
        """
        return f"${self.get_total_cost():.4f}"

    def format_summary(self) -> str:
        """Multi-line summary with per-model breakdown.

        Shows total cost, total tokens, and per-model rows.
        """
        total_input, total_output = self.get_total_tokens()
        total_cost = self.get_total_cost()
        request_count = self.get_total_request_count()

        lines = [
            f"Total cost:    {self.format_cost()}",
            f"Total tokens:  {total_input:,} input, {total_output:,} output",
            f"Total requests: {request_count}",
        ]

        if len(self._records) > 1:
            lines.append("")
            lines.append("Usage by model:")
            for model, record in self._records.items():
                lines.append(
                    f"  {model}: {record.input_tokens:,} input, "
                    f"{record.output_tokens:,} output "
                    f"({record.request_count} requests, "
                    f"${record.estimated_cost_usd:.4f})"
                )

        return "\n".join(lines)

    def save(self) -> None:
        """Persist current session costs to last_session.json."""
        data = {
            "model_usage": {
                model: record.to_dict()
                for model, record in self._records.items()
            },
            "total_cost_usd": round(self.get_total_cost(), 6),
            "total_input_tokens": self.get_total_tokens()[0],
            "total_output_tokens": self.get_total_tokens()[1],
            "total_request_count": self.get_total_request_count(),
        }

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load_last_session(self) -> dict[str, Any] | None:
        """Load previous session costs from last_session.json.

        Returns None if file doesn't exist.
        """
        if not self._persistence_path.exists():
            return None

        try:
            text = self._persistence_path.read_text(encoding="utf-8")
            return json.loads(text)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load last session costs: {e}")
            return None

    def format_last_session(self) -> str:
        """Format previous session costs for display."""
        data = self.load_last_session()
        if data is None:
            return "No previous session data found."

        lines = [
            "Previous session costs:",
            f"  Total cost:    ${data.get('total_cost_usd', 0):.4f}",
            f"  Total tokens:  {data.get('total_input_tokens', 0):,} input, "
            f"{data.get('total_output_tokens', 0):,} output",
            f"  Total requests: {data.get('total_request_count', 0)}",
        ]

        model_usage = data.get("model_usage", {})
        if model_usage:
            lines.append("")
            lines.append("  Usage by model:")
            for model, usage in model_usage.items():
                lines.append(
                    f"    {model}: {usage.get('input_tokens', 0):,} input, "
                    f"{usage.get('output_tokens', 0):,} output "
                    f"({usage.get('request_count', 0)} requests, "
                    f"${usage.get('estimated_cost_usd', 0):.4f})"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all accumulated data. Used for testing."""
        self._records.clear()
