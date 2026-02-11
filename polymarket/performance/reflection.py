# polymarket/performance/reflection.py
"""Reflection engine for AI-powered performance analysis."""

import json
from typing import Dict, Optional
from datetime import datetime
import structlog
from openai import AsyncOpenAI

from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.metrics import MetricsCalculator
from polymarket.config import Settings

logger = structlog.get_logger()


class ReflectionEngine:
    """AI-powered trading performance analysis."""

    def __init__(self, db: PerformanceDatabase, settings: Settings):
        """
        Initialize reflection engine.

        Args:
            db: PerformanceDatabase instance
            settings: Bot settings
        """
        self.db = db
        self.settings = settings
        self.metrics = MetricsCalculator(db)
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy init OpenAI client."""
        if self._client is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def analyze_performance(
        self,
        trigger_type: str,
        trades_analyzed: int
    ) -> Dict:
        """
        Analyze trading performance and generate insights.

        Args:
            trigger_type: '3_losses', '10_trades', 'end_of_day'
            trades_analyzed: Number of trades to analyze

        Returns:
            Dict with insights and recommendations
        """
        try:
            # Generate prompt
            prompt = await self._generate_prompt(trade_count=trades_analyzed)

            # Call OpenAI
            insights = await self._call_openai(prompt)

            # Store in database
            self._store_reflection(trigger_type, trades_analyzed, insights)

            logger.info(
                "Reflection complete",
                trigger_type=trigger_type,
                trades_analyzed=trades_analyzed,
                insights_count=len(insights.get("insights", []))
            )

            return insights

        except Exception as e:
            logger.error("Reflection failed", error=str(e), trigger_type=trigger_type)
            return {"insights": [], "patterns": {}, "recommendations": []}

    async def _generate_prompt(self, trade_count: int) -> str:
        """Generate reflection prompt with current metrics."""
        # Calculate metrics
        win_rate = self.metrics.calculate_win_rate(days=7)
        total_profit = self.metrics.calculate_total_profit(days=7)
        signal_perf = self.metrics.calculate_signal_performance(days=7)

        # Format signal performance
        signal_breakdown = "\n".join([
            f"- {sig}: {perf['win_rate']*100:.1f}% win rate ({perf['wins']}W-{perf['losses']}L), avg profit ${perf['avg_profit']:.2f}"
            for sig, perf in signal_perf.items()
        ])

        prompt = f"""You are analyzing your own Polymarket trading performance as a self-improving AI.

**Recent Performance (Last 7 Days):**
- Trades: {trade_count}
- Win Rate: {win_rate*100:.1f}%
- Total Profit: ${total_profit:.2f}

**Signal Performance:**
{signal_breakdown}

**Current Parameters:**
- Confidence Threshold: {self.settings.bot_confidence_threshold}
- Max Position: ${self.settings.bot_max_position_dollars}
- Max Exposure: {self.settings.bot_max_exposure_percent*100:.0f}%

**Analysis Tasks:**
1. What patterns led to winning trades?
2. What mistakes are being repeated?
3. Should confidence threshold be adjusted? Why?
4. Which signal types should be trusted more/less?
5. Recommend 2-3 specific parameter adjustments with detailed reasoning.

**Output Format (JSON):**
{{
  "insights": [
    "Specific actionable insight 1",
    "Specific actionable insight 2"
  ],
  "patterns": {{
    "winning": ["Pattern that leads to wins"],
    "losing": ["Pattern that leads to losses"]
  }},
  "recommendations": [
    {{
      "parameter": "bot_confidence_threshold",
      "current": {self.settings.bot_confidence_threshold},
      "recommended": 0.70,
      "reason": "Detailed reasoning",
      "tier": 2,
      "expected_impact": "Impact description"
    }}
  ]
}}
"""
        return prompt

    async def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API for reflection."""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a self-improving trading AI analyzing your own performance. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        insights = json.loads(content)

        return insights

    def _store_reflection(self, trigger_type: str, trades_analyzed: int, insights: Dict):
        """Store reflection results to database."""
        cursor = self.db.conn.cursor()

        cursor.execute("""
            INSERT INTO reflections (timestamp, trigger_type, trades_analyzed, insights, adjustments_made)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            trigger_type,
            trades_analyzed,
            json.dumps(insights),
            None  # Adjustments filled in later by Parameter Adjuster
        ))

        self.db.conn.commit()
