"""CoinGecko PRO signals for AI Analysis v2.

Provides three signals that CoinGecko PRO uniquely enables:

1. Coinbase Premium Index (CPI): Coinbase BTC price vs Binance BTC price.
   Positive = institutional buying on US exchanges = bullish leading indicator.
   Research: CPI has documented 5-30 minute lead on broader market moves.

2. Cross-Exchange Volume Spike Ratio: Current 5-min volume vs 30-min rolling average,
   aggregated across 600+ exchanges. A spike validates the current price move.
   Kraken alone is a mid-sized exchange — Binance spikes are invisible in Kraken data.

3. Funding Rate at Extremes: Only acts when outside ±0.05% range.
   Normal funding (0.01-0.03%) is noise. Extremes signal crowded positioning risk.

Why Kraken for OHLCV (not CoinGecko):
   Kraken: real-time 1-min true OHLCV, free, no auth.
   CoinGecko ohlc: 15-min cache, 30-min finest granularity → useless for 5-15 min prediction.
"""
import asyncio
import aiohttp
import structlog
from typing import Optional

from AI_analysis_upgrade import config

logger = structlog.get_logger()

# CoinGecko PRO base URL
_BASE = "https://pro-api.coingecko.com/api/v3"

# Funding rate thresholds — only signal when truly extreme
_FUNDING_EXTREME_THRESHOLD = 0.05   # ±0.05% = approximately 2x normal


class CoinGeckoSignals:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or config.COINGECKO_API_KEY
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"x-cg-pro-api-key": self._api_key}
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str, params: Optional[dict] = None):
        session = await self._get_session()
        url = f"{_BASE}/{endpoint}"
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def fetch_coinbase_premium(self) -> float:
        """Compute Coinbase Premium Index: (Coinbase_BTC - Binance_BTC) / Binance_BTC * 100.

        Returns:
            CPI in percent. Positive = Coinbase higher (institutional buying).
            Returns 0.0 on any error (neutral signal).
        """
        try:
            # Fetch Coinbase (gdax) and Binance BTC tickers in parallel
            coinbase_data, binance_data = await asyncio.gather(
                self._get("exchanges/gdax/tickers", {"coin_ids": "bitcoin", "order": "trust_score_desc"}),
                self._get("exchanges/binance/tickers", {"coin_ids": "bitcoin", "order": "trust_score_desc"}),
            )

            # Find BTC/USD on Coinbase and BTC/USDT on Binance
            coinbase_price = next(
                (t["last"] for t in coinbase_data.get("tickers", [])
                 if t.get("base") == "BTC" and t.get("coin_id") == "bitcoin"),
                None
            )
            binance_price = next(
                (t["last"] for t in binance_data.get("tickers", [])
                 if t.get("base") == "BTC" and t.get("coin_id") == "bitcoin"),
                None
            )

            if not coinbase_price or not binance_price or binance_price == 0:
                return 0.0

            cpi = (coinbase_price - binance_price) / binance_price * 100.0
            logger.debug("Coinbase Premium Index", cpi=f"{cpi:+.4f}%",
                        cb=coinbase_price, bnb=binance_price)
            return cpi

        except Exception as e:
            logger.warning("fetch_coinbase_premium failed", error=str(e))
            return 0.0

    async def fetch_volume_spike_ratio(self) -> float:
        """Compute cross-exchange BTC volume spike ratio.

        Fetches last 24h of 5-minute BTC volume aggregated across 600+ exchanges.
        Returns current_5min_vol / rolling_30min_avg_vol.

        ratio > 1.8 = abnormal volume (validates or challenges current move)
        ratio < 0.5 = unusually thin market (treat price level with less conviction)
        Returns 1.0 on error (neutral).
        """
        try:
            data = await self._get("coins/bitcoin/market_chart", {
                "vs_currency": "usd",
                "days": "1",
            })
            volumes = data.get("total_volumes", [])
            if len(volumes) < 6:
                return 1.0

            # volumes[-1] is most recent 5-min point
            # volumes[-31:-1] is the preceding 30 points (150 minutes) for rolling avg
            current_vol = volumes[-1][1]
            rolling_avg = sum(v[1] for v in volumes[-31:-1]) / 30 if len(volumes) >= 31 else (
                sum(v[1] for v in volumes[:-1]) / (len(volumes) - 1)
            )

            if rolling_avg <= 0:
                return 1.0

            ratio = current_vol / rolling_avg
            logger.debug("Volume spike ratio", ratio=f"{ratio:.2f}",
                        current=f"{current_vol:.0f}", avg=f"{rolling_avg:.0f}")
            return ratio

        except Exception as e:
            logger.warning("fetch_volume_spike_ratio failed", error=str(e))
            return 1.0

    async def fetch_funding_rate(self) -> Optional[float]:
        """Fetch BTC perpetual funding rate from CoinGecko /derivatives.

        Only returns a value when the rate is extreme (outside ±0.05%).
        Returns None in the normal range — no signal.

        Extremes:
            > +0.05%: crowded longs, mean-reversion risk for YES markets
            < -0.05%: crowded shorts, squeeze risk for NO markets
        """
        try:
            data = await self._get("derivatives")
            btc_perps = [
                d for d in data
                if d.get("index_id") == "BTC" and d.get("contract_type") == "perpetual"
                and d.get("funding_rate") is not None
            ]
            if not btc_perps:
                return None

            # Weighted average by volume_24h
            total_vol = sum(float(d.get("volume_24h", 0) or 0) for d in btc_perps)
            if total_vol <= 0:
                avg_rate = sum(float(d["funding_rate"]) for d in btc_perps) / len(btc_perps)
            else:
                avg_rate = sum(
                    float(d["funding_rate"]) * float(d.get("volume_24h", 0) or 0)
                    for d in btc_perps
                ) / total_vol

            logger.debug("Funding rate", avg=f"{avg_rate:+.4f}%", n_perps=len(btc_perps))

            # Only return if extreme
            if abs(avg_rate) >= _FUNDING_EXTREME_THRESHOLD:
                return avg_rate
            return None

        except Exception as e:
            logger.warning("fetch_funding_rate failed", error=str(e))
            return None

    async def fetch_all(self) -> dict:
        """Fetch all three signals concurrently. Returns dict with keys: cpi, volume_spike, funding_rate."""
        cpi, volume_spike, funding_rate = await asyncio.gather(
            self.fetch_coinbase_premium(),
            self.fetch_volume_spike_ratio(),
            self.fetch_funding_rate(),
        )
        return {
            "cpi": cpi,
            "volume_spike": volume_spike,
            "funding_rate": funding_rate,
        }
