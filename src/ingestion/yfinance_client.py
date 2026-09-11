"""yfinance client — market prices, shares outstanding, market cap.

FMP's free tier is spent on statements, so price history comes from yfinance:
no key, no quota. It matters beyond charting — the Altman Z-Score's X4 term needs
*market* value of equity, and without it Layer 2 has to fall back to the
book-value Z''-Score variant. This client therefore reports failure explicitly
rather than returning an empty series that would look like "no price movement".

yfinance is an unofficial scraper and does break from time to time; every call
here is defensive, and a failure degrades the run to `partial` instead of
aborting it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from src.ingestion.schemas import MarketDataPoint

logger = logging.getLogger(__name__)


class YFinanceUnavailableError(RuntimeError):
    """yfinance is not installed, or returned nothing usable for the ticker."""


def _import_yfinance() -> Any:
    try:
        import yfinance as yf  # noqa: PLC0415 - optional at import time
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise YFinanceUnavailableError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return yf


class YFinanceClient:
    """Fetches price history and share counts for a ticker."""

    def __init__(self, session: Any | None = None) -> None:
        self._yf = _import_yfinance()
        self._session = session

    def _ticker(self, symbol: str) -> Any:
        if self._session is not None:
            return self._yf.Ticker(symbol, session=self._session)
        return self._yf.Ticker(symbol)

    def get_shares_outstanding(self, symbol: str) -> float | None:
        """Share count used to convert price into market value of equity."""
        try:
            info = self._ticker(symbol).info or {}
        except Exception as exc:  # noqa: BLE001 - yfinance raises many things
            logger.warning("yfinance info lookup failed for %s: %s", symbol, exc)
            return None
        for key in ("sharesOutstanding", "impliedSharesOutstanding", "floatShares"):
            value = info.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    def get_price_history(
        self,
        symbol: str,
        years: int = 5,
        interval: str = "1d",
        shares_outstanding: float | None = None,
    ) -> list[MarketDataPoint]:
        """Daily closes for the last `years` years, as MarketDataPoint records.

        `market_cap` is derived as close x shares outstanding when the share count
        is known, and left None otherwise — never estimated.
        """
        start = date.today() - timedelta(days=int(years * 365.25) + 5)
        try:
            history = self._ticker(symbol).history(
                start=start.isoformat(), interval=interval, auto_adjust=False
            )
        except Exception as exc:  # noqa: BLE001
            raise YFinanceUnavailableError(f"yfinance history failed for {symbol}: {exc}") from exc

        if history is None or history.empty:
            raise YFinanceUnavailableError(f"yfinance returned no price history for {symbol}")

        if shares_outstanding is None:
            shares_outstanding = self.get_shares_outstanding(symbol)

        points: list[MarketDataPoint] = []
        for index, row in history.iterrows():
            observed = index.date() if isinstance(index, datetime) else index
            if hasattr(observed, "date"):  # pandas Timestamp
                observed = observed.date()
            close = row.get("Close")
            close_value = float(close) if close is not None and close == close else None
            volume = row.get("Volume")
            volume_value = float(volume) if volume is not None and volume == volume else None
            market_cap = (
                close_value * shares_outstanding
                if close_value is not None and shares_outstanding
                else None
            )
            points.append(
                MarketDataPoint(
                    ticker=symbol,
                    date=observed,
                    close_price=close_value,
                    volume=volume_value,
                    market_cap=market_cap,
                    shares_outstanding=shares_outstanding,
                )
            )

        logger.info("yfinance: %d price points for %s", len(points), symbol)
        return points

    def get_fiscal_year_end_prices(
        self, symbol: str, years: int = 5
    ) -> list[MarketDataPoint]:
        """One observation per month — enough for period-end valuation ratios.

        Storing every trading day for many tickers bloats the SQLite file for no
        analytical gain, since ratios are computed at period ends.
        """
        daily = self.get_price_history(symbol, years=years)
        by_month: dict[tuple[int, int], MarketDataPoint] = {}
        for point in daily:
            by_month[(point.date.year, point.date.month)] = point  # last of month wins
        return [by_month[key] for key in sorted(by_month)]
