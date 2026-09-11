"""Representative API payloads used across the ingestion tests.

Two shapes of the same statement are kept deliberately: FMP's legacy `/api/v3`
spelling and its newer `/stable` spelling. Tests assert both normalize to
identical canonical records, which is the whole point of `field_maps.py`.
"""

from __future__ import annotations

from typing import Any

# --- Income statement ------------------------------------------------------

INCOME_V3: dict[str, Any] = {
    "date": "2023-09-30",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "cik": "0000320193",
    "fillingDate": "2023-11-03",
    "acceptedDate": "2023-11-02 18:08:27",
    "calendarYear": "2023",
    "period": "FY",
    "revenue": 383285000000,
    "costOfRevenue": 214137000000,
    "grossProfit": 169148000000,
    "researchAndDevelopmentExpenses": 29915000000,
    "sellingGeneralAndAdministrativeExpenses": 24932000000,
    "operatingExpenses": 54847000000,
    "operatingIncome": 114301000000,
    "ebitda": 125820000000,
    "depreciationAndAmortization": 11519000000,
    "interestIncome": 3750000000,
    "interestExpense": 3933000000,
    "incomeBeforeTax": 113736000000,
    "incomeTaxExpense": 16741000000,
    "netIncome": 96995000000,
    "eps": 6.16,
    "epsdiluted": 6.13,
    "weightedAverageShsOut": 15744231000,
    "weightedAverageShsOutDil": 15812547000,
}

# Same figures, newer key spellings: filingDate / fiscalYear / epsDiluted.
INCOME_STABLE: dict[str, Any] = {
    "date": "2023-09-30",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "cik": "0000320193",
    "filingDate": "2023-11-03",
    "acceptedDate": "2023-11-02 18:08:27",
    "fiscalYear": "2023",
    "period": "FY",
    "revenue": 383285000000,
    "costOfRevenue": 214137000000,
    "grossProfit": 169148000000,
    "researchAndDevelopmentExpenses": 29915000000,
    "sellingGeneralAndAdministrativeExpenses": 24932000000,
    "operatingExpenses": 54847000000,
    "operatingIncome": 114301000000,
    "ebitda": 125820000000,
    "depreciationAndAmortization": 11519000000,
    "interestIncome": 3750000000,
    "interestExpense": 3933000000,
    "incomeBeforeTax": 113736000000,
    "incomeTaxExpense": 16741000000,
    "netIncome": 96995000000,
    "eps": 6.16,
    "epsDiluted": 6.13,
    "weightedAverageShsOut": 15744231000,
    "weightedAverageShsOutDil": 15812547000,
}

# --- Balance sheet ---------------------------------------------------------

BALANCE_V3: dict[str, Any] = {
    "date": "2023-09-30",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "fillingDate": "2023-11-03",
    "calendarYear": "2023",
    "period": "FY",
    "cashAndCashEquivalents": 29965000000,
    "shortTermInvestments": 31590000000,
    "cashAndShortTermInvestments": 61555000000,
    "netReceivables": 60985000000,
    "inventory": 6331000000,
    "otherCurrentAssets": 14695000000,
    "totalCurrentAssets": 143566000000,
    "propertyPlantEquipmentNet": 43715000000,
    "goodwill": 0,
    "intangibleAssets": 0,
    "longTermInvestments": 100544000000,
    "totalNonCurrentAssets": 209017000000,
    "totalAssets": 352583000000,
    "accountPayables": 62611000000,
    "shortTermDebt": 15807000000,
    "deferredRevenue": 8061000000,
    "otherCurrentLiabilities": 58829000000,
    "totalCurrentLiabilities": 145308000000,
    "longTermDebt": 95281000000,
    "totalNonCurrentLiabilities": 145129000000,
    "totalLiabilities": 290437000000,
    "commonStock": 73812000000,
    "retainedEarnings": -214000000,
    "totalStockholdersEquity": 62146000000,
    "totalEquity": 62146000000,
    "totalDebt": 111088000000,
    "netDebt": 81123000000,
}

# `accountsReceivables` replaces `netReceivables`; goodwill is simply absent.
BALANCE_STABLE: dict[str, Any] = {
    **{k: v for k, v in BALANCE_V3.items() if k not in {"netReceivables", "goodwill", "fillingDate", "calendarYear"}},
    "accountsReceivables": 60985000000,
    "filingDate": "2023-11-03",
    "fiscalYear": "2023",
}

# --- Cash flow -------------------------------------------------------------

CASH_FLOW_V3: dict[str, Any] = {
    "date": "2023-09-30",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "fillingDate": "2023-11-03",
    "calendarYear": "2023",
    "period": "FY",
    "netIncome": 96995000000,
    "depreciationAndAmortization": 11519000000,
    "stockBasedCompensation": 10833000000,
    "changeInWorkingCapital": -6577000000,
    "accountsReceivables": -1688000000,
    "inventory": -1618000000,
    "netCashProvidedByOperatingActivities": 110543000000,
    "investmentsInPropertyPlantAndEquipment": -10959000000,
    "acquisitionsNet": 0,
    "netCashUsedForInvestingActivites": 3705000000,
    "debtRepayment": -11151000000,
    "commonStockRepurchased": -77550000000,
    "dividendsPaid": -15025000000,
    "netCashUsedProvidedByFinancingActivities": -108488000000,
    "netChangeInCash": 5760000000,
    "cashAtEndOfPeriod": 30737000000,
    "operatingCashFlow": 110543000000,
    "capitalExpenditure": -10959000000,
    "freeCashFlow": 99584000000,
}

CASH_FLOW_STABLE: dict[str, Any] = {
    **{
        k: v
        for k, v in CASH_FLOW_V3.items()
        if k
        not in {
            "netCashUsedForInvestingActivites",
            "netCashUsedProvidedByFinancingActivities",
            "investmentsInPropertyPlantAndEquipment",
            "fillingDate",
            "calendarYear",
            "dividendsPaid",
        }
    },
    "netCashProvidedByInvestingActivities": 3705000000,
    "netCashProvidedByFinancingActivities": -108488000000,
    "netDividendsPaid": -15025000000,
    "filingDate": "2023-11-03",
    "fiscalYear": "2023",
}

# --- Profile ---------------------------------------------------------------

PROFILE_V3: dict[str, Any] = {
    "symbol": "AAPL",
    "price": 178.72,
    "mktCap": 2794144143000,
    "companyName": "Apple Inc.",
    "currency": "USD",
    "cik": "0000320193",
    "exchange": "NASDAQ Global Select",
    "exchangeShortName": "NASDAQ",
    "industry": "Consumer Electronics",
    "sector": "Technology",
    "country": "US",
    "description": "Apple Inc. designs, manufactures and markets smartphones.",
}

PROFILE_BANK: dict[str, Any] = {
    **PROFILE_V3,
    "symbol": "JPM",
    "companyName": "JPMorgan Chase & Co.",
    "sector": "Financial Services",
    "industry": "Banks - Diversified",
}


def statements_payload(shape: str = "v3") -> dict[str, list[dict[str, Any]]]:
    """The three statement lists as the FMP client returns them."""
    if shape == "stable":
        return {
            "income_statement": [INCOME_STABLE],
            "balance_sheet": [BALANCE_STABLE],
            "cash_flow": [CASH_FLOW_STABLE],
        }
    return {
        "income_statement": [INCOME_V3],
        "balance_sheet": [BALANCE_V3],
        "cash_flow": [CASH_FLOW_V3],
    }
