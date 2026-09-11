"""Canonical field name -> source field aliases.

FMP serves two generations of its API with slightly different key spellings
(`/stable/...` vs legacy `/api/v3/...`: `epsDiluted` vs `epsdiluted`,
`filingDate` vs `fillingDate`, `accountsReceivables` vs `netReceivables`, and so
on). Rather than branching on which endpoint answered, every canonical field
lists the aliases it may arrive under and the normalizer takes the first one
present. Adding a second data source later means adding aliases here, not
rewriting the pipeline.

Order matters: aliases are tried left to right, so put the preferred spelling
first.
"""

from __future__ import annotations

# --- Fields shared by all three statements ---------------------------------

STATEMENT_META_ALIASES: dict[str, list[str]] = {
    "period_end_date": ["date"],
    "filing_date": ["filingDate", "fillingDate"],
    "reported_currency": ["reportedCurrency"],
    "fiscal_year": ["fiscalYear", "calendarYear"],
    "period": ["period"],
}


INCOME_STATEMENT_ALIASES: dict[str, list[str]] = {
    "revenue": ["revenue"],
    "cost_of_revenue": ["costOfRevenue"],
    "gross_profit": ["grossProfit"],
    "research_and_development": ["researchAndDevelopmentExpenses"],
    "selling_general_admin": ["sellingGeneralAndAdministrativeExpenses"],
    "operating_expenses": ["operatingExpenses"],
    # `ebit` is the stable API's explicit EBIT; operatingIncome is the v3 stand-in.
    "operating_income": ["operatingIncome", "ebit"],
    "ebitda": ["ebitda", "EBITDA"],
    "depreciation_amortization": ["depreciationAndAmortization"],
    "interest_expense": ["interestExpense"],
    "interest_income": ["interestIncome"],
    "income_before_tax": ["incomeBeforeTax"],
    "income_tax_expense": ["incomeTaxExpense"],
    "net_income": ["netIncome"],
    "eps": ["eps"],
    "eps_diluted": ["epsDiluted", "epsdiluted"],
    "weighted_average_shares": ["weightedAverageShsOut"],
    "weighted_average_shares_diluted": ["weightedAverageShsOutDil"],
}


BALANCE_SHEET_ALIASES: dict[str, list[str]] = {
    "cash_and_equivalents": ["cashAndCashEquivalents"],
    "short_term_investments": ["shortTermInvestments"],
    "cash_and_short_term_investments": ["cashAndShortTermInvestments"],
    "net_receivables": ["netReceivables", "accountsReceivables"],
    "inventory": ["inventory"],
    "other_current_assets": ["otherCurrentAssets"],
    "total_current_assets": ["totalCurrentAssets"],
    "property_plant_equipment_net": ["propertyPlantEquipmentNet"],
    "goodwill": ["goodwill"],
    "intangible_assets": ["intangibleAssets"],
    "long_term_investments": ["longTermInvestments"],
    "total_non_current_assets": ["totalNonCurrentAssets"],
    "total_assets": ["totalAssets"],
    "accounts_payable": ["accountPayables", "accountsPayables"],
    "short_term_debt": ["shortTermDebt"],
    "deferred_revenue": ["deferredRevenue"],
    "other_current_liabilities": ["otherCurrentLiabilities"],
    "total_current_liabilities": ["totalCurrentLiabilities"],
    "long_term_debt": ["longTermDebt"],
    "total_non_current_liabilities": ["totalNonCurrentLiabilities"],
    "total_liabilities": ["totalLiabilities"],
    "common_stock": ["commonStock"],
    "retained_earnings": ["retainedEarnings"],
    # totalStockholdersEquity excludes minority interest — the right base for
    # ROE and for the Z''-Score book-value variant.
    "total_equity": ["totalStockholdersEquity", "totalEquity"],
    "total_debt": ["totalDebt"],
    "net_debt": ["netDebt"],
}


CASH_FLOW_ALIASES: dict[str, list[str]] = {
    "net_income": ["netIncome"],
    "depreciation_amortization": ["depreciationAndAmortization"],
    "stock_based_compensation": ["stockBasedCompensation"],
    "change_in_working_capital": ["changeInWorkingCapital"],
    "accounts_receivable_change": ["accountsReceivables"],
    "inventory_change": ["inventory"],
    "operating_cash_flow": ["operatingCashFlow", "netCashProvidedByOperatingActivities"],
    "capital_expenditure": ["capitalExpenditure", "investmentsInPropertyPlantAndEquipment"],
    "acquisitions_net": ["acquisitionsNet"],
    "investing_cash_flow": [
        "netCashProvidedByInvestingActivities",
        "netCashUsedForInvestingActivites",  # v3 ships this misspelling
    ],
    "debt_repayment": ["debtRepayment"],
    "dividends_paid": ["netDividendsPaid", "dividendsPaid"],
    "common_stock_repurchased": ["commonStockRepurchased"],
    "financing_cash_flow": [
        "netCashProvidedByFinancingActivities",
        "netCashUsedProvidedByFinancingActivities",
    ],
    "net_change_in_cash": ["netChangeInCash"],
    "cash_at_end_of_period": ["cashAtEndOfPeriod"],
    "free_cash_flow": ["freeCashFlow"],
}


PROFILE_ALIASES: dict[str, list[str]] = {
    "name": ["companyName"],
    "sector": ["sector"],
    "industry": ["industry"],
    "exchange": ["exchangeShortName", "exchange"],
    "country": ["country"],
    "currency": ["currency"],
    "cik": ["cik"],
    "description": ["description"],
    "market_cap": ["mktCap", "marketCap"],
    "price": ["price"],
    "shares_outstanding": ["sharesOutstanding"],
}


# Fields whose absence makes most downstream analysis impossible. Missing ones
# are logged at WARNING and surfaced as a Data Integrity red flag in Layer 2,
# rather than being silently tolerated.
CRITICAL_FIELDS: dict[str, set[str]] = {
    "income_statement": {"revenue", "net_income", "operating_income"},
    "balance_sheet": {
        "total_assets",
        "total_liabilities",
        "total_equity",
        "total_current_assets",
        "total_current_liabilities",
    },
    "cash_flow": {"operating_cash_flow"},
}


# Sectors for which the Altman Z-Score is not defined (Layer 2 returns
# "Not Applicable — Financial Sector" instead of a misleading number).
FINANCIAL_SECTOR_LABELS: set[str] = {
    "financial services",
    "financials",
    "financial",
    "banks",
    "banking",
    "insurance",
    "capital markets",
}
