"""Machine-readable Jump-Crash portfolio figures from the paper's Table 6."""

from __future__ import annotations


# Long-short columns: annual mean, annual standard deviation, Sharpe, drawdown.
# Table 6 covers January 2006 through December 2022 and is gross of our costs.
_ROWS = {
    "lightgbm": {
        "B": (.10, .20, .51, -.42), "SDGTV_cp": (.30, .22, 1.37, -.28),
        "S_cp": (.20, .22, .94, -.53), "D_cp": (.26, .21, 1.28, -.51),
        "G_cp": (.13, .21, .62, -.38), "T_cp": (.10, .19, .50, -.35),
        "V_cp": (.13, .21, .62, -.39), "SDGTV_c": (.17, .22, .75, -.38),
        "S_c": (.18, .21, .83, -.36), "D_c": (.19, .21, .92, -.54),
        "G_c": (.09, .19, .45, -.44), "T_c": (.11, .20, .56, -.44),
        "V_c": (.11, .22, .50, -.59), "SDGTV_p": (.29, .23, 1.26, -.40),
        "S_p": (.23, .21, 1.08, -.35), "D_p": (.23, .21, 1.12, -.55),
        "G_p": (.05, .22, .21, -.54), "T_p": (.11, .20, .55, -.41),
        "V_p": (.11, .22, .50, -.47),
    },
    "logistic": {
        "B": (.04, .18, .22, -.42), "SDGTV_cp": (.28, .27, 1.01, -.44),
        "S_cp": (.23, .18, 1.24, -.27), "D_cp": (.19, .19, .98, -.42),
        "G_cp": (.00, .17, -.03, -.56), "T_cp": (.04, .18, .20, -.28),
        "V_cp": (.03, .18, .16, -.49), "SDGTV_c": (.14, .29, .47, -.62),
        "S_c": (.13, .18, .73, -.26), "D_c": (.12, .18, .70, -.33),
        "G_c": (.02, .19, .11, -.45), "T_c": (.03, .18, .16, -.42),
        "V_c": (.03, .18, .16, -.51), "SDGTV_p": (.26, .28, .93, -.52),
        "S_p": (.21, .18, 1.18, -.25), "D_p": (.18, .19, .96, -.40),
        "G_p": (.00, .18, -.01, -.54), "T_p": (.02, .19, .10, -.54),
        "V_p": (.02, .18, .13, -.51),
    },
}


def compare_table_six(portfolio_report: dict) -> dict:
    names = ("annualized_arithmetic_mean", "annualized_volatility", "sharpe", "max_drawdown")
    comparisons = {}
    for model, rows in _ROWS.items():
        for specification, values in rows.items():
            key = f"{specification}/{model}/full"
            observed = portfolio_report.get(key, {}).get("gross")
            reference = dict(zip(names, values))
            comparisons[key] = {
                "paper": reference,
                "observed": {name: observed.get(name) for name in names} if observed else None,
                "difference": {name: observed[name] - reference[name]
                               if observed and observed.get(name) is not None else None
                               for name in names},
            }
    return {
        "paper_table": 6, "paper_period": "2006-01 through 2022-12",
        "comparison_basis": "gross annual arithmetic mean, annualized standard deviation, Sharpe, maximum drawdown",
        "warning": "Compare only after universe, source, timestamps and return intervals are reconciled; rounded paper values have 0.01 precision.",
        "models": comparisons,
    }
