# Research hypotheses and constraints

These are candidate research ideas, not instructions to override experiment
configuration or evidence that a profitable strategy has been established.
The consolidated research commands preserve the existing targets, universes,
execution timing and accounting unless a new experiment explicitly changes them.

1. Compare relative-return and raw-return targets in a separately declared
   chronological experiment. Cross-sectional ranking alone does not prove
   market beta caused earlier losses. An assumed next-open target must be
   reconciled with actual feature publication times before it is executable.
2. Examine dated size, momentum, volatility, skew and turnover controls.
   Similar column names and fallback formulas do not establish reproduction
   of the reference paper's nine controls or their publication lags.
3. Compare compact and training-only pruned features with full controls.
   Collinearity does not prove noise amplification; the inspected experiments
   do not establish that any compact set is universally superior or profitable.
4. Examine long and short legs, holding periods, concentration and costs.
   Longer holding periods may reduce turnover, but no fixed percentage
   reduction or profit guarantee follows without a measured comparison.
5. Audit data quality and any investability rule using information available
   at the decision. `is_target_valid_5d` depends on future outcomes: it may
   identify matured training/scoring labels, never remove a stock from a
   historical portfolio universe. Investability flags cannot guarantee future
   exit prices, quote quality, borrow availability or corporate-action accuracy.

Exact 30/60/90-day features and mixed-maturity fallback fields remain distinct.
Any adoption of new controls, filters or engineered inputs needs definitions,
source/availability checks and a controlled comparison. Frozen past results
remain private reference evidence and are not rewritten by consolidation.
