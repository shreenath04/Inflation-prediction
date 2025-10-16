# Inflation Prediction (1983–2023)
**Author:** Shreenath Gandhi  
**Tech Stack:** Python, Pandas, Scikit-learn, Seaborn, Matplotlib

## 🎯 Objective
Can pre-crisis monetary and macroeconomic indicators (interest rates, GDP, unemployment) predict future U.S. inflation trends?

## 📊 Data Pipeline
- **Source:** Federal Reserve Economic Data (FRED)
- **Preprocessing:** 
  - Removed pre-1983 data for stability
  - Created synthetic target rate (post-2008 range midpoints)
  - Annualized GDP, unemployment, inflation
  - Engineered “Deviation” between effective and target rates

## 🧠 Model
- Two **Random Forest regressors** trained on:
  - Positive drivers (Fed targets, GDP, etc.)
  - Negative drivers (unemployment, deviations)
- Ensemble weighted by R² and inverse MSE
- Regime-based flags for tightening (+1) / easing (-1) cycles

## 📈 Results
- R² ≈ **0.63** on historical inflation (1983–2017)
- Directionally stable forecasts for 2018–2023
- Captures structural trends and regime shifts

## 🧩 Key Insights
- Inflation is highly sensitive to deviation between target and effective rates.
- Post-2008 monetary regimes behave differently—captured by feature flags.
- Ensemble blending balances demand and supply-side effects.

## 📷 Visuals
<img width="1148" height="574" alt="image" src="https://github.com/user-attachments/assets/002ee22f-9814-487d-9556-91de6a2efe3c" />


## 🧮 Tools
- pandas, numpy, matplotlib, seaborn
- scikit-learn (RandomForestRegressor, LOOCV)
- plotly (optional, for interactive visuals)

## Thought process and reasoning behind decisions
This project wasn’t built in one pass. Below is the decision log—what we tried, why we did it, what broke, and how we fixed it.

1) Data scope & stability
- Decision: Start at 1983 and drop earlier rows.
  
Why: The early 80s are a volatile transition out of the high-inflation 70s. Starting in 1983 gives a more stable regime while preserving enough cycles for learning.
Pitfall avoided: Including pre-1983 amplified noise and regime breaks that hurt generalization.


2) Targets vs effective rate (policy mechanics)
- Decision: Build a Synthetic Target Federal Funds Rate (SFFTR):
- 1983–2008: use the single Target Rate.
- 2008–2017: use the midpoint of the Target Range (avg of upper & lower).

Why: Post-2008 policy is expressed as a range. The midpoint is the most defensible single number.
Related feature: Deviation = Effective – SyntheticTarget (signed), plus Deviation_abs during exploration.
- What we learned: Targets matter because they encode policy stance, while Effective reflects market/operations. The gap (Deviation) adds information beyond either series alone.


3) Handling missing values
- Decision:
- Do not impute GDP/Inflation for modeling (we later either dropped or interpolated minimally and documented it).
- Do not over-impute EFFR; for annual analysis we aggregate instead of filling every missing daily/monthly cell.

Why: We model at the annual level; filling sub-annual gaps creates false certainty and can bias aggregates.


5) Annual aggregation (fixing the “wild %” bug)
- Initial mistake: We compounded values that were already annualized or duplicated across months, creating absurd results (e.g., GDP 35%, CPI 70%).
- Fix:
  - If the source is already annualized (common for GDP growth and often for CPI YoY), use mean/median, not compounding.
  - If true sub-annual % changes, deduplicate to one value per period before compounding.
  - Added a safe aggregator that chooses mean for annualized series and product only for true sub-annual, de-duplicated changes.
Outcome: Realistic ranges: GDP ~ −3% to +7%, CPI ~ 1% to 5%, Unemployment ~ 4% to 10%.


6) Event diagnostics & interpretation
- Observation: Large deviation spikes/dips around 1987 and 2008; other blips mid-90s and early-2000s.
- Decision: Keep Deviation and Deviation_abs (during EDA) to see stress.
- Interpretation: Not all big deviations = “crisis.” In 2008–09, policy cut rates to zero (small deviation but extreme context).
- Conclusion: Do not redefine “crisis” by deviation magnitude alone; treat deviation as a continuous signal and crisis as a context flag.


8) Crisis flags: static vs dynamic
- Initial idea: Dynamic is_crisis from deviation spikes.
- Decision: Keep is_crisis static (e.g., 1987–88, 2001–02, 2008–09).
  
Why: Preserves interpretability and consistency across analyses.
- Refinement: Add regime3 (−1 easing crisis, 0 normal, +1 tightening crisis) using crisis flag and deviation sign (≤ small negative threshold → easing).
- Benefit: Encodes direction of policy stress without moving the goalposts.


9) Final feature set (annual, interpretable)
- Core columns: Year, Synthetic_Target_Rate, Deviation, Real GDP (Percent Change), Unemployment Rate, Inflation Rate.
- Context flags: is_post_2008, is_crisis, and regime3 (−1/0/+1).
- Rationale: Compact, macro-intuitive set that captures stance (target), slippage (deviation), real activity (GDP), slack (unemployment), and regime.


10) Modeling strategy (small-sample, interpretable)
- Baseline: Linear models for sanity checks.
- Final approach: Directional Ensemble (“positive vs negative drivers”):
- Positive model (forest): Synthetic_Target_Rate, Real GDP, Deviation, is_post_2008, regime3.
- Negative model (forest): Unemployment, Deviation, is_post_2008, regime3.
- Ensemble weights: R² and inverse MSE from LOOCV (out-of-fold).

Why: Mirrors macro logic (inflationary vs disinflationary forces) while staying data-driven.
- LOOCV: Chosen due to ~35 annual samples. Uses nearly all data while measuring generalization per point.

11) Things tried that didn’t move much (and why)
- Time-weighted training (RF): Sample weights had limited effect—RF splits were similar under smooth Great-Moderation dynamics; ensemble dominated by the positive model.
- GBRT / HistGBR: Considered for stronger response to weights; required NaN handling (we used HistGBR for NaN tolerance). Kept RF for clarity and stability in the final repo.
- Dynamic crisis from deviation magnitude: Conceptually neat, but confused semantics; we kept crisis static and used deviation separately.

12) Post-2017 forecasting & limitations
- GDP & Inflation forecasts (2018–2023): We used the trained structure to project out-of-sample.
- Reality check: The model misses COVID-era extremes (2020–2021) because no such shock exists in training (1983–2017).
- Mitigation: Mark pandemic years as is_crisis=1 and set regime3 (easing). Still, structural breaks remain hard without new features (e.g., Fed balance sheet %, supply-chain stress, energy shocks).
- Honest positioning: Forecasts are directional/structural, not point-perfect during shocks.

14) Reproducibility & repo hygiene
- Kept a minimal requirements.txt (pandas/numpy/sklearn/matplotlib/seaborn/statsmodels/plotly).
- Separated notebooks:
- Data Preparation (cleaning, aggregation, features)
- Prediction (modeling, LOOCV, plots)
- Saved final dataset for quick starts and review.

⸻

Key takeaways
- Don’t compound what’s already annualized (fixes wild % errors).
- Model stance, slippage, and regime: target rate, deviation, and flags jointly matter.
- Directional ensemble is a practical way to encode macro intuition: expansionary vs contractionary forces.
- Be explicit about limitations around structural breaks (COVID/2008-style shocks).

This transparent decision trail is the backbone of the project’s credibility—and why the final model is both explainable and reproducible.
