# Results: Algorithmic Working Capital & Margin Optimization

Generated 2026-10-24 00:43. Data window 2017-01-09 to 2018-08-20.
Panel: 118 SKUs, 6,104 SKU-weeks, 8,315 units,
52% zero-sale weeks.

## 1. Unit economics (CM2 = P - COGS - F - 10% fee)
| cat_en | median_price | cogs_share | median_freight | median_cm2_unit | share_cm2_negative |
|---|---|---|---|---|---|
| bed_bath_table | 79.90 | 0.58 | 16.15 | 7.74 | 0.21 |
| computers_accessories | 81.99 | 0.80 | 16.02 | -10.27 | 0.86 |
| telephony | 29.99 | 0.80 | 15.10 | -12.01 | 0.92 |
| watches_gifts | 119.00 | 0.45 | 15.90 | 34.42 | 0.07 |

Low-ticket electronics are **structurally CM2-negative** when the seller absorbs freight:
a BRL 30 phone accessory carries ~BRL 15 freight against a 20% gross margin.

## 2. Forecasting (walk-forward, 3 models, one-step-ahead, refit every 4 weeks)
| model | RMSE | MAE | WAPE | bias_ratio | stockout_loss_brl | excess_holding_brl | safety_stock_capital_brl |
|---|---|---|---|---|---|---|---|
| Seasonal-Naive | 3.149 | 1.591 | 1.048 | -0.035 | 33656.041 | 274.395 | 24244.688 |
| SARIMAX | 3.177 | 1.621 | 1.068 | -0.035 | 32110.797 | 282.842 | 26259.224 |
| LightGBM-Tweedie | 2.359 | 0.879 | 0.579 | -0.088 | 20385.398 | 140.168 | 17543.020 |

Financial reading: moving from Seasonal-Naive to LightGBM-Tweedie cuts forgone CM2 on
stockouts by **BRL 13,271**
(39%) over the hold-out and releases
**BRL 6,702** of safety-stock working capital
(95% cycle-service level, 1-week lead time). Excess-holding cost is two orders of magnitude
smaller than stockout loss: with critical ratio CM2/(CM2+h) near 1, the newsvendor-optimal
policy leans toward over-stocking, and the forecaster's negative bias ratio is the costly error.

## 3. Causal elasticity (FE Double-ML, blocked 5-fold cross-fitting, SKU-clustered SE)
Coefficient on ln P with outcome ln(Q+1):
| cat_en | eps_dml | se_dml | ci_lo | ci_hi | eps_ols_pooled | eps_ols_fe | n_obs | n_skus |
|---|---|---|---|---|---|---|---|---|
| ALL | -0.338 | 0.060 | -0.455 | -0.220 | 0.060 | -0.986 | 6104 | 118 |
| bed_bath_table | -0.569 | 0.176 | -0.913 | -0.224 | 0.093 | -0.688 | 2159 | 40 |
| computers_accessories | -0.177 | 0.101 | -0.374 | 0.021 | 0.172 | -1.015 | 1413 | 32 |
| telephony | -0.113 | 0.094 | -0.297 | 0.072 | 0.037 | -0.489 | 857 | 19 |
| watches_gifts | -0.541 | 0.099 | -0.735 | -0.347 | 0.000 | -1.353 | 1675 | 27 |

Robustness, intensive margin (weeks with Q>0, outcome ln Q):
| cat_en | eps_dml | se_dml | ci_lo | ci_hi | eps_ols_pooled | eps_ols_fe | n_obs | n_skus |
|---|---|---|---|---|---|---|---|---|
| ALL | -0.770 | 0.141 | -1.047 | -0.493 | 0.106 | -1.173 | 2936 | 118 |
| bed_bath_table | -1.383 | 0.383 | -2.133 | -0.632 | 0.092 | -1.254 | 1049 | 40 |
| computers_accessories | -0.440 | 0.244 | -0.918 | 0.037 | 0.266 | -1.017 | 757 | 32 |
| telephony | -0.280 | 0.259 | -0.788 | 0.228 | 0.058 | -0.426 | 354 | 19 |
| watches_gifts | -1.219 | 0.198 | -1.608 | -0.830 | 0.072 | -1.721 | 776 | 27 |

Naive within-SKU OLS overstates price sensitivity (sellers cut prices into Black Friday and
platform-growth peaks, so price and demand co-move negatively for non-causal reasons).
Pooled cross-SKU OLS even has the wrong sign. Pricing off the naive estimate would
systematically over-discount and leak gross margin.

SKU elasticities (log1p-de-attenuated, DerSimonian-Laird shrinkage, sign-restricted):
| cat_en | skus | median_eps | share_elastic | sign_restricted |
|---|---|---|---|---|
| bed_bath_table | 40 | -1.606 | 0.650 | 6 |
| computers_accessories | 32 | -0.323 | 0.156 | 8 |
| telephony | 19 | -0.270 | 0.053 | 6 |
| watches_gifts | 27 | -0.945 | 0.481 | 0 |

## 4. Optimisation: portfolio impact for decision week 2018-08-20 (model-implied, BRL/week)
| scenario | mode | skus | profit_base | profit_star | cm2_base | cm2_star | hold_base | hold_star | wc_freed | dsi_base | dsi_star | price_up | floor_binding | band_overridden | profit_uplift_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aged (12 wks supply) | liquidity | 72 | -6.0 | 528.1 | 599.6 | 1208.4 | 605.6 | 680.3 | -336.4 | 105.0 | 109.0 | 69 | 0 | 0 | 8969.1 |
| aged (12 wks supply) | margin | 72 | 495.4 | 1110.3 | 599.6 | 1215.4 | 104.2 | 105.2 | -383.7 | 105.0 | 111.3 | 72 | 0 | 0 | 124.1 |
| normal (4 wks supply) | liquidity | 72 | 549.7 | 1157.1 | 599.6 | 1217.2 | 49.9 | 60.1 | -380.4 | 35.0 | 37.5 | 72 | 3 | 3 | 110.5 |
| normal (4 wks supply) | margin | 72 | 569.8 | 1186.7 | 599.6 | 1217.5 | 29.9 | 30.8 | -384.7 | 35.0 | 37.5 | 72 | 3 | 3 | 108.3 |

**Lerner discount test.** A price cut raises CM2 only if |eps| > 1 / (CM2 / net revenue).
| cat_en | skus | median_eps | median_breakeven | discount_accretive |
|---|---|---|---|---|
| bed_bath_table | 22 | -1.35 | -8.96 | 0 |
| computers_accessories | 21 | -0.32 | -inf | 0 |
| telephony | 10 | -0.31 | -inf | 0 |
| watches_gifts | 19 | -1.10 | -3.66 | 0 |

Thin CM2 margins make discounting value-destroying for almost every SKU: the
breakeven elasticity is far beyond anything identified in the data. Price cuts only
become rational once carrying cost bites (high cost of capital, heavy overstock), which is
what the liquidity mode and the aged-stock floor relaxation capture.

Margin mode is the spec objective (one week of carry). Liquidity mode charges carrying cost
until leftover stock is sold through, which pushes elastic SKUs toward the floor.
`band_overridden` counts SKUs whose 1.15x COGS floor exceeds +15% of the current price,
where the loss-prevention floor overrides the volatility band.

## 5A. Phase 5A: reinforcement-learning sell-through pricing (26-week season, SELIC 6.5%)

| regime | policy | npv | gap_to_oracle | sell_through | avg_price_ratio | roi |
|---|---|---|---|---|---|---|
| lean (<13 wks) | CE-DP | 325.91 | -0.79 | 0.93 | 1.37 | 0.67 |
| lean (<13 wks) | PPO agent | 326.15 | -0.55 | 0.94 | 1.37 | 0.67 |
| lean (<13 wks) | myopic (Phase 4) | 322.02 | -4.69 | 0.92 | 1.36 | 0.66 |
| lean (<13 wks) | oracle DP | 326.71 | 0.00 | 0.94 | 1.37 | 0.67 |
| lean (<13 wks) | status quo | 75.80 | -250.90 | 0.98 | 1.01 | 0.16 |
| full (13-26 wks) | CE-DP | 499.66 | -4.93 | 0.79 | 1.36 | 0.49 |
| full (13-26 wks) | PPO agent | 497.93 | -6.66 | 0.79 | 1.36 | 0.48 |
| full (13-26 wks) | myopic (Phase 4) | 493.45 | -11.14 | 0.76 | 1.38 | 0.48 |
| full (13-26 wks) | oracle DP | 504.59 | 0.00 | 0.80 | 1.35 | 0.49 |
| full (13-26 wks) | status quo | 111.51 | -393.07 | 0.90 | 1.01 | 0.11 |
| overstock (>26 wks) | CE-DP | 61.83 | -21.58 | 0.55 | 1.34 | 0.04 |
| overstock (>26 wks) | PPO agent | 61.75 | -21.66 | 0.55 | 1.34 | 0.04 |
| overstock (>26 wks) | myopic (Phase 4) | 58.60 | -24.81 | 0.50 | 1.38 | 0.04 |
| overstock (>26 wks) | oracle DP | 83.41 | 0.00 | 0.57 | 1.32 | 0.05 |
| overstock (>26 wks) | status quo | -203.79 | -287.20 | 0.65 | 1.01 | -0.12 |

`gap_to_oracle`: mean NPV shortfall (R$) versus the exact DP that knows the true elasticity. `roi` = total NPV / inventory capital at cost.

## 5B. Phase 5B: GenAI foundation models vs LightGBM (probabilistic, walk-forward)

| model | WAPE_median | WQL | coverage_80 | coverage_q90 | stockout_loss_brl | excess_holding_brl | fill_rate | avg_capital_brl_wk | wc_at_risk_90_brl_wk |
|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt (zero-shot) | 0.856 | 0.573 | 0.810 | 0.886 | 7254.311 | 1188.747 | 0.889 | 27914.737 | 17666.908 |
| Chronos-2 + covariates (zero-shot) | 0.870 | 0.570 | 0.690 | 0.893 | 5745.089 | 1178.386 | 0.901 | 27844.670 | 17603.358 |
| LightGBM quantile | 0.562 | 0.377 | 0.843 | 0.922 | 8595.613 | 698.793 | 0.880 | 19666.735 | 9036.571 |

Order-up-to-q90 newsvendor backtest. `avg_capital_brl_wk` = mean weekly stock value at COGS; `wc_at_risk_90` = sum (q90 - q50) x COGS, the capital buying the last 40 points of service level. `coverage_q90` should be ~0.90 if the q90 is calibrated.


## 5C. Phase 5C: LLM-coded customer voice in the causal model

DML elasticity with and without review-derived confounders:

| cat_en | eps_dml | se_dml | eps_dml_with_reviews | se_dml_with_reviews |
|---|---|---|---|---|
| ALL | -0.338 | 0.060 | -0.446 | 0.071 |
| bed_bath_table | -0.569 | 0.176 | -0.496 | 0.139 |
| computers_accessories | -0.177 | 0.101 | -0.423 | 0.090 |
| telephony | -0.113 | 0.094 | -0.070 | 0.081 |
| watches_gifts | -0.541 | 0.099 | -0.789 | 0.126 |

Moderation: change in the ln(Q+1) price coefficient per 1 SD of each review signal (positive = less price-sensitive). Four moderators are tested, so the Bonferroni 5% threshold is |z| > 2.50:

| moderator | theta0 | theta0_se | theta1_per_sd | theta1_se | z_stat |
|---|---|---|---|---|---|
| value_perception | -0.436 | 0.066 | 0.110 | 0.083 | 1.334 |
| voice_quality_bad | -0.446 | 0.066 | 0.168 | 0.075 | 2.251 |
| voice_delivery_late | -0.437 | 0.072 | -0.039 | 0.079 | -0.492 |
| voice_sentiment | -0.448 | 0.069 | -0.063 | 0.075 | -0.844 |

LLM validation against star ratings (which the LLM never saw): sentiment matches the star polarity for 87.7% of 1-2 and 4-5 star reviews. Row shares by star rating:

| review_score | -1 | 0 | 1 |
|---|---|---|---|
| 1 | 0.95 | 0.04 | 0.01 |
| 2 | 0.77 | 0.18 | 0.04 |
| 3 | 0.52 | 0.28 | 0.20 |
| 4 | 0.13 | 0.20 | 0.67 |
| 5 | 0.02 | 0.05 | 0.93 |
