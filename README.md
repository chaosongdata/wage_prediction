# wage_prediction
use prophet and Dynamic Linear Model to predict wage hours
#worked wages forecast

#File: wage_forecast.py

Data File:

This is used to predict the worked wages in PR region. There are two parts #[A] Punched Wages (a) Model Function: Output: mixedmodel = estimate_and_predict_dlm_PR + estimate_and_predict_prophet_PR Backtest Function: testpunched (Run this) Utility Function: mape, predict_proportion, proportion, removehurricane

(b) Model: Using Prophet as a long-term model + dlm with sales data as a local trend.

(c) How to use sales data:

  1)	Global Model: Use actual sales data and forecasting result in the testset 
  2)	Local Model: Use proportion (=club sales/total_sales_PR) and forecasting of proportion in testset. 
     Assumption: i) proportion data is more consistent and more predictable
                             ii) in the short period, like test set's period (30-60-90 days)
                                 total sales in the PR region will not change much (??Holiday Season??)
#[B] Sum of Other Categories:

(a) Function: compeltetable . & . testresidualworked (b) Model: Fill empty slot with zero then use Prophet model for individual club. (c) Run (testresidualworked) function directly

#TODO: 1. check more logic part in the modeling, especially regressor part. 2. parametrize to be more automatic 3. consider corner case not considered in modeling process 4. Production code.
