# Walmart Retail Data Analysis and Sales Forecasting


## About Dataset

 Historical sales data for 45 Walmart stores located in different regions are available. There are certain events and holidays which impact sales on each day. The business is facing a challenge due to unforeseen demands and runs out of stock some times, due to inappropriate machine learning algorithm. Walmart would like to predict the sales and demand accurately.


 ## 1. Exploratory Data Analysis

1.  I visualized the Total Sales by Store (Cumulative Over Weeks), and discovered that store 20 had the highest sales of $301,397,800 (2010-2012)
2.  I also explored the standard deviation of the stores to find out how far of sales deviate from each store's average sales.
3.  I also calculated the Coefficient of variation (CV) for all stores. CV is a useful metric when you want to compare variability across datasets that may have different units or scales.
4.  The quaterly growth rate (Q3) for 2012 was also calculated to find out which stores had the best growth rate.

**Stores that have good Q3 growth rate in 2012**

- Store 7  - 13.33%
- Store 16 - 8.49%
- Store 35 - 4.47%
- Store 26 - 3.96%
- Store 39 - 2.48%
- Store 41 - 2.46%
- Store 44 - 2.43%
- Store 24 - 1.65%
- Store 40 - 1.14%
- Store 23 - 0.83%

5. I also pulled out Holiday Weeks with Higher Sales than Non-Holiday Mean, knowing which holidays have higher sales.

6. **Monthly Trend Analysis:**  I grouped the weekly sales into months and years and plotted using a line plot to get insights. Here are some of the insights I got:

- There is a consistent peak in sales around december from 2010 to 2011, this could be mainly as a result of the Christmas Holidays.
- There is also a consistent spike in sales around April from 2010 -2012 and then a dip of sales in May,
- The month of July has high sales all through from 2010-2012

7. **Semester Trend Analysis:** I grouped the weekly sales into semester periods (6 months) and plotted a line plot to get insights. I discovered that:

- There is a consistent increase in sales in the second half (H2) from 2010-2011, but there is a huge decline on sales in 2012, and this could be beacuse the 2012 sales record ends in October and not December, which usually accounts for much of the sales.
- In general, H2 tends to have higher sales than H1.



## Prediction Model to Predict Demand for Store

### Feature Engineering

- I created new features like Month_sin', 'Month_cos', 'Week_sin', 'Week_cos',
    'Is_Is_Thanksgiving', 'Is_Is_SuperBowl', 'Is_Is_LaborDay', 'Is_Is_Christmas',
    'Monthly_Avg_Sales', 'Lag_52', 'Is_Pre_Is_Thanksgiving', 'Is_Pre_Is_Christmas', 'Is_Pre_Is_SuperBowl', 'Is_Pre_Is_LaborDay', Lag_4, Lag_8, MA_4, MA_8 etc to increase the prediction strength of the model.

**Here are some intresting things I did during the feature engineering:**

<b> 1. Cyclic Encoding for Month & Week </b>
<p> Why? Month and week are cyclical — December is followed by January, and Week 52 wraps to Week 1. </p>
<b> Purpose of `week_sin`, `week_cos`, `month_sin`, and `month_cos`:</b>

- These are **cyclical (or periodic) features** used to model **seasonality** in time series data — especially for algorithms like XGBoost or linear regression that don't natively understand the cyclical nature of time.

<b> The Problem with Raw `Month` or `Week`:</b>

- Raw `Month` or `WeekOfYear` values (e.g., January = 1, December = 12) **falsely imply an order and distance**.

<b>For example: </b>

- December (12) and January (1) are only 1 month apart in time…
- But numerically, they’re 11 units apart! 

<b> This breaks machine learning models’ understanding of **cyclical patterns** like:</b>

- Weekly sales cycles (e.g., sales spike every Friday or in Week 52)
- Monthly/seasonal trends (e.g., December holiday surge)

<b> What This Helps Capture </b>

These features help the model learn:

* **Seasonal cycles** (e.g., yearly shopping patterns)
* **Weekly trends** (e.g., Black Friday is always in Week 47–48)
* **Smooth transitions** (December → January is no longer a "jump")


* `sin` and `cos` together help the model "see" the **position on the cycle**, not just a flat number.
* It’s like saying “we’re at 3 o’clock” instead of “we’re at minute 15.”

- <b>Visualization on how it works:</b> https://chatgpt.com/s/t_688b45c5fbcc819185fb73fc23b126e2 

- Using sin and cos lets the model understand this circular nature.
- Otherwise, the model might think January (1) and December (12) are far apart.</p>

<br>

<b> 2. Why "Pre-Holiday Weeks" Work Well </b>
<p>Consumer behavior spikes before holidays:</p>

- For example, people buy more in the week(s) before Christmas, Thanksgiving, or Super Bowl.
- Your original model didn’t know this — it treated every week the same.
- Now it sees these spikes coming and adjusts its predictions.
- Pre-holiday patterns are repeatable and predictable every year, which helps the model generalize well.
- You're giving the model more temporal context, similar to how humans think ("It's the week before Christmas, so sales should rise").


<b> 3. Monthly Average Sales </b>

- Adds a new column Monthly_Avg_Sales showing the average sales for each month (year-specific).
- This gives the model context — e.g., how this week’s sales compare to the typical monthly average.

<b> 4. Lag Feature — Previous Year’s Sales </b>

- Adds a lag feature: sales from exactly 52 weeks ago (same week, previous year).
- Helps model yearly seasonality. For instance, if Week 10 of 2011 had high sales, maybe Week 10 of 2012 will too.

<b> 5. Drop Rows with Missing Values </b>

- The first 52 rows (from the lag feature) will have missing values, so they are removed to avoid the model from crashing



### Using XGBoost (Extreme Gradient Boosting)

- XGBoost is a powerful machine learning algorithm used for regression, classification, and ranking problems. It's one of the most accurate, efficient, and fast algorithms for tabular data suitable for the walmart dataset.</p>
- At first, using the XGBoost model with features like 'Month', 'Week', 'Holiday_Flag', 'CPI', 'Unemployment', 'Fuel_Price', 'Lag_1', 'Lag_2', 'MA_4'
gave a very bad r2 score ("-0.51"). So I worked on imoroving it's prediction power

#### Things done to improve the prediction accuracy of the XGBoost model**

1. **Checking Feature importance:** to know you how much each input feature contributes to the prediction of the model.
2. **Checking Permutation Importance:** to know how much the prediction accuracy is affected when each feature is shuffled. Features with low permutaion importance were removed. The model's RMSE and r2 score improved to $87,072.35 and R² Score: of -0.4587 respectively afterwards.
3. **Hyperparameter Tuning:** to find the best combination of settings (called hyperparameters) that control how the XGBoost model learns from the walmart data. After the hyperparameter tuning, the model's performance increased thus: RMSE: 80949.42 and R² Score: -0.2607
4. **Retraining the XGBoost model, taking into account Seasonality:** After some research, I discovered that my model wasn't taking into account the seasonal trends in the dataset and thus not predicting the weekly sales as it should, so I engineered some new features like Month_sin', 'Month_cos', 'Week_sin', 'Week_cos', 'Is_Is_Thanksgiving', 'Is_Is_SuperBowl', 'Is_Is_LaborDay', 'Is_Is_Christmas', 'Monthly_Avg_Sales', 'Lag_52', 'Is_Pre_Is_Thanksgiving', 'Is_Pre_Is_Christmas', 'Is_Pre_Is_SuperBowl', 'Is_Pre_Is_LaborDay'.   I did this so the XGBoost model could be more informed of seasonality trends.

- After retraining the model taking into account seasonality, the model's performance improved drastically to an **RMSE of 46,017.19** and a **R² Score of 0.5926!**


## Linear Regression Model

<p> I trained a Linear Regression Model using features like CPI', 'Unemployment', 'Fuel_Price', 'Month', 'Year', 'WeekOfYear', 'IsHoliday',
'Lag_1', 'Lag_2', 'MA_4', 'MA_8' to hypothesize if CPI, unemployment, and fuel price have any impact on sales.</p>

- The RMSE using Linear Regression was 135,392.49 and the R² Score using Linear Regression was 0.3287. These numbers don't represent isn't the best prediction accuracy, but I only used the linear regression model to find determine the coefficients of CPI, Unemployment and Fuel price and it's effect of weekly sales.
- The three had positive effects on sales, more insights of these results and theor interpretation in the notebook.












































