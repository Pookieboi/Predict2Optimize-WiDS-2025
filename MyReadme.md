# Financial Time Series Analysis & Prediction

## About This Project
This repo contains my work on analyzing financial data and trying to predict stock returns without "cheating."

The biggest challenge in financial ML is **look-ahead bias**—accidentally using data from tomorrow to predict today. If you just scramble your data like you would for image classification, you get amazing results that fail completely in the real world. This project is all about doing it the hard way: strictly using past data to predict the future.

## What's Inside?

I split the work into two main notebooks:

**1. task1.ipynb (The Setup)**
This is where I did all the initial digging. I pulled historical data for big tech companies (Apple, Nvidia, etc.) from 2015 to 2024. The main focus was cleaning up the data and calculating things like **Log Returns** and **Rolling Volatility**. You can't really model raw stock prices because they drift all over the place (non-stationary), so converting them to returns first was a key step.

**2. task2.ipynb (The Modeling)**
This is the actual prediction engine. I built a few models to guess future returns:
* **Baselines:** A "Zero Predictor" and a simple moving average. I needed these to see if my ML models were actually learning anything useful.
* **ML Models:** I tried Linear Regression, Ridge, and Random Forest.
* **Validation:** Instead of a random train-test split, I used **Walk-Forward Validation**. It trains on a window of time and tests on the specific period right after it, moving forward step-by-step.

## Technologies
I used Python 3 for everything.
* **yfinance** to grab the stock history.
* **Pandas & NumPy** to crunch the numbers.
* **Matplotlib** for the charts.
* **Scikit-Learn** for the modeling and validation logic.

## Key Takeaways
Working on this taught me a few specific things:
* **Don't trust random splits.** If you randomly split time-series data, your model effectively sees the future. You have to respect the timeline.
* **Noise is high.** Honest validation shows that it's actually really hard to beat a boring "zero return" baseline.
* **Stationarity matters.** Raw prices are messy; returns are much easier for statistical models to handle.