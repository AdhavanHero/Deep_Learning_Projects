# 📈 Stock Market Prediction using RNN with LSTM

An end-to-end Deep Learning project that predicts the closing stock price of Titan Company Ltd. (NSE: TITAN) using historical market data and a Stacked Long Short-Term Memory (LSTM) neural network.

## 🚀 Project Overview
Time-series forecasting is a complex predictive modeling problem. This project fetches 5 years of live historical stock data using the Yahoo Finance API, preprocesses it, and trains a sequential Deep Learning model to predict future price points based on the previous 100 days of trading data.

## 🛠️ Technologies & Libraries Used
* **Python 3.x**
* **TensorFlow / Keras** (Deep Learning architecture)
* **yfinance** (Live market data fetching)
* **Scikit-Learn** (Data normalization)
* **Pandas & NumPy** (Data manipulation)
* **Matplotlib** (Data visualization)

## 🧠 Model Architecture
The model uses a **Stacked LSTM** structure which is highly suited for finding patterns in sequential data:
1. **LSTM Layer 1:** 50 units, `return_sequences=True`
2. **LSTM Layer 2:** 50 units, `return_sequences=True`
3. **LSTM Layer 3:** 50 units
4. **Dense Layer:** 1 unit (Output predicting the scaled price)
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

## 📊 Workflow
1. **Data Extraction:** Automated fetching of 5-year data for `TITAN.NS` via Yahoo Finance.
2. **Feature Engineering:** Extracted the `Close` price and normalized it to a `(0, 1)` range using MinMax Scaling.
3. **Sequence Generation:** Transformed the 1D array into sequential data using a time-step of 100 days.
4. **Train/Test Split:** Chronological split of 75% training data and 25% testing data.
5. **Model Training:** Trained the LSTM over 100 epochs.

## 💻 How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
