# Building-Energy-Anomaly-Detection
https://www.kaggle.com/competitions/energy-anomaly-detection/overview

This is a Kaggle competition I worked on to practice detecting time-series anomalies. The purpose of the competition is to predict when individual buildings have anomalous energy meter readings. My solution was to use traditional classification approaches to predict anomalous energy meter readings in combination with a time series forecast to help detect moments where the measured meter reading deviated significantly from what was expected. This combined approach leads to a top 10 solution in the Kaggle leaderboard.

Some of the key features of solution are summarized below:

1. Creating a Prophet-based time series forecast to improve anomaly detection

By creating a forecasted meter reading and comparing it to the actual meter reading allows for the creation of a residual value between predicted and actual readings, and using this I created a residual z-score for each meter reading for each building which was the second most important feature in predicting anomalous values

2. Adding temporal features

Using lagged meter readings with varying shift steps (from 1 hour to 168 hours) create features that helps the model detect unusual meter readings

3. Combining these approaches with traditional feature engineering and data cleaning

Using all three of these approaches on an XGBoost classifier leads to an AUC that places in the top 10 of the Kaggle competition leaderboard
