# HomeValue AI: House Price Prediction App

## Project Overview

HomeValue AI is an end-to-end Data Science and Machine Learning demo project.

The project uses housing data to train a regression model that estimates house sale prices from property features.

## Dataset

This project uses the Kaggle House Prices dataset.

Dataset link:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv

For Streamlit deployment, place `train.csv` in the same GitHub repository folder as `app.py`.

## Business Use Case

Real estate agents, homeowners, developers, and analysts often need quick property price estimates. This app demonstrates how a machine learning model can support that decision-making process.

## Workflow

1. Load housing dataset
2. Inspect and clean data
3. Perform exploratory data analysis
4. Create visualizations
5. Engineer useful features
6. Train regression models
7. Evaluate model performance
8. Save final model
9. Build Streamlit app
10. Deploy app online

## Files

- `house_price_prediction_end_to_end_live_demo.ipynb` — full notebook workflow
- `app.py` — Streamlit web app
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python version for deployment
- `train.csv` — Kaggle training data used by the app if the saved model cannot load
- `house_price_model_pipeline.pkl` — saved model generated after running the notebook
- `house_price_app_metadata.pkl` — metadata generated after running the notebook

## Important Note

This app is a teaching prototype. It should not be used as a professional valuation tool without further validation using current local market data.
