# Predicting Unemployment from FOMC Statements

NLP-based model that extracts features from Federal Reserve policy statements to predict the next month's unemployment rate, deployed as a Streamlit web app.

## Key Finding

![Unemployment Rate History](charts/fed-minutes-fig1-unemployment-rate-history.png)

*Historical U.S. unemployment rate showing the cyclical patterns the model aims to predict from FOMC statement text.*

## Overview

Federal Reserve meeting minutes contain forward-looking language that reflects the committee's assessment of labor market conditions. This project uses natural language processing to extract sentiment scores and word count features from FOMC statements, then trains a regression model to predict the following month's unemployment rate. The model achieves a mean absolute error of 0.49 percentage points, demonstrating that Fed communications carry meaningful predictive signal for labor market outcomes.

## Tools & Technologies

- Python
- Pandas
- scikit-learn
- NLTK
- Streamlit
- Matplotlib

## Results

FOMC statement sentiment and word count features predict the unemployment rate with a MAE of 0.49 percentage points:

![Prediction Accuracy](charts/fed-minutes-fig4-prediction-accuracy.png)

*Model predictions vs. actual unemployment rate, showing close tracking across both stable and volatile periods.*

## Live App

Try the interactive prediction tool: [Streamlit App](https://fed-minutes-unemployment-prediction-dg3k7ksudiphejwegmb4ja.streamlit.app/)

## View Full Analysis

For the complete writeup with all charts and methodology, visit the [project page on scottmonaco.com](https://scottmonaco.com/fed-minutes).
