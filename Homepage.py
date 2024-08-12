import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Function to load and preprocess the initial data
def initial_preprocesser(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    return data

# Function to load and preprocess the UNRATE data from the URL
def second_preprocessor(data):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    unrates = pd.read_csv(url)
    unrates["DATE"] = pd.to_datetime(unrates["DATE"])
    unrates["year"] = unrates["DATE"].dt.year
    unrates["month"] = unrates["DATE"].dt.month
    unrates = unrates[["year", "month", "UNRATE"]]
    data = pd.merge(data, unrates, on=["year", "month"], how="left")
    data = data.rename(columns={"UNRATE": "unrate", "Text": "text", "Date": "date"})
    return data

# Function to calculate word features
def word_magician(df):
    df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
    df["num_chars"] = df["text"].apply(lambda x: len(str(x)))
    df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in ['the', 'a', 'and', 'is', 'of', 'to', 'for']]))
    df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
    return df

# Function to apply feature engineering
def feature_engineering_func(df):
    df = df.rename(columns={"unrate": "target"})
    df["target_shift1"] = df["target"].shift(1)
    df["target_shift2"] = df["target"].shift(2)
    df["target_shift3"] = df["target"].shift(3)
    df = df.dropna().reset_index(drop=True)
    return df

# Main Script
st.title('Unemployment Rate Predictor')

# Load initial data
csv_file_path = "./data/scraped_data_all_years_true.csv"

df = initial_preprocesser(csv_file_path)
df = second_preprocessor(df)

date_input = st.text_input('Enter Date (YYYYMMDD):', '')
text_input = st.text_area('Enter FED Minutes Text:', '')

if st.button('Submit'):
    if date_input and text_input:
        new_input = pd.DataFrame({"Date": [pd.to_datetime(date_input)], "Text": [text_input]})
        df = pd.concat([df, new_input], ignore_index=True)
        df = second_preprocessor(df)
        df = word_magician(df)
        df = feature_engineering_func(df)
        df.fillna(0, inplace=True)

        # Train model
        train = df[:-1]
        test = df[-1:]
        X = train[['num_words', 'num_chars', 'num_stopwords', 'target_shift1', 'target_shift2']]
        y = train["target"]
        test_real = test[X.columns]

        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(test_real)[0]

        df.loc[df.index[-1], "target"] = preds
        df["fed_minutes_released_date"] = df["year"].astype(str) + "/" + df["month"].astype(str)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Plot the last 10 data points plus the prediction
        df_temp = df[-10:]
        df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
        fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")

        st.write(f'Unemployment rate prediction for next month is: {preds:.2f}%')
        st.plotly_chart(fig)








# import pandas as pd
# import streamlit as st
# import logging
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import plotly.express as px

# # Set up basic logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )

# # Load the initial dataset
# df = pd.read_csv("./data/scraped_data_all_years_true.csv")

# def initial_preprocesser(data):
#     data.Date = data.Date.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
#     data.Text = data.Text.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
#     normal = data[24:]
#     need_to_reverse = data[:24]
#     need_to_reverse.columns = ["Text", "Date"]
#     need_to_reverse = need_to_reverse[["Text", "Date"]]
#     data = pd.concat([need_to_reverse, normal], ignore_index=True)
#     return data

# def second_preprocessor(data):
#     url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
#     unrates = pd.read_csv(url)
    
#     unrates.UNRATE = unrates.UNRATE.shift(-1)
#     unrates["DATE"] = pd.to_datetime(unrates["DATE"])
#     unrates["year"] = unrates.DATE.dt.year
#     unrates["month"] = unrates.DATE.dt.month
#     unrates = unrates.sort_values("DATE").reset_index(drop=True)
#     unrates.drop(["DATE"], axis=1, inplace=True)
    
#     data["Date"] = pd.to_datetime(data.Date)
#     data["year"] = data.Date.dt.year
#     data["month"] = data.Date.dt.month
#     data = data.sort_values("Date").reset_index(drop=True)
#     data = data.merge(unrates, on=["year", "month"], how="left")
#     data.columns = ["text", "date", "year", "month", "unrate"]
#     return data

# def word_magician(df):
#     df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
#     df["num_chars"] = df["text"].apply(lambda x: len(str(x)))
#     df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in ['the', 'a', 'and', 'is', 'of', 'to', 'for']]))
#     df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
#     return df

# def feature_engineering_func(df):
#     df = df.rename(columns={"unrate": "target"})
#     df["target_shift1"] = df.target.shift(1)
#     df["target_shift2"] = df.target.shift(2)
#     df["target_shift3"] = df.target.shift(3)
#     df = df.dropna().reset_index(drop=True)
#     return df

# # Streamlit app layout
# st.title('Unemployment Rate Predictor')
# st.write('Enter the date and the FED minutes text to predict the unemployment rate for the next month.')

# date_input = st.text_input('Enter Date (YYYYMMDD):', '')
# text_input = st.text_area('Enter FED Minutes Text:', '')

# if st.button('Submit'):
#     logging.info("Processing new input...")
#     new_input = pd.DataFrame({"Date": [date_input], "Text": [text_input]})
#     df = initial_preprocesser(df)
#     logging.info("Initial preprocessing completed.")
#     df2 = pd.concat([df, new_input], ignore_index=True)
#     df2 = second_preprocessor(df2)
#     logging.info("Second preprocessing completed.")
#     df2 = word_magician(df2)
#     logging.info("Word magician function completed.")
#     df2 = feature_engineering_func(df2)
#     df2.fillna(0, inplace=True)

#     train = df2[:-1]
#     last_month_unemployment_rate = train.iloc[-1].target
#     test = df2[-1:]
#     X = train[['num_words', 'num_chars', 'num_stopwords', 'target_shift1', 'target_shift2']]
#     y = train.target
#     test_real = test[X.columns]
    
#     # Very simple linear regression model
#     model = LinearRegression()
    
#     model.fit(X, y)
#     preds = model.predict(test_real)[0]
#     logging.info("Model prediction complete. Prediction: %s", preds)
    
#     df2.loc[(df2.date == date_input), "target"] = float(preds)
#     df2["fed_minutes_released_date"] = df2.year.astype(str) + "/" + df2.month.astype(str)
#     df2 = df2.sort_values("date").reset_index(drop=True)
#     df_temp = df2[-10:]
#     df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
#     fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")
    
#     st.write(f'Unemployment rate prediction for next month is: {preds:.2f}%')
#     st.plotly_chart(fig)
