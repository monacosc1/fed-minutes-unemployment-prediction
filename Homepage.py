import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load data
df = pd.read_csv("./data/scraped_data_all_years_true.csv")

def initial_preprocesser(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The CSV file is empty. Please ensure it contains data.")
        
        if 'Text' not in data.columns:
            raise ValueError("The 'Text' column is missing from the CSV file.")
        
        data.Date = data.Date.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
        data.Text = data.Text.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
        
        normal = data[24:]
        need_to_reverse = data[:24]
        need_to_reverse.columns = ["Text", "Date"]
        need_to_reverse = need_to_reverse[["Text", "Date"]]
        
        data = pd.concat([need_to_reverse, normal], ignore_index=True)
        return data

    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file. The CSV file might be empty or corrupted.")
    except FileNotFoundError:
        raise ValueError(f"The file {file_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the CSV file: {str(e)}")


def second_preprocessor(data):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    unrates = pd.read_csv(url)
    
    # Convert UNRATE data to datetime
    unrates['DATE'] = pd.to_datetime(unrates['DATE'], errors='coerce')
    unrates['year'] = unrates.DATE.dt.year
    unrates['month'] = unrates.DATE.dt.month
    unrates = unrates.sort_values('DATE').reset_index(drop=True)
    unrates['UNRATE'] = unrates['UNRATE'].shift(-1)
    
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data.Date, errors='coerce')
    
    # Drop rows where Date couldn't be parsed
    data = data.dropna(subset=['Date']).reset_index(drop=True)
    
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data = data.merge(unrates[['year', 'month', 'UNRATE']], on=['year', 'month'], how='left')
    data = data.rename(columns={"UNRATE": "target"})
    return data


def word_magician(df):
    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'text' column.")
    
    df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
    df["num_chars"] = df["text"].apply(lambda x: len(str(x)))
    df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in ['the', 'a', 'and', 'is', 'of', 'to', 'for']]))
    df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
    return df


def feature_engineering_func(df):
    df["target_shift1"] = df.target.shift(1)
    df["target_shift2"] = df.target.shift(2)
    df["target_shift3"] = df.target.shift(3)
    df = df.dropna().reset_index(drop=True)
    return df

# Streamlit interface
st.title('Unemployment Rate Predictor')

date_input = st.text_input('Enter Date (YYYYMMDD):', '')
text_input = st.text_area('Enter FED Minutes Text:', '')

if st.button('Submit'):
    logging.info(f"Received input: date_input={date_input}, text_input={text_input}")
    
    if date_input and text_input:
        new_input = pd.DataFrame({"Date": [date_input], "Text": [text_input]})
        
        df = initial_preprocesser(df)
        df = pd.concat([df, new_input], ignore_index=True)
        df = second_preprocessor(df)
        df = word_magician(df)
        df = feature_engineering_func(df)
        df.fillna(0, inplace=True)

        # Train model
        train = df[:-1]
        test = df[-1:]
        X = train[['num_words', 'num_chars', 'num_stopwords', 'target_shift1', 'target_shift2']]
        y = train.target
        test_real = test[X.columns]

        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(test_real)[0]

        df.loc[df.index[-1], "target"] = preds
        df["fed_minutes_released_date"] = df.year.astype(str) + "/" + df.month.astype(str)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Plot the last 10 data points plus the prediction
        df_temp = df[-10:].append(df.iloc[-1])  # Include the new prediction
        df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
        fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")

        st.write(f'Unemployment rate prediction for next month is: {preds:.2f}%')
        st.plotly_chart(fig)



