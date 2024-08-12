import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load data
df = pd.read_csv("./data/scraped_data_all_years_true.csv")

# Define the functions (assuming they are defined elsewhere in your script)
def initial_preprocesser(file_path):
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Ensure 'Date' and 'Text' columns exist
        if 'Date' not in data.columns or 'Text' not in data.columns:
            raise ValueError("The CSV file must contain 'Date' and 'Text' columns.")
        
        # Clean the 'Date' and 'Text' columns
        data['Date'] = data['Date'].apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
        data['Text'] = data['Text'].apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
        
        # Fix any ordering issues if needed
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
    
    # Convert DATE to datetime and handle parsing errors
    unrates['DATE'] = pd.to_datetime(unrates['DATE'], errors='coerce')
    
    # Check if there are any missing dates
    if unrates['DATE'].isnull().any():
        raise ValueError("Some dates could not be parsed in the UNRATE data.")
    
    unrates['year'] = unrates.DATE.dt.year
    unrates['month'] = unrates.DATE.dt.month
    unrates = unrates.sort_values('DATE').reset_index(drop=True)
    
    # Shift the UNRATE values
    unrates['UNRATE'] = unrates['UNRATE'].shift(-1)
    
    # Merge with the main data
    data['Date'] = pd.to_datetime(data.Date)
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data = data.sort_values('Date').reset_index(drop=True)
    data = data.merge(unrates[['year', 'month', 'UNRATE']], on=['year', 'month'], how='left')
    data.columns = ["text", "date", "year", "month", "unrate"]
    
    return data

# Main Script
st.title('Unemployment Rate Predictor')

# Load initial data
csv_file_path = "./data/scraped_data_all_years_true.csv"
try:
    df = initial_preprocesser(csv_file_path)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Continue processing if no errors
try:
    df = second_preprocessor(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

# The rest of your code continues here, assuming df is now correctly loaded and preprocessed
st.write("Data loaded and preprocessed successfully.")

# Input fields for user input
date_input = st.text_input('Enter Date (YYYYMMDD):', '')
text_input = st.text_area('Enter FED Minutes Text:', '')

if st.button('Submit'):
    st.write(f"Received input: date_input={date_input}, text_input={text_input}")
    
    if date_input and text_input:
        new_input = pd.DataFrame({"Date": [date_input], "Text": [text_input]})
        
        # The data is already preprocessed at this point, so no need to call initial_preprocesser(df) again
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
        df_temp = df[-10:].append(df.iloc[[-1]], ignore_index=True)  # Include the new prediction
        df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
        fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")

        st.write(f'Unemployment rate prediction for next month is: {preds:.2f}%')
        st.plotly_chart(fig)