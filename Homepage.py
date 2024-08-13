import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from datetime import datetime

# Function to scrape or load FOMC statements
def load_fomc_statements():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all the links to the HTML statements
    links = soup.find_all('a', href=True, string='HTML')
    
    if not links:
        print("No links found. The page structure might have changed.")
        return None
    
    statements = []
    dates = []
    
    for link in links:
        try:
            # The link is relative, so we need to create the full URL
            full_url = requests.compat.urljoin(url, link['href'])
            
            # Visit each FOMC meeting page
            meeting_response = requests.get(full_url)
            meeting_soup = BeautifulSoup(meeting_response.content, 'html.parser')
            
            # Extract the meeting date
            date_tag = link.find_previous('td')
            if date_tag:
                date_text = date_tag.get_text().strip()
            else:
                date_text = "Unknown Date"
            
            # Extract the statement text, assuming it is within specific tags
            content_div = meeting_soup.find('div', class_='col-xs-12 col-sm-8 col-md-9')
            if content_div:
                paragraphs = content_div.find_all('p')
                statement_text = ' '.join([para.get_text().strip() for para in paragraphs])
            else:
                statement_text = "No statement text found."
            
            # Store the scraped data
            statements.append(statement_text)
            dates.append(date_text)
        except Exception as e:
            print(f"An error occurred while processing link {full_url}: {e}")
            continue
    
    if not statements or not dates:
        print("No statements or dates could be scraped. Please check the scraping logic.")
        return None
    
    # Create a DataFrame with the date and statement text
    df_statements = pd.DataFrame({
        'Date': pd.to_datetime(dates, errors='coerce'),
        'Statement': statements
    })
    
    return df_statements

# Function to extract dates and clean the statements
def extract_date_from_statement(statement):
    # Use regex to find date patterns in the statement
    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}[-–]?\d{0,2}?,?\s\d{4}', statement)
    
    if date_match:
        date_str = date_match.group()
        try:
            # Try parsing for full date with day range (e.g., "January 30–31, 2024")
            if "–" in date_str or "-" in date_str:
                date_str = re.sub(r'[-–]\d{1,2}', '', date_str)  # Remove the day range
            return datetime.strptime(date_str, '%B %d, %Y')
        except ValueError:
            try:
                # Handle cases where only one day is present without a range
                return datetime.strptime(date_str, '%B %d, %Y')
            except ValueError:
                # Handle cases where month and year are present without a specific day
                return datetime.strptime(date_str, '%B %Y')
    else:
        return pd.NaT

def clean_statements_df(statements_df):
    # Extract dates from statements
    statements_df['Date'] = statements_df['Statement'].apply(extract_date_from_statement)
    
    # Ensure that we capture all non-null dates
    statements_df = statements_df.dropna(subset=['Date'])
    
    # Add year and month columns based on the parsed Date
    statements_df['year'] = statements_df['Date'].dt.year
    statements_df['month'] = statements_df['Date'].dt.month
    
    return statements_df

# Function to load unemployment rates
def load_unrates():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    unrates = pd.read_csv(url)
    unrates['DATE'] = pd.to_datetime(unrates['DATE'])
    unrates['year'] = unrates['DATE'].dt.year
    unrates['month'] = unrates['DATE'].dt.month
    unrates = unrates[['year', 'month', 'UNRATE']].sort_values(['year', 'month']).reset_index(drop=True)
    return unrates

# Function to preprocess the data and merge FOMC statements with unemployment rates
def preprocess_data(statements, unrates):
    # Ensure 'Date' is in datetime format
    statements['Date'] = pd.to_datetime(statements['Date'], errors='coerce')
    
    # Extract year and month from the 'Date' column
    statements['year'] = statements['Date'].dt.year
    statements['month'] = statements['Date'].dt.month
    
    # Shift the unemployment rate by one month to align with the following month's FOMC statement
    unrates['next_month_unrate'] = unrates['UNRATE'].shift(-1)
    
    # Merge the statements with the corresponding unemployment rate of the next month
    merged_data = pd.merge(statements, unrates[['year', 'month', 'next_month_unrate']], on=['year', 'month'], how='inner')
    
    # Drop any rows with missing data
    merged_data = merged_data.dropna().reset_index(drop=True)
    
    return merged_data

# Function to extract features from text
# Function to extract features from text
def extract_features(df):
    # Basic Textual Features
    df['num_words'] = df['Statement'].apply(lambda x: len(x.split()))
    df['num_chars'] = df['Statement'].apply(lambda x: len(x))
    df['num_sentences'] = df['Statement'].apply(lambda x: len(x.split('.')))
    df['avg_word_length'] = df['Statement'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    
    # Sentiment Analysis
    df['polarity'] = df['Statement'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['Statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Specific Word Counts
    key_terms = ['inflation', 'employment', 'growth', 'unemployment', 'rate']
    for term in key_terms:
        df[f'count_{term}'] = df['Statement'].apply(lambda x: x.lower().split().count(term))
    
    # N-grams (using Bigrams as an example)
    ngram_matrix = vectorizer.transform(df['Statement'])  # Apply the vectorizer fitted earlier
    ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Concatenate the N-gram features with the original dataframe
    df = pd.concat([df, ngram_df], axis=1)
    
    return df


# Function to train the model
def train_model(data):
    # Define features and target
    X = data.drop(columns=['next_month_unrate', 'Statement', 'Date', 'year', 'month'])
    y = data['next_month_unrate']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    
    return model

# Streamlit UI
st.title('Unemployment Rate Predictor')

date_input = st.text_input('Enter Date (YYYYMMDD):', '')
text_input = st.text_area('Enter FED Minutes Text:', '')

# Initialize the vectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))

if st.button('Submit'):
    # Load and preprocess data
    statements_df = load_fomc_statements()
    cleaned_statements_df = clean_statements_df(statements_df)
    unrates_df = load_unrates()
    data = preprocess_data(cleaned_statements_df, unrates_df)

    # Fit the CountVectorizer based on the training data
    vectorizer.fit(data['Statement'])  # Fit the vectorizer here
    data = extract_features(data)

    # Train the model
    model = train_model(data)

    # Predict the next month's unemployment rate
    new_data = pd.DataFrame({"Statement": [text_input]})
    new_data = extract_features(new_data)

    # Align features between training data and new data
    X_new = new_data[data.drop(columns=['next_month_unrate', 'Statement', 'Date', 'year', 'month']).columns]
    prediction = model.predict(X_new)[0]

    # Display the prediction and plot the results
    st.write(f'Unemployment rate prediction for next month is: {prediction:.2f}%')

    # Create a new DataFrame for the predicted value
    new_row = pd.DataFrame({'year': [int(date_input[:4])], 'month': [int(date_input[4:6])], 'UNRATE': [prediction]})

    # Concatenate the new row with the existing DataFrame
    unrates_df = pd.concat([unrates_df, new_row], ignore_index=True)

    # Convert year and month into a datetime column for plotting
    unrates_df['Date'] = pd.to_datetime(unrates_df[['year', 'month']].assign(day=1))

    # Plot the historical data with the prediction
    fig = px.bar(unrates_df[-10:], x='Date', y='UNRATE', title='Unemployment Rate Over Time')
    st.plotly_chart(fig)

