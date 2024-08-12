# -*- coding: utf-8 -*-
"""FED Minutes - Dash

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GRRDsuDLyW8xjxFgAWvMXwzE9KGN2mkb

### Scraping FED scripts
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

# Existing import statements...
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def scrape_93to95(url):
    logging.info(f"Scraping data from URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from URL: {url} with status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all <p> tags within the relevant <div> or other elements unique to the 1993 version
    # You'll need to inspect the HTML of the 1993 version to identify the appropriate elements
    p_tags = soup.find_all('p')  # Adjust this based on the actual HTML structure


    # Define a regular expression pattern to match the date part
    date_pattern = r"/(\d{8})min\.htm"

    # Use re.search to find the date in the URL
    match = re.search(date_pattern, url)

    if match:
        date = match.group(1)
    else:
        print("Date not found in the URL.")
    scraped_text = []

    # Extract text from <p> tags, removing newlines and carriage returns
    url_text = ' '.join([p_tag.get_text().replace('\n', ' ').replace('\r', '') for p_tag in p_tags])
    return url_text, date
def scrape_1996to2007(url):
    logging.info(f"Scraping data from URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table with cellpadding="5"
    target_table = soup.find("table", {"cellpadding": "5"})
    url_parts = url.split('/')

    # Extract the date part (second-to-last element) from the URL
    date = url.split("/")[-1].split(".")[0]
    # Check if the table was found
    if target_table:
        # Find all <p> tags within the table
        p_tags = target_table.find_all("p")

        # Extract and join the text from <p> tags, removing newlines and carriage returns
        table_text = ' '.join([p.get_text().replace('\n', ' ').replace('\r', '') for p in p_tags])

        return table_text,date

    else:
        return None,date
def scrape_2007to2011(url):
    logging.info(f"Scraping data from URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all <p> tags on the page
    p_tags = soup.find_all('p')
    date = url[-12:-4]
    url_text = ' '.join([p_tag.get_text().replace('\n', ' ').replace('\r', '') for p_tag in p_tags])

    return url_text,date
def scrape_2012to2017(url):
    logging.info(f"Scraping data from URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    date = url[-12:-4]
    # Find all elements with class "col-xs-12 col-sm-8 col-md-9"
    target_elements = soup.find_all(class_="col-xs-12 col-sm-8 col-md-9")

    # Check if any elements were found
    if target_elements:
        # Find all <p> tags within the target elements
        p_tags = [element.find_all("p") for element in target_elements]

        # Extract and join the text from <p> tags, removing newlines and carriage returns
        all_text = ''
        for p_tag_group in p_tags:
            for p_tag in p_tag_group:
                all_text += p_tag.get_text().replace('\n', ' ').replace('\r', '') + ' '
        return all_text,date

    else:
        return None,date

def find_minutes_urls1(main_url):
    logging.info(f"Fetching main page: {main_url}")
    # Send an HTTP GET request to the main page
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find and collect the links with names "1993", "1994", and "1995"
    year_links = soup.find_all("a", text=["1995", "1994", "1993"])

    minutes_urls = []

    # Get the base URL to construct absolute URLs
    base_url = main_url

    # Loop through the year links and visit each year's page
    for year_link in year_links:
        # Get the relative URL of the year's page
        year_url_relative = year_link["href"]

        # Construct an absolute URL for the year's page using urljoin
        year_url_absolute = urljoin(base_url, year_url_relative)

        # Send an HTTP GET request to the year's page
        year_response = requests.get(year_url_absolute)
        year_soup = BeautifulSoup(year_response.content, "html.parser")

        # Find and collect the URLs with the name "Minutes"
        minutes_links = year_soup.find_all("a", text="Minutes")

        # Extract the URLs from the "Minutes" links and append them to the list
        minutes_urls.extend([
            urljoin(year_url_absolute, minutes_link["href"])
            for minutes_link in minutes_links
            if minutes_link["href"].startswith("/fomc/MINUTES/") and not minutes_link["href"].endswith("#phone")
        ])
        minutes_urls.reverse()
    return minutes_urls
def find_minutes_urls2(main_url):
    logging.info(f"Fetching main page: {main_url}")
    # Send an HTTP GET request to the main page
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the link for "1996" to start the navigation
    start_year_link = soup.find("a", text="1996")

    if start_year_link is None:
        return []  # Return an empty list if the "1996" link is not found

    minutes_urls = []

    # Get the base URL to construct absolute URLs
    base_url = main_url

    # Loop through the years from 1996 to 2006
    for year in range(1996, 2008):
        # Construct the link text for the year
        year_text = str(year)

        # Find the link for the current year
        year_link = soup.find("a", text=year_text)

        # If the link is found, click on it and navigate to the year's page
        if year_link:
            year_url_relative = year_link["href"]
            year_url_absolute = urljoin(base_url, year_url_relative)

            # Send an HTTP GET request to the year's page
            year_response = requests.get(year_url_absolute)
            year_soup = BeautifulSoup(year_response.content, "html.parser")

            # Find and collect the URLs with the name "Minutes"
            minutes_links = year_soup.find_all("a", text="Minutes")

            # Extract and filter the URLs from the "Minutes" links
            minutes_urls.extend([
                urljoin(year_url_absolute, minutes_link["href"])
                for minutes_link in minutes_links
                if minutes_link["href"].startswith("/fomc/minutes/") and minutes_link["href"].endswith(".htm")
            ])

    # Filter out URLs that don't start with the desired pattern
    minutes_urls = [
        url
        for url in minutes_urls
        if url.startswith("https://www.federalreserve.gov/fomc/minutes/")
    ]
    return minutes_urls
def find_minutes_urls3(main_url):
    logging.info(f"Fetching main page: {main_url}")
    # Send an HTTP GET request to the main page
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find and collect the links for years 2008 to 2011
    year_links = soup.find_all("a", text=["2008", "2009", "2010", "2011"])

    minutes_urls = []

    # Get the base URL to construct absolute URLs
    base_url = main_url

    # Loop through the year links and visit each year's page
    for year_link in year_links:
        # Get the URL of the year's page
        year_url_relative = year_link["href"]
        year_url_absolute = urljoin(base_url, year_url_relative)

        # Send an HTTP GET request to the year's page
        year_response = requests.get(year_url_absolute)
        year_soup = BeautifulSoup(year_response.content, "html.parser")

        # Find and collect the URLs with "/monetarypolicy/fomcminutes" and ending with ".htm"
        all_links = year_soup.find_all("a")

        for link in all_links:
            href = link.get("href")
            if href and "/monetarypolicy/fomcminutes" in href and href.endswith(".htm"):
                full_url = urljoin(year_url_absolute, href)
                minutes_urls.append(full_url)
    minutes_urls.append("https://www.federalreserve.gov/monetarypolicy/fomcminutes20071031.htm")
    minutes_urls.append("https://www.federalreserve.gov/monetarypolicy/fomcminutes20071211.htm")
    minutes_urls.reverse()
    return minutes_urls
def find_minutes_urls4(main_url):
    logging.info(f"Fetching main page: {main_url}")
    # Send an HTTP GET request to the main page
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find and collect the links for years 2012 to 2017
    year_links = soup.find_all("a", text=["2012", "2013", "2014", "2015", "2016", "2017"])

    minutes_urls = []

    # Get the base URL to construct absolute URLs
    base_url = main_url

    # Loop through the year links and visit each year's page
    for year_link in year_links:
        # Get the URL of the year's page
        year_url_relative = year_link["href"]
        year_url_absolute = urljoin(base_url, year_url_relative)

        # Send an HTTP GET request to the year's page
        year_response = requests.get(year_url_absolute)
        year_soup = BeautifulSoup(year_response.content, "html.parser")

        # Find and collect the URLs with "/monetarypolicy/fomcminutes" and ending with ".htm"
        all_links = year_soup.find_all("a")

        for link in all_links:
            href = link.get("href")
            if href and "/monetarypolicy/fomcminutes" in href and href.endswith(".htm"):
                full_url = urljoin(year_url_absolute, href)
                minutes_urls.append(full_url)

    # Reverse the order of collected URLs
    minutes_urls.reverse()

    return minutes_urls
# Example usage:
# Replace with the actual URL of the main page

def find_minutes_urls_after_2017(main_url):
    logging.info(f"Fetching main page: {main_url}")
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")

    minutes_urls = []

    # Find all links that include "/monetarypolicy/fomcminutes" and end with ".htm"
    all_links = soup.find_all("a", href=True)
    for link in all_links:
        href = link.get("href")
        if href and "/monetarypolicy/fomcminutes" in href and href.endswith(".htm"):
            full_url = urljoin(main_url, href)
            minutes_urls.append(full_url)

    return minutes_urls
# Example usage:
scraped_data = []

# Scrape data for 1993 to 1995
main_page_url_1993_to_1995 = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
minutes_urls_1993_to_1995 = find_minutes_urls1(main_page_url_1993_to_1995)
minutes_urls_1993_to_1995.reverse()
for url in minutes_urls_1993_to_1995:
    date, text = scrape_93to95(url)
    scraped_data.append([date, text])

# Scrape data for 1996 to 2007
main_page_url_1996_to_2007 = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
minutes_urls_1996_to_2007 = find_minutes_urls2(main_page_url_1996_to_2007)
for url in minutes_urls_1996_to_2007:
    text, date = scrape_1996to2007(url)
    if date:
        scraped_data.append([date, text])

# Scrape data for 2007 to 2011
main_page_url_2007_to_2011 = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
minutes_urls_2007_to_2011 = find_minutes_urls3(main_page_url_2007_to_2011)
for url in minutes_urls_2007_to_2011:
    text, date = scrape_2007to2011(url)
    if date:
        scraped_data.append([date, text])

# Scrape data for 2012 to 2017
main_page_url_2012_to_2017 = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
minutes_urls_2012_to_2017 = find_minutes_urls4(main_page_url_2012_to_2017)
for url in minutes_urls_2012_to_2017:
    text, date = scrape_2012to2017(url)
    if date:
        scraped_data.append([date, text])

# Scrape data for years after 2017
main_page_url_after_2017 = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
minutes_urls_after_2017 = find_minutes_urls_after_2017(main_page_url_after_2017)
for url in minutes_urls_after_2017:
    text, date = scrape_2012to2017(url)
    if date:
        scraped_data.append([date, text])

# Create a Pandas DataFrame with date and text columns
df = pd.DataFrame(scraped_data, columns=["Date", "Text"])
# Save the DataFrame to a CSV file
df.to_csv("./data/scraped_data_all_years_true.csv", index=False)
#df = pd.read_csv("scraped_data_all_years_true-2.csv")
"""### Importing Modules"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import string
import numpy as np
import re
# import nltk
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error as mape
from tqdm import tqdm
# import dash
from dash import Dash
from dash import dcc
# import dash_core_components as dcc
# import dash_html_components as html
from dash import html
from dash.dependencies import Input, Output, State
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Import the string dictionary that we'll use to remove punctuation
import string
#preprocessing
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
import warnings
warnings.filterwarnings('ignore')
# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from textblob import TextBlob
from tqdm import tqdm
# Other imports
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
import multiprocessing
# %matplotlib inline

import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error as mape
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

"""### Preprocessing & Custom Functions"""

def initial_preprocesser(data):
    data = pd.read_csv("./data/scraped_data_all_years_true.csv")
    data.Date = data.Date.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
    data.Text = data.Text.apply(lambda x: str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
    normal = data[24:]
    need_to_reverse = data[:24]
    need_to_reverse.columns = ["Text", "Date"]
    need_to_reverse = need_to_reverse[["Text", "Date"]]
    data = pd.concat([need_to_reverse, normal], ignore_index=True)
    return data

def second_preprocessor(data):
    # FRED API URL for unemployment rate data (UNRATE)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    # Reading the data directly from the URL
    unrates = pd.read_csv(url)
    
    unrates.UNRATE = unrates.UNRATE.shift(-1)
    unrates["DATE"] = pd.to_datetime(unrates["DATE"])
    unrates["year"] = unrates.DATE.dt.year
    unrates["month"] = unrates.DATE.dt.month
    unrates = unrates.sort_values("DATE").reset_index(drop=True)
    unrates.drop(["DATE"], axis=True, inplace=True)
    
    data["Date"] = pd.to_datetime(data.Date)
    data["year"] = data.Date.dt.year
    data["month"] = data.Date.dt.month
    data = data.sort_values("Date").reset_index(drop=True)
    data = data.merge(unrates, on=["year", "month"], how="left")
    data.columns = ["text", "date", "year", "month", "unrate"]
    return data

def calculate_essential_features(text):
    stops = set(['the', 'a', 'and', 'is', 'of', 'to', 'for'])
    punctuations = set(string.punctuation)
    
    num_words = len(text.split())
    num_unique_words = len(set(text.split()))
    num_chars = len(text)
    mean_word_len = np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0
    
    # Reduced feature set
    return num_words, num_unique_words, num_chars, mean_word_len

# Parallel batch processing
def process_batch(df_batch):
    logging.info("Processing batch with %d records.", len(df_batch))
    
    # Apply the feature calculation function in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda text: calculate_essential_features(text), df_batch['text']))
    
    # Assign calculated features back to DataFrame
    df_batch[['num_words', 'num_unique_words', 'num_chars', 'mean_word_len']] = results
    
    logging.info("Batch processing completed.")
    return df_batch

def word_magician(df, batch_size=1000):
    df_batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    with ThreadPoolExecutor() as executor:
        df_processed_batches = list(executor.map(process_batch, df_batches))
    
    df = pd.concat(df_processed_batches, ignore_index=True)
    
    # Drop rows where num_words is 0
    df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
    logging.info("Word magician function completed.")
    
    return df

def feature_engineering_func(df):
    def get_textBlob_score(sent):
        polarity = TextBlob(sent).sentiment.polarity
        return polarity
    
    df["textblob_sentiment_score"] = df["text"].apply(get_textBlob_score)
    df = df.rename(columns={"unrate": "target"})
    df["target_shift2"] = df.target.shift(2)
    df["target_shift4"] = df.target.shift(4)
    df["target_shift8"] = df.target.shift(8)
    df["target_shift12"] = df.target.shift(12)

    df = df.loc[(df.target_shift12.notnull())].reset_index(drop=True)
    df["target_rolling3"] = df.target.shift(2).rolling(window=3).mean()
    df["target_rolling2"] = df.target.shift(2).rolling(window=2).mean()
    df["target_rolling5"] = df.target.shift(2).rolling(window=5).mean()
    df["target_rolling7"] = df.target.shift(2).rolling(window=7).mean()
    df["target_rolling12"] = df.target.shift(2).rolling(window=12).mean()
    logging.info("Feature engineering completed.")
    return df

"""### Plotly Object"""

app = Dash(__name__)
server = app.server
import os

app.layout = html.Div(style={'backgroundColor': 'white', 'padding': '20px'}, children=[
    html.H1(
        children='Unemployment Rate Predictor',
        style={
            'textAlign': 'center',
            'color': '#0074e4',
            'fontSize': '36px'
        }
    ),
    html.Div(style={'textAlign': 'center'}, children=[
        html.Label('Enter Date:', style={'fontSize': '20px'}),
        dcc.Input(id='date-input', type='text', placeholder='Enter date (e.g., YYYYMMDD)', style={'fontSize': '18px', 'margin-right': '20px'}),

        html.Label('Enter FED Minutes Text:', style={'fontSize': '20px'}),
        dcc.Textarea(id='text-input', placeholder='Enter FED Minutes text...', style={'fontSize': '18px', 'margin-right': '20px',
                                                                                      "margin-bottom": -5}),
        html.Button('Submit', id='submit-button', style={'fontSize': '20px', 'backgroundColor': '#0074e4', 'color': 'white'}),
    ]),
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            html.Div(
                id='output-box',
                style={
                    'backgroundColor': 'lightgreen',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'textAlign': 'center',
                    'margin-top': '20px',
                    'fontSize': '24px',
                },
            ),
        ],
    )
])

@app.callback(
    Output('output-box', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('date-input', 'value'),
    Input('text-input', 'value')
)

def update_output(n_clicks, date_input, text_input):
    logging.info("Received callback with inputs: date_input=%s, text_input=%s", date_input, text_input)
    if n_clicks is not None:
        logging.info("Processing new input...")
        new_input = pd.DataFrame({"Date": [date_input], "Text": [text_input]})
        global df  # Declare df as a global variable to update it
        df = initial_preprocesser(df)
        logging.info("Initial preprocessing completed.")
        df2 = pd.concat([df, new_input], ignore_index=True)
        df2 = second_preprocessor(df2)
        logging.info("Second preprocessing completed.")
        df2 = word_magician(df2, batch_size=1000)  # Increased batch size
        df2 = feature_engineering_func(df2)
        df2.fillna(0, inplace=True)

        train = df2[:-1]
        last_month_unemployment_rate = train.iloc[-1].target
        test = df2[-1:]
        X = train[['year', 'num_words', 'num_unique_words', 'num_chars', 'mean_word_len', 
                   'target_shift2', 'target_shift4', 'target_rolling3', 'target_rolling2']]
        y = train.target
        test_real = test[X.columns]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,  # Reduced number of trees
            learning_rate=0.05,  # Slower learning rate
            max_depth=2  # Shallower trees
        )
        
        model.fit(X, y)
        preds = model.predict(test_real)[0]
        logging.info("Model prediction complete. Prediction: %s", preds)
        
        df2.loc[(df2.date == date_input), "target"] = float(preds)
        df2["fed_minutes_released_date"] = df2.year.astype(str) + "/" + df2.month.astype(str)
        df2 = df2.sort_values("date").reset_index(drop=True)
        df_temp = df2[-10:]
        df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
        fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")

        change = str(float(preds) - last_month_unemployment_rate)[:4]
        return html.Div([
            html.P(f'Unemployment rate prediction for next month is: {preds}%', style={'fontSize': '24px', 'textAlign': 'center'}),
            dcc.Graph(figure=fig)
        ])

logging.info("Starting Dash app...")
if __name__ == '__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4000)), jupyter_mode="external")




# """### Preprocessing & Custom Functions"""

# def initial_preprocesser(data):
#   data = pd.read_csv("./data/scraped_data_all_years_true.csv")
#   data.Date = data.Date.apply(lambda x : str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
#   data.Text = data.Text.apply(lambda x : str(x).replace('  ', ' ').replace('\r', '').replace('\n', ' '))
#   normal = data[24:]
#   need_to_reverse = data[:24]
#   need_to_reverse.columns = ["Text", "Date"]
#   need_to_reverse = need_to_reverse[["Text", "Date"]]

#   data = pd.concat([need_to_reverse, normal], ignore_index= True)
#   return data

# def second_preprocessor(data):
#     # FRED API URL for unemployment rate data (UNRATE)
#     url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
#     # # Reading the data directly from the URL
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

# def word_magician(df, batch_size=50):
#     # Define a reduced set of stopwords and keywords
#     stops = ['the', 'a', 'and', 'is', 'of', 'to', 'for']
#     keywords = ["inflation", "recession", "risk"]

#     # Initialize columns for text features
#     df["num_words"] = np.nan
#     df["num_unique_words"] = np.nan
#     df["num_chars"] = np.nan
#     df["num_stopwords"] = np.nan
#     df["num_punctuations"] = np.nan
#     df["num_words_upper"] = np.nan
#     df["mean_word_len"] = np.nan
#     df["inflation"] = np.nan
#     df["recession"] = np.nan
#     df["risk"] = np.nan

#     def process_batch(batch):
#         for index, row in batch.iterrows():
#             text = str(row["text"])
#             df.at[index, "num_words"] = len(text.split())
#             df.at[index, "num_unique_words"] = len(set(text.split()))
#             df.at[index, "num_chars"] = len(text)
#             df.at[index, "num_stopwords"] = len([w for w in text.lower().split() if w in stops])
#             df.at[index, "num_punctuations"] = len([c for c in text if c in string.punctuation])
#             df.at[index, "num_words_upper"] = len([w for w in text.split() if w.isupper()])
#             df.at[index, "mean_word_len"] = np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0

#             # Count occurrences of each keyword
#             for keyword in keywords:
#                 df.at[index, keyword] = text.lower().count(keyword)

#     # Process the dataframe in batches
#     logging.info("Processing text features in batches.")
#     for i in range(0, len(df), batch_size):
#         batch = df.iloc[i:i + batch_size]
#         with ThreadPoolExecutor() as executor:
#             executor.submit(process_batch, batch)

#     # Drop rows where num_words is 0 (after processing)
#     df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
    
#     logging.info("Batch processing completed.")
#     return df

# def feature_engineering_func(df):
#     def get_textBlob_score(sent):
#       # This polarity score is between -1 to 1
#       polarity = TextBlob(sent).sentiment.polarity
#       return polarity
#     scores=[]
#     for i in tqdm(range(len(df['text']))):
#         score = get_textBlob_score(df['text'][i])
#         scores.append(score)
#     df["textblob_sentiment_score"] = scores
#     df = df.rename(columns = {"unrate" :"target"})
#     df["target_shift2"] = df.target.shift(2)
#     df["target_shift4"] = df.target.shift(4)
#     df["target_shift8"] = df.target.shift(8)
#     df["target_shift12"] = df.target.shift(12)

#     df = df.loc[(df.target_shift12.notnull())].reset_index(drop = True)
#     df["target_rolling3"] = df.target.shift(2).rolling(window= 3).mean()
#     df["target_rolling2"] = df.target.shift(2).rolling(window= 2).mean()
#     df["target_rolling5"] = df.target.shift(2).rolling(window= 5).mean()
#     df["target_rolling7"] = df.target.shift(2).rolling(window= 7).mean()
#     df["target_rolling12"] = df.target.shift(2).rolling(window= 12).mean()
#     return df

# """### Plotly Object"""


# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import io
# import base64
# # app = dash.Dash(__name__)
# app = Dash(__name__)
# server = app.server
# import os
# app.layout = html.Div(style={'backgroundColor': 'white', 'padding': '20px'}, children=[
#     html.H1(
#         children='Unemployment Rate Predictor',
#         style={
#             'textAlign': 'center',
#             'color': '#0074e4',
#             'fontSize': '36px'
#         }
#     ),
#     html.Div(style={'textAlign': 'center'}, children=[
#         html.Label('Enter Date:', style={'fontSize': '20px'}),
#         dcc.Input(id='date-input', type='text', placeholder='Enter date (e.g., YYYYMMDD)', style={'fontSize': '18px', 'margin-right': '20px'}),

#         html.Label('Enter FED Minutes Text:', style={'fontSize': '20px'}),
#         dcc.Textarea(id='text-input', placeholder='Enter FED Minutes text...', style={'fontSize': '18px', 'margin-right': '20px',
#                                                                                       "margin-bottom": -5}),
#         html.Button('Submit', id='submit-button', style={'fontSize': '20px', 'backgroundColor': '#0074e4', 'color': 'white'}),
#     ]),
#     dcc.Loading(
#         id="loading",
#         type="default",
#         children=[
#             html.Div(
#                 id='output-box',
#                 style={
#                     'backgroundColor': 'lightgreen',
#                     'borderRadius': '10px',
#                     'padding': '20px',
#                     'textAlign': 'center',
#                     'margin-top': '20px',
#                     'fontSize': '24px',
#                 },
#             ),
#         ],
#     )
# ])

# @app.callback(
#     Output('output-box', 'children'),
#     Input('submit-button', 'n_clicks'),
#     Input('date-input', 'value'),
#     Input('text-input', 'value')
# )

# def update_output(n_clicks, date_input, text_input):
#     logging.info("Received callback with inputs: date_input=%s, text_input=%s", date_input, text_input)
#     output = html.Div('Performing some operation...')
#     if n_clicks is not None:
#         logging.info("Processing new input...")
#         new_input = pd.DataFrame({"Date": [date_input], "Text": [text_input]})
#         global df  # Declare df as a global variable to update it
#         df = initial_preprocesser(df)
#         logging.info("Initial preprocessing completed.")
#         df2 = pd.concat([df, new_input], ignore_index=True)
#         df2 = second_preprocessor(df2)
#         logging.info("Second preprocessing completed.")
#         df2 = word_magician(df2)
#         logging.info("Word magician function completed.")
#         df2 = feature_engineering_func(df2)
#         logging.info("Feature engineering completed.")
#         df2.fillna(0, inplace = True)

#         train = df2[:-1]
#         last_month_unemployment_rate = train.iloc[-1].target
#         test = df2[-1:]
#         X = train[['year', 'num_words', 'num_unique_words', 'num_chars', 'num_stopwords',
#        'num_punctuations', 'num_words_upper', 'num_words_title',
#        'mean_word_len', 'inflation', 'recession', 'risk', 'target_shift2',
#        'target_shift4', 'target_rolling3', 'target_rolling2']]
#         y = train.target
#         test_real = test[X.columns]
#         model = xgb.XGBRegressor(
#             objective='reg:squarederror',  # For regression
#             n_estimators=100,  # Number of boosting rounds
#             learning_rate=0.1,  # Step size shrinkage to prevent overfitting
#             max_depth=3  # Maximum depth of trees
#         )
#         model.fit(X, y)
#         preds = model.predict(test_real)[0]
#         logging.info("Model prediction complete. Prediction: %s", preds)
#         preds = str(preds)[:4]
#         df2.loc[(df2.date == date_input), "target"] = float(preds)
#         df2["fed_minutes_released_date"] = df2.year.astype(str) + "/" + df2.month.astype(str)
#         # Generate some example data for the line plot
#         df2 = df2.sort_values("date").reset_index(drop = True)
#         df_temp = df2[-10:]
#         df_temp = df_temp.rename(columns = {"target": "unemployment_rate_of_next_month"})
#         fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")

#         change = str(float(preds)-last_month_unemployment_rate)[:4]
#         return  html.Div([
#             html.P(f'Unemployment rate prediction for next month is: {preds}%', style={'fontSize': '24px', 'textAlign': 'center'}),
#             dcc.Graph(figure=fig)
#         ])
# print("selamun aleykum")
# n_clicks = None
# """### Link"""

# logging.info("Starting Dash app...")
# if __name__ == '__main__':
#     app.run(host=os.getenv('IP', '0.0.0.0'),
#             port=int(os.getenv('PORT', 4000)), jupyter_mode= "external")



