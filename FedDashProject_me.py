"""### Importing Modules"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import logging
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
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    unrates = pd.read_csv(url)
    
    unrates.UNRATE = unrates.UNRATE.shift(-1)
    unrates["DATE"] = pd.to_datetime(unrates["DATE"])
    unrates["year"] = unrates.DATE.dt.year
    unrates["month"] = unrates.DATE.dt.month
    unrates = unrates.sort_values("DATE").reset_index(drop=True)
    unrates.drop(["DATE"], axis=1, inplace=True)
    
    data["Date"] = pd.to_datetime(data.Date)
    data["year"] = data.Date.dt.year
    data["month"] = data.Date.dt.month
    data = data.sort_values("Date").reset_index(drop=True)
    data = data.merge(unrates, on=["year", "month"], how="left")
    data.columns = ["text", "date", "year", "month", "unrate"]
    return data

def word_magician(df):
    df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
    df["num_chars"] = df["text"].apply(lambda x: len(str(x)))
    df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in ['the', 'a', 'and', 'is', 'of', 'to', 'for']]))
    df = df.drop(df.loc[df.num_words == 0].index).reset_index(drop=True)
    return df

def feature_engineering_func(df):
    df = df.rename(columns={"unrate": "target"})
    df["target_shift1"] = df.target.shift(1)
    df["target_shift2"] = df.target.shift(2)
    df["target_shift3"] = df.target.shift(3)
    df = df.dropna().reset_index(drop=True)
    return df

"""### Plotly Object"""

import plotly.express as px  # Ensure plotly.express is imported
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
        df2 = word_magician(df2)
        logging.info("Word magician function completed.")
        df2 = feature_engineering_func(df2)
        df2.fillna(0, inplace=True)

        train = df2[:-1]
        last_month_unemployment_rate = train.iloc[-1].target
        test = df2[-1:]
        X = train[['num_words', 'num_chars', 'num_stopwords', 'target_shift1', 'target_shift2']]
        y = train.target
        test_real = test[X.columns]
        
        # Very simple linear regression model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        model.fit(X, y)
        preds = model.predict(test_real)[0]
        logging.info("Model prediction complete. Prediction: %s", preds)
        
        df2.loc[(df2.date == date_input), "target"] = float(preds)
        df2["fed_minutes_released_date"] = df2.year.astype(str) + "/" + df2.month.astype(str)
        df2 = df2.sort_values("date").reset_index(drop=True)
        df_temp = df2[-10:]
        df_temp = df_temp.rename(columns={"target": "unemployment_rate_of_next_month"})
        fig = px.bar(df_temp, x="fed_minutes_released_date", y="unemployment_rate_of_next_month")
        
        return html.Div([
            html.P(f'Unemployment rate prediction for next month is: {preds:.2f}%', style={'fontSize': '24px', 'textAlign': 'center'}),
            dcc.Graph(figure=fig)
        ])

logging.info("Starting Dash app...")
if __name__ == '__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 4000)), jupyter_mode="external")





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



