import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from application import app

import os
import base64


df = pd.read_csv('predicted.csv')
df = df.rename(columns={"target": "sentiment"})
df.sentiment = df.sentiment.apply(lambda x:'negative' if x == 0 else 'positive')

fig1 = px.histogram(df, x='sentiment', color='sentiment')


### Word Cloud Creation
text = " ".join(review for review in df.text)

#print ("There are {} words in the combination of all review.".format(len(text)))

stopwords = set(STOPWORDS)
stopwords.update(["quot","u"])
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

word_home = 'word_home.png' # replace with your own image
home_img = base64.b64encode(open(word_home, 'rb').read())


layout = html.Div(children=[
    dbc.Container([
        dbc.Row(children=[
            dbc.Col(html.H1("Twitter Sentiment Prediction Dashboard", className="text-center"),
            className="mx-1 my-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5('This application predicts sentiment of tweets', className="text-center"),className="mx-1 my-1") 
        ]),
        dbc.Row([
            dbc.Col(html.P('The dataset can be found on Kaggle under "Sentiment140 dataset with 1.6 million tweets". The dataset contains two columns of interest: sentiment (denoted by 0 for negative and 1 for positive) and text (containing the content of each tweet). The original dataset was balanced with 800K each of positive and negative tweets. Our model building data represents a random sample of 5000 of each sentiment.', className="text-justify")) 
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                    id="chr-1",
                    figure=fig1),className="col-6"),
            dbc.Col(children=[dcc.Input(
                    id="input_tweet",
                    placeholder="Input to test your tweet here",
                    type='text',
                    style={'width': '100%'
                            ,'height': '10%'   
                            , 'textAlign': 'center'}),
                    html.Br(),
                    html.P(id="output_sent",style={'color':'navy', 'fontSize': 21}),
            dbc.Col([html.Div([html.Img(src='data:image/png;base64,{}'.format(home_img.decode()),style={'width': '100%'})])]),
                        ]
                    )
        ]),  
        dbc.Row([
            dbc.Col(html.H3('Sample Tweets', className="text-center"),
            className="mx-1 my-1")
        ]),
        dbc.Row([
            dbc.Col(style={'box-shadow': '8px 8px 8px grey'}, children = [
                html.H5('Predicted Positive', className='card-title'),
                html.P(id="opos"),
                dbc.Button("Randomize", id ="bpos", color="secondary", className="mr-1")
                ],
            className="col-6"),
            dbc.Col(style={'box-shadow': '8px 8px 8px grey'}, children = [
                html.H5('Predicted Negative', className='card-title'),
                html.P(id="oneg"),
                dbc.Button("Randomize", id ="bneg", color="secondary", className="mr-1")
                ],
            className="col-6")
        ])
    ])
])
