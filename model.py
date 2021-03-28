import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_core_components as dcc
import pandas as pd

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import os
import base64

#Import for img files
import cv2

df_build = pd.read_csv('df_build.csv', index_col=0)
df_build_pos = df_build[df_build.target == 1]
df_build_neg = df_build[df_build.target == 0]

text_pos = " ".join(tweet for tweet in df_build_pos.text)
text_neg = " ".join(tweet for tweet in df_build_neg.text)

#print ("There are {} words in the combination of all review.".format(len(text_pos)))
#print ("There are {} words in the combination of all review.".format(len(text_neg)))

stopwords = set(STOPWORDS)
stopwords.update(["quot","u"])

wordcloud_pos = WordCloud(stopwords=stopwords, background_color="white").generate(text_pos)
wordcloud_neg = WordCloud(stopwords=stopwords, background_color="white").generate(text_neg)

# plt.imshow(wordcloud_pos, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# plt.imshow(wordcloud_neg, interpolation='bilinear')
# plt.axis("off")
# plt.show()

word_pos = 'word_pos.png' # replace with your own image
pos_cloud = base64.b64encode(open(word_pos, 'rb').read())

word_neg = 'word_neg.png' # replace with your own image
neg_cloud = base64.b64encode(open(word_neg, 'rb').read())

CM = cv2.imread('confusion_matrix.png')
confusion_img = px.imshow(CM)

colors = {
    'background': '#A9A9A9',
    #'background': '#FFFFFF',
    'text': '#3c4363'
}
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Training Dataset"),
            className="mx-1 my-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("To build our training dataset our Team random sampled the 1.6 million tweets from the data we found on kaggle (https://www.kaggle.com/kazanova/sentiment140). This would greatly reduce the computational requirements and speed up model building. We ended with training data that was perfectly balanced - 5000 negativly marked tweets and corresponding information and 5000 positively marked tweets. For our initial model we dropped all of the columns that were not the target sentiment 1/0 or the actual tweet itself. (NOTE: Left figure is positive sentiments & right figure is negative sentiments)"),
            className="mx-1 my-1")
        ]),    
        dbc.Row([
            dbc.Col([html.Div([html.Img(src='data:image/png;base64,{}'.format(pos_cloud.decode()),style={'width': '100%'})])]),
            dbc.Col([html.Div([html.Img(src='data:image/png;base64,{}'.format(neg_cloud.decode()),style={'width': '100%'})])])
        ]),
        dbc.Row([
            dbc.Col(html.H2("Model Training"),
            className="mx-1 my-1")
        ]),
        dbc.Row([
            dbc.Col(html.P(" In order to train this model we had to import a variety of popular data science modules for Python. numpy, pandas, several sklearn modules, seaborn for the confusion matrix, support vector classifier, spacy, re, joblib - to save the model, matplotlib, a tweet tokenizer, and spacy-stopwords. After cutting our data into a training and testing set by random sampling for a balanced amount of tweets - we then began putting together the model. We wrote a function that prepares the data, droping columns that are not needed and then setting a target variable (sentiment 1+ or 0-). We then setup the model with sklearns' typical train,test, split arguments and chose to reserve thirty percent of the model for cross validation. We then vectorized the actual tweets which were stored as a single string. After the vectorization was complete we wrote a function to use a linear support vector machine learning algorithm to be trained on our training set of data. We then looked at model accuracy in terms of the cross validation score and its ability to detect sentiment in sample and out of sample. The model appears to be stable at around 70 percent accuracy for detecting binary sentiment."),
            className="mx-1 my-1")
        ]),
        html.Div(children=[
        html.Div(style={'color': colors['text'], 'box-shadow': '8px 8px 8px grey' }, children=[
            html.H3('Model Confusion Levels'),
            dcc.Graph(id='Titanic-ex5', figure=confusion_img)],className='six columns'),
    ])
    ])
])
