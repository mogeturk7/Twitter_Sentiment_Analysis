
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import home
import navigation
import model
from  sentiment_predictor import apply_text
import joblib

TWITTER_MODEL = 'TWITTER_MODEL.joblib'
prediction_results = pd.read_csv('predicted.csv', index_col=0)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server


def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)


app.layout = html.Div([
    dcc.Location(id='url'),
    navigation.navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/model':
        return model.layout
    else:
        return home.layout

@app.callback(Output("output_sent", "children"), [Input("input_tweet", "value")])
def output_text(value):
    value = apply_text(eval('["'+value+'"]'))
    return value

@app.callback(Output("opos", "children"), [Input("bpos", "n_clicks")])
def output_pos(ex1):
    pos_data = prediction_results.text[prediction_results.target == 1]
    ex1 = pos_data.sample(n=1)
    ex1 = ex1.reset_index().drop(columns=['index'])
    ex1 = ex1.text[0]
    return ex1

@app.callback(Output("oneg", "children"), [Input("bneg", "n_clicks")])
def output_neg(ex2):
    neg_data = prediction_results.text[prediction_results.target == 0]
    ex2 = neg_data.sample(n=1)
    ex2 = ex2.reset_index().drop(columns=['index'])
    ex2 = ex2.text[0]
    return ex2



if __name__ == '__main__':
    application.run(debug=True, port=8000)




