import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load data from CSV file
df_customer1 = pd.read_csv('comments.csv')






# move dropdown menu to a new Div
dropdown_source = html.Div([
    dcc.Dropdown(id='customer1-dropdown',
                 options=[{'label': s, 'value': s} for s in df_customer1.source.unique()],
                 multi=True,
                 placeholder='Red social...')
], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'})


# tab
bankTab = dcc.Tab(
    label='Cliente',
    value='tab-1',
    children=[
        html.Div([dcc.Graph(id='customer1-time-series')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='customer1-barplot')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'}),
        dropdown_source,
        html.Div([dcc.Graph(id='customer1-word-cloud')],
                  className='six columns', 
                  style={'width': '45%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='customer1-subset-plot')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'})
    ]
)

