import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load data from CSV file
df_customer2 = pd.read_csv('bank_comments2.csv')



# move dropdown menu to a new Div
dropdown_source2 = html.Div([
    dcc.Dropdown(id='customer2-dropdown',
                 options=[{'label': s, 'value': s} for s in df_customer2.source.unique()],
                 multi=True,
                 placeholder='Red social...')
], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'})


# tab
DOFATab = dcc.Tab(
    label='Cliente2',
    value='tab-2',
    children=[
        html.Div([dcc.Graph(id='customer2-time-series')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='customer2-barplot')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'}),
        dropdown_source2,
        html.Div([dcc.Graph(id='customer2-word-cloud')],
                  className='six columns', 
                  style={'width': '45%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='customer2-subset-plot')],
                 className='six columns', 
                 style={'width': '45%', 'display': 'inline-block'})
    ]
)
