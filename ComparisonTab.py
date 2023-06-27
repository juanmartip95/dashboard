import pandas as pd
from dash import dcc
from dash import html
# Load data from CSV file

df3 = pd.read_csv('bank_comments.csv')
selected_bank='Banco BCR'
# graphs

timeSeries=html.Div([dcc.Graph(id='time-series-comparison'),],
    className='six columns', style={'width': '50%', 'display': 'inline-block'})

barplot=html.Div([dcc.Graph(id='barplot-comparison'),],
    className='six columns', style={'width': '50%', 'display': 'inline-block'})

wordCloudBank=html.Div([dcc.Graph(id='word-cloud-bank'),],
    className='six columns', style={'width': '25%', 'display': 'inline-block'})

wordCloudComp=html.Div([dcc.Graph(id='word-cloud-comparison'),],
    className='six columns', style={'width': '25%', 'display': 'inline-block'})

dropdownBank=html.Div([
    dcc.Dropdown(id='dropdown-bank',
        options=[{'label': s, 'value': s} for s in df3.bank.unique()
        if s!=selected_bank],
        multi=True,
        placeholder='Select source...'),],
    className='six columns')



# subsetPlot=html.Div([
#     dcc.Dropdown(id='dropdown',
#         options=[{'label': s, 'value': s} for s in df.source.unique()],
#         multi=True,
#         placeholder='Select source...'),
#     dcc.Graph(id='subset-plot'),],
#     className='six columns', style={'width': '50%', 'display': 'inline-block'})




# tab

comparisonTab=dcc.Tab(label='Comparison', value='tab-2',
    children=[dropdownBank,timeSeries,barplot,wordCloudBank,wordCloudComp])

