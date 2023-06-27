import pandas as pd
from dash import dcc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash
from dash.dependencies import Output, Input



# Load data from CSV file
df = pd.read_csv('comments.csv')
df_DOFA = pd.read_csv('dofa.csv')

# Create the Dash app instance
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_plotly_table(df, column_name):
    fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>{}</b>'.format(column_name)],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align=['left','center'],
                    font=dict(color='white', size=12),
                    height=30),
        cells=dict(values=[df[column_name]],
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align=['left', 'center'],
                   font_size=12,
                   height=30))
    ])
    fig.update_layout(
        autosize=True,
        paper_bgcolor="#B0E0E6",  
    )
    return fig

D_comments = dbc.Card(
    [
        #dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Debilidades", className="card-title"),
                html.P(
                    ""
                    "",
                    className="card-text", id='table-D',
                ),
        #        dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    className="w-50"
 )
# D_comments = dcc.Graph(
#     id='table-D',
#     figure=create_plotly_table(df_DOFA, 'D'),
#     style={'width': '45%', 'display': 'inline-block'}
# )

O_comments = dbc.Card(
    [
        #dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Oportunidades", className="card-title"),
                html.P(
                    ""
                    "",
                    className="card-text", id='table-O',
                ),
        #        dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    className="w-50"
 ) 
# dcc.Graph(
#     id='table-O',
#     figure=create_plotly_table(df_DOFA, 'O'),
#     style={'width': '45%', 'display': 'inline-block'}
#)

F_comments = dbc.Card(
    [
        #dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Fortalezas", className="card-title"),
                html.P(
                    ""
                    "",
                    className="card-text", id='table-F',
                ),
        #        dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    className="w-50"
 ) 

# dcc.Graph(
#     id='table-F',
#     figure=create_plotly_table(df_DOFA, 'F'),
#     style={'width': '45%', 'display': 'inline-block'}
# )

A_comments = dbc.Card(
    [
        #dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Amenazas", className="card-title"),
                html.P(
                    ""
                    "",
                    className="card-text", id='table-A',
                ),
        #        dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    className="w-50"
 ) 

# dcc.Graph(
#     id='table-A',
#     figure=create_plotly_table(df_DOFA, 'A'),
#     style={'width': '45%', 'display': 'inline-block'}
# )

Recommendations = dbc.Card(
    [
        # dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Recomendaciones", className="card-title"),
                dcc.Markdown(
                    """
                    Oportunidades de mejora:
                    - Mejorar la eficiencia y rapidez en la resolucion de tramites y reclamos de los clientes.
                    - Fortalecer la seguridad en las transacciones y proteccion de las cuentas de los clientes.
                    - Mejorar la disponibilidad y funcionamiento de la aplicacion movil y el sistema Sinpe.
                    - Mejorar la atencion al cliente en el call center y sucursales.
                    
                    Ventajas competitivas:
                    - Ofrecer la tarjeta Carey, que parece ser un producto atractivo para los clientes.
                    - Mantener protocolos de seguridad Covid en las sucursales.
                    - Ofrecer promociones en negocios comerciales para los clientes.
                    
                    Consejos publicitarios:
                    - Destacar la disponibilidad de la tarjeta Carey y sus beneficios.
                    - Promocionar la eficiencia y rapidez en la resolucion de tramites y reclamos de los clientes.
                    - Destacar la seguridad en las transacciones y proteccion de las cuentas de los clientes.
                    - Promocionar las promociones en negocios comerciales para los clientes.
                    """
                ),
            ],
            className="card-text",
            id='table-R'
        ),  # Move closing parenthesis here
    ],
    className="w-100"
)




# # graphs
# D_comments = html.Div([
#     dash_table.DataTable(
#         id='table-D',
#         columns=[{"name": 'D', "id": 'D'}],
#         data=df_DOFA[['D']].to_dict('records'),
#         style_cell={'textAlign': 'left'},
#         style_data_conditional=[
#             {
#                 'if': {'row_index': 'odd'},
#                 'backgroundColor': 'rgb(248, 248, 248)'
#             }
#         ],
#         style_header={
#             'backgroundColor': 'rgb(230, 230, 230)',
#             'fontWeight': 'bold'
#         }
#     )],
#     className='six columns', 
#     style={'width': '45%', 'display': 'inline-block'}
# )


# O_comments = html.Div([
#     dash_table.DataTable(
#         id='table-O',
#         columns=[{"name": 'O', "id": 'O'}],
#         data=df_DOFA[['O']].to_dict('records'),
#         style_cell={'textAlign': 'left'},
#         style_data_conditional=[
#             {
#                 'if': {'row_index': 'odd'},
#                 'backgroundColor': 'rgb(248, 248, 248)'
#             }
#         ],
#         style_header={
#             'backgroundColor': 'rgb(230, 230, 230)',
#             'fontWeight': 'bold'
#         }
#     )],
#     className='six columns', 
#     style={'width': '45%', 'display': 'inline-block'}
# )


# F_comments = html.Div([
#     dash_table.DataTable(
#         id='table-F',
#         columns=[{"name": 'F', "id": 'F'}],
#         data=df_DOFA[['F']].to_dict('records'),
#         style_cell={'textAlign': 'left'},
#         style_data_conditional=[
#             {
#                 'if': {'row_index': 'odd'},
#                 'backgroundColor': 'rgb(248, 248, 248)'
#             }
#         ],
#         style_header={
#             'backgroundColor': 'rgb(230, 230, 230)',
#             'fontWeight': 'bold'
#         }
#     )],
#     className='six columns', 
#     style={'width': '45%', 'display': 'inline-block'}
# )



# A_comments = html.Div([
#     dash_table.DataTable(
#         id='table-A',
#         columns=[{"name": 'A', "id": 'A'}],
#         data=df_DOFA[['A']].to_dict('records'),
#         style_cell={'textAlign': 'left'},
#         style_data_conditional=[
#             {
#                 'if': {'row_index': 'odd'},
#                 'backgroundColor': 'rgb(248, 248, 248)'
#             }
#         ],
#         style_header={
#             'backgroundColor': 'rgb(230, 230, 230)',
#             'fontWeight': 'bold'
#         }
#     )],
#     className='six columns', 
#     style={'width': '45%', 'display': 'inline-block'}
# )

import dash_bootstrap_components as dbc

dropdown_menus = dbc.Row([
    dcc.Dropdown(id='dropdown-source-1',
                 options=[{'label': s, 'value': s} for s in df.source.unique()],
                 multi=True,
                 placeholder='Red social ...'),
    dcc.Dropdown(id='dropdown-bank-1',
                 options=[{'label': s, 'value': s} for s in df.source.unique()],
                 multi=True,
                 placeholder='Banco ...'),
    dcc.Dropdown(id='dropdown-week',
                 options=[{'label': s, 'value': s} for s in df.source.unique()],
                 multi=True,
                 placeholder='Semana ...')
])

tables = html.Div([dbc.Row([D_comments, O_comments]), dbc.Row([F_comments, A_comments]),dbc.Row([Recommendations])])

# tab
ComparativeanalysisTab = dcc.Tab(label='DOFA', value='tab-2', children=[
    dbc.Row([dropdown_menus, tables],style={'height': '100vh', 'justify-content': 'center'})
])



# Callback for heatmap, strengths-weaknesses graph
@app.callback(
    Output('heatmap', 'figure'),
    Output('strengths-weaknesses', 'children'),
    Input('customer1-dropdown', 'value'),
    Input('customer2-dropdown', 'value')
)
def update_comparative_analysis(customer1_selected_sources, customer2_selected_sources):
    if customer1_selected_sources is None:
        customer1_selected_sources = df_customer1.source.unique()

    if customer2_selected_sources is None:
        customer2_selected_sources = df_customer2.source.unique()

    customer1_filtered_df = df_customer1[df_customer1['source'].isin(customer1_selected_sources)]
    customer2_filtered_df = df_customer2[df_customer2['source'].isin(customer2_selected_sources)]

    # Update heatmap
    heatmap_data = [
        go.Heatmap(
            x=customer1_filtered_df['date'],
            y=customer1_filtered_df['source'],
            z=customer1_filtered_df['sentiment'],
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Sentiment')
        ),
        go.Heatmap(
            x=customer2_filtered_df['date'],
            y=customer2_filtered_df['source'],
            z=customer2_filtered_df['sentiment'],
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Sentiment')
        )
    ]
    heatmap_layout = go.Layout(title='Customer Sentiment Heatmap')
    heatmap_fig = go.Figure(data=heatmap_data, layout=heatmap_layout)

    # Update strengths and weaknesses
    customer1_comments = customer1_filtered_df['comments'].tolist()
    customer2_comments = customer2_filtered_df['comments'].tolist()

    customer1_strengths = summarize_strengths_weaknesses("Strengths for Customer 1:", customer1_comments)
    customer2_strengths = summarize_strengths_weaknesses("Strengths for Customer 2:", customer2_comments)
    customer1_weaknesses = summarize_strengths_weaknesses("Weaknesses for Customer 1:", customer1_comments)
    customer2_weaknesses = summarize_strengths_weaknesses("Weaknesses for Customer 2:", customer2_comments)

    # Create text elements for strengths and weaknesses
    strengths_text = html.P(f"<strong>Customer 1 Strengths:</strong> {customer1_strengths}")
    strengths_text += html.P(f"<strong>Customer 2 Strengths:</strong> {customer2_strengths}")
    weaknesses_text = html.P(f"<strong>Customer 1 Weaknesses:</strong> {customer1_weaknesses}")
    weaknesses_text += html.P(f"<strong>Customer 2 Weaknesses:</strong> {customer2_weaknesses}")

    # Wrap strengths and weaknesses in a container element
    strengths_weaknesses_container = html.Div([strengths_text, weaknesses_text])

    return heatmap_fig, strengths_weaknesses_container






@app.callback(Output('topics-graph', 'figure'), [Input('comment-data', 'data')])
def update_topics_figure(data):
    # Step 1: Load the comments from the DataFrame
    if 'df_combined' in data:
        df_combined = data['df_combined']
    else:
        # Handle the case where 'df_combined' key is missing
        return {}

    if 'comments' in df_combined:
        comments_list = df_combined['comments'].tolist()
    else:
        # Handle the case where 'comments' column is missing
        return {}

    # Step 2: Perform sentiment analysis and obtain sentiment ratings for each comment
    sentiment_responses = []
    for comment in comments_list:
        sentiment_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Classify the sentiment in this comment:\n\n{comment}\n\nSentiment rating:",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        sentiment_responses.append(sentiment_response.choices[0].text.strip())

    # Step 3: Extract keywords from the sentiment responses
    keyword_responses = []
    for sentiment_response in sentiment_responses:
        keyword_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Extract keywords from this sentiment response:\n\n{sentiment_response}\n\nKeywords:",
            temperature=0.5,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.0
        )
        keyword_responses.append(keyword_response.choices[0].text.strip().split(":")[1].strip())

    # Step 4: Map sentiment ratings to themes
    sentiment_mapping = {
        'Positive': 'Positive Theme',
        'Negative': 'Negative Theme',
        'Neutral': 'Neutral Theme'
    }
    themes = [sentiment_mapping.get(sentiment) for sentiment in sentiment_responses]

    # Step 5: Combine keywords with their corresponding themes
    theme_data = []
    for keywords, theme in zip(keyword_responses, themes):
        keyword_list = [keyword.strip() for keyword in keywords.split(",")]
        for keyword in keyword_list:
            theme_data.append({'theme': theme, 'keyword': keyword})

    return theme_data


def update_topics_figure(theme_data):
    if theme_data is None or len(theme_data) == 0:
        # Handle the case when theme data is not available
        return {}

    # Count the occurrences of each theme
    theme_counts = theme_data['theme'].value_counts()

    # Create a bar chart using Plotly graph objects
    fig = go.Figure(data=[go.Bar(x=theme_counts.index, y=theme_counts.values)])
    fig.update_layout(
        title='Theme Analysis',
        xaxis_title='Theme',
        yaxis_title='Count'
    )
    fig.update_traces(marker=dict(shape='bar'))

    return fig

# tab
comparativeanalysisTab = dcc.Tab(
    label='comparativeanalysisTab',
    value='tab-3',
    children=[
        dcc.Graph(id='heatmap'),
        html.Div(id='strengths-weaknesses'),
        dcc.Graph(id='topics-graph'),
    ]
)
