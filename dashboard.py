import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import string
import dash
import dash_bootstrap_components as dbc
import transformers
from pysentimiento import create_analyzer
from dash import dcc, html
from dash.dependencies import Input, Output
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from dash.exceptions import PreventUpdate
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import base64
import numpy as np
import json
import os
import openai
from wordcloud import STOPWORDS
from dash import dcc
from dash.dependencies import Input, Output
from datetime import datetime
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


# Load data from CSV files for both customers
df_customer1 = pd.read_csv('comments.csv')
df_customer2 = pd.read_csv('bank_comments2.csv')
df_combined = pd.concat([df_customer1, df_customer2], ignore_index=True)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

# Define the layout
app.layout = html.Div([
    html.Div(id='app-container', className=''),
    html.H1('Customer Sentiment Analysis'),
    dcc.Tabs(id='tabs', value='customer1', children=[
        dcc.Tab(label='Customer 1', value='customer1', children=[
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    html.H3('Time Series'),
                    dcc.Graph(id='customer1-time-series'),
                ]),
                html.Div(className='six columns', children=[
                    html.H3('Barplot'),
                    dcc.Graph(id='customer1-barplot'),
                ]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    html.H3('Subset Plot'),
                    dcc.Graph(id='customer1-subset-plot'),
                ]),
                html.Div(className='six columns', children=[
                    html.H3('Word Cloud'),
                    dcc.Graph(id='customer1-word-cloud'),
                ]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='six columns', style={'text-align': 'right'}, children=[
                    dcc.Dropdown(
                        id='customer1-dropdown',
                        options=[{'label': s, 'value': s} for s in df_customer1.source.unique()],
                        multi=True,
                        placeholder='Select source...'
                    ),
                ]),
            ]),
        ]),
        dcc.Tab(label='Customer 2', value='customer2', children=[
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    html.H3('Time Series'),
                    dcc.Graph(id='customer2-time-series'),
                ]),
                html.Div(className='six columns', children=[
                    html.H3('Barplot'),
                    dcc.Graph(id='customer2-barplot'),
                ]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    html.H3('Subset Plot'),
                    dcc.Graph(id='customer2-subset-plot'),
                ]),
                html.Div(className='six columns', children=[
                    html.H3('Word Cloud'),
                    dcc.Graph(id='customer2-word-cloud'),
                ]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='six columns', style={'text-align': 'right'}, children=[
                    dcc.Dropdown(
                        id='customer2-dropdown',
                        options=[{'label': s, 'value': s} for s in df_customer2.source.unique()],
                        multi=True,
                        placeholder='Select source...'
                    ),
                ]),
            ]),
        ]),
        dcc.Tab(
            label='Comparative Analysis',
            value='tab-3',
            children=[
                html.Div(className='row', children=[
                    html.Div(className='six columns', children=[
                        dcc.Graph(id='heatmap'),
                    ]),
                    html.Div(className='six columns', children=[
                        html.H3('Strengths and Weaknesses'),
                        html.Div(id='strengths-weaknesses'),  # Updated: use a Div component to display the responses
                    ]),
                ]),

                html.Div(className='row', children=[
                    dcc.Store(id='comment-data', data=df_combined['comments']),  # Store component to hold the comment data
                    html.Div(className='six columns', children=[
                        html.H3('Topics'),
                        dcc.Graph(id='topics-graph'),  # Modified: changed id to 'topics-graph'
                    ]),
                ]),
                dcc.Store(id='comment-themes', data=[]),
            ],
        ),
    ]),
])



 

        







# Define the sentiment analysis
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')).union(STOPWORDS)
openai.api_key = os.getenv("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")
emotion_analyzer = create_analyzer(task="emotion", lang="en")


# Function to preprocess comments
def preprocess_comment(comment):
    # Remove punctuation
    comment = comment.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize words
    tokens = word_tokenize(comment)
    
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    
    # Join tokens back to a single string
    processed_comment = ' '.join(tokens)
    
    return processed_comment

# Function to compute sentiment scores
def compute_sentiment_scores(comments):
    sentiment_scores = []
    for comment in comments:
        sentiment_scores.append(sia.polarity_scores(comment)['compound'])
    return sentiment_scores
def compute_emotion_scores(comments):
    emotion_scores = []
    for comment in comments:
        output = emotion_analyzer.predict(comment)
        emotion_scores.append(output.output)
    return emotion_scores

# Define a custom color function based on sentiment scores
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    sentiment_score = compute_sentiment_scores([word])[0]  # Compute sentiment score for the word
    if sentiment_score > 0:  # Positive sentiment
        return 'limegreen'
    elif sentiment_score < 0:  # Negative sentiment
        return 'red'
    else:  # Neutral sentiment
        return 'yellow'

# Function to calculate word frequencies with dates
def calculate_word_frequencies_with_dates(dataset):
    word_frequencies = {}
    for comment in dataset:
        words = comment.lower().split()  # Convert to lowercase and split into words
        comment_date = df_customer1["date"]  # Replace with the actual date of the comment
        for word in words:
            if word not in word_frequencies:
                word_frequencies[word] = [(comment_date, 1)]
            else:
                # Check if the comment date already exists for the word
                dates = [date for date, _ in word_frequencies[word]]
                if comment_date in dates:
                    # Increment the frequency count for the existing date
                    index = dates.index(comment_date)
                    word_frequencies[word][index] = (comment_date, word_frequencies[word][index][1] + 1)
                else:
                    # Add a new entry for the comment date
                    word_frequencies[word].append((comment_date, 1))
    return word_frequencies

# Calculate word frequencies with dates for the dataset
word_frequencies1 = calculate_word_frequencies_with_dates(df_customer1)
word_frequencies2= calculate_word_frequencies_with_dates(df_customer2)

# Generate word cloud with custom color function
def generate_word_cloud(text, sentiment_scores):
    wc = WordCloud(width=800, height=400, background_color='white', color_func=color_func)
    wc.generate_from_text(text)
    return wc


# Function to analyze strengths and weaknesses
def analyze_strengths_weaknesses(comments):
    processed_comments = [preprocess_comment(comment) for comment in comments]
    positive_comments = [comment for comment, score in zip(comments, compute_sentiment_scores(comments)) if score > 0.2]
    negative_comments = [comment for comment, score in zip(comments, compute_sentiment_scores(comments)) if score < -0.2]
    return positive_comments, negative_comments


# Callback for customer 1 time series graph
@app.callback(
    Output('customer1-time-series', 'figure'),
    Input('customer1-dropdown', 'value')
)
def update_customer1_time_series(selected_sources):
    if selected_sources is None:
        selected_sources = df_customer1.source.unique()
    
    filtered_df = df_customer1[df_customer1['source'].isin(selected_sources)]
    fig = px.line(filtered_df, x='date', y='sentiment', color='source')
    return fig


# Callback for customer 1 barplot graph
@app.callback(
    Output('customer1-barplot', 'figure'),
    Input('customer1-dropdown', 'value')
)
def update_customer1_barplot(selected_sources):
    if not selected_sources:
        selected_sources = df_customer1.source.unique()
    
    filtered_df = df_customer1[df_customer1['source'].isin(selected_sources)]
    fig = px.bar(filtered_df, x='source', y='sentiment', color='source', barmode='group')
    return fig



# Callback for customer 1 word cloud graph
@app.callback(
    Output('customer1-word-cloud', 'figure'),
    Input('customer1-dropdown', 'value')
)
def update_customer1_word_cloud(selected_sources):
        if selected_sources:
            filtered_df = df_customer1[df_customer1.source.isin(selected_sources)]
        else:
            filtered_df = df_customer1
        # Perform sentiment analysis and preprocess comments
        filtered_df['comments'] = filtered_df['comments'].apply(lambda x: preprocess_comment(x))

        # Compute sentiment scores for each word
        sentiment_scores = compute_sentiment_scores(filtered_df['comments'])

        # Combine comments for word cloud generation
        text = ' '.join(filtered_df.comments)

        # Generate word cloud with custom color function
        wc = generate_word_cloud(text, sentiment_scores)  # Pass sentiment_scores as an argument

        # Create the figure
        fig = go.Figure(data=[go.Image(z=wc)])
        fig.update_layout(
            title='Word cloud of comments',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white'
        )

        return fig


# Callback for customer 1 subset plot graph

@app.callback(Output('customer1-subset-plot', 'figure'),
              Input('customer1-dropdown', 'value')
)
def update_customer1_subset_plot(selected_sources):
      if selected_sources:
        filtered_df = df_customer1[df_customer1.source.isin(selected_sources)]
      else:
        filtered_df = df_customer1
      fig = px.histogram(filtered_df, x='comments', color='sentiment', barmode='group', title='Sentiment by comment')
      return fig
    

# Callback for customer 2 time series graph
@app.callback(
    Output('customer2-time-series', 'figure'),
    Input('customer2-dropdown', 'value')
)
def update_customer2_time_series(selected_sources):
    if selected_sources is None:
        selected_sources = df_customer2.source.unique()
    
    filtered_df = df_customer2[df_customer2['source'].isin(selected_sources)]
    fig = px.line(filtered_df, x='date', y='sentiment', color='source')
    return fig


# Callback for customer 2 barplot graph
@app.callback(
    Output('customer2-barplot', 'figure'),
    Input('customer2-dropdown', 'value')
)
def update_customer2_barplot(selected_sources):
    if selected_sources is None:
        selected_sources = df_customer2.source.unique()
    
    filtered_df = df_customer2[df_customer2['source'].isin(selected_sources)]
    fig = px.bar(filtered_df, x='source', y='sentiment', color='source', barmode='group')
    return fig


# Callback for Customer2 Word Cloud Graph
@app.callback(
    Output('customer2-word-cloud', 'figure'),
    Input('customer2-dropdown', 'value')
)
def update_customer2_word_cloud(selected_sources):
        if selected_sources:
            filtered_df = df_customer2[df_customer2.source.isin(selected_sources)]
        else:
            filtered_df = df_customer2
        # Perform sentiment analysis and preprocess comments
        filtered_df['comments'] = filtered_df['comments'].apply(lambda x: preprocess_comment(x))

        # Compute sentiment scores for each word
        sentiment_scores = compute_sentiment_scores(filtered_df['comments'])

        # Combine comments for word cloud generation
        text = ' '.join(filtered_df.comments)

        # Generate word cloud with custom color function
        wc = generate_word_cloud(text, sentiment_scores)  # Pass sentiment_scores as an argument

        # Create the figure
        fig = go.Figure(data=[go.Image(z=wc)])
        fig.update_layout(
            title='Word cloud of comments',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white'
        )

        return fig
    

# Callback for customer 1 subset plot graph

@app.callback(Output('customer2-subset-plot', 'figure'),
              Input('customer2-dropdown', 'value')
)
def update_customer2_subset_plot(selected_sources):
      if selected_sources:
        filtered_df = df_customer2[df_customer2.source.isin(selected_sources)]
      else:
        filtered_df = df_customer2
      fig = px.histogram(filtered_df, x='comments', color='sentiment', barmode='group', title='Sentiment by comment')
      return fig

def summarize_strengths_weaknesses(prompt, comments):
    openai.api_key = os.getenv("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")
    prompt = prompt + "\n\n" + "\n".join(comments)
    response = openai.Completion.create(
        model="text-davinci-003",
        temperature=0.7,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1,
        prompt=prompt
    )
    return response.choices[0].text.strip()


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



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)