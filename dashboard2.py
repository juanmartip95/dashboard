import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp 
import dash
from dash import dcc, dash_table,html
from dash.dependencies import Input, Output
import wordcloud
import transformers
from pysentimiento import create_analyzer
import dash_bootstrap_components as dbc
import os
import openai
from gensim import corpora, models
from comparativeanalysisTab import comparativeanalysisTab
from BankTab import bankTab
from ComparisonTab import comparisonTab, selected_bank
from dofaTab import DOFATab
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import base64
import numpy as np
from wordcloud import STOPWORDS
from datetime import datetime
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from dash_bootstrap_templates import load_figure_template
# Load data from CSV files for both customers
df_customer1 = pd.read_csv('comments.csv')
df_customer2 = pd.read_csv('bank_comments2.csv')
df_combined = pd.concat([df_customer1, df_customer2], ignore_index=True)
df_AC1=pd.read_csv('Analisis_Completo_1.csv')

# Define the sentiment analysis
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')).union(STOPWORDS)
openai.api_key = os.getenv("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")
emotion_analyzer = create_analyzer(task="emotion", lang="en")


def preprocess_comment(comment):
    # Remove punctuation
    comment = comment.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize words
    tokens = word_tokenize(comment)
    
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word.lower() not in stop_words]
    
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
    wc = WordCloud(width=800, height=600,stopwords=["Bank", "customer"], background_color='Darkblue', color_func=color_func)
    wc.generate_from_text(text)
    return wc

# Load data from CSV file
df = pd.read_csv('comments.csv')
df_AC1=pd.read_csv('Analisis_Completo_1.csv')
df2 = pd.read_csv('bank_comments2.csv')
df3= pd.read_csv('bank_comments.csv')
df_DOFA = pd.read_csv('dofa.csv')
df_customer1 = pd.read_csv('comments.csv')
df_customer2 = pd.read_csv('bank_comments2.csv')
df_combined = pd.concat([df_customer1, df_customer2], ignore_index=True)

# Initialize the Dash app
load_figure_template("superhero")
#app=JupyterDash(__name__)

dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.8/dbc.min.css"
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO, dbc_css])




# Define the layout of the dashboard
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='customers', children=[bankTab, DOFATab, comparativeanalysisTab])],
    className='dbc'
    #,style={'backgroundColor': '#000033'}
    )



# Callback for customer 1 time series graph
@app.callback(
    Output('customer1-time-series', 'figure'),
    Input('customer1-dropdown', 'value')
)
def update_customer1_time_series(selected_sources):
    if selected_sources is None:
        selected_sources = df_customer1.source.unique()
    
    filtered_df = df_customer1[df_customer1['source'].isin(selected_sources)]
    fig = px.line(filtered_df, x='date', y='sentiment', title='Sentimiento a traves del tiempo')
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
    fig = px.histogram(filtered_df, x='source', color='sentiment', barmode='group', title='Sentimiento por red social')
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
    
    filtered_df['comments'] = filtered_df['comments'].apply(lambda x: preprocess_comment(x))
    sentiment_scores = compute_sentiment_scores(filtered_df['comments'])
    text = ' '.join(filtered_df.comments)
    wc = generate_word_cloud(text, sentiment_scores)
    fig = go.Figure(data=[go.Image(z=wc)])
    fig.update_layout(title='Nube de comentarios', xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# Callback for customer 1 subset plot graph
@app.callback(
    Output('customer1-subset-plot', 'figure'),
    Input('customer1-dropdown', 'value')
)
def update_customer1_subset_plot(selected_sources):
    if selected_sources:
        filtered_df = df_customer1[df_customer1['source'].isin(selected_sources)].copy()
    else:
        filtered_df = df_customer1.copy()

    # Apply preprocessing function to comments
    filtered_df['processed_comment'] = filtered_df['comments'].apply(preprocess_comment)

    # Tokenize the processed comments
    tokenized_comments = [comment.split() for comment in filtered_df['processed_comment']]

    # Create a dictionary from the tokenized comments
    dictionary = corpora.Dictionary(tokenized_comments)

    # Create a bag-of-words representation of the comments
    corpus = [dictionary.doc2bow(comment) for comment in tokenized_comments]

    # Perform topic modeling using LDA
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # Assign the main topic to each comment
    filtered_df['main_topic'] = [max(lda_model[comment], key=lambda x: x[1])[0] for comment in corpus]

    fig = px.histogram(filtered_df, x='main_topic', color='sentiment', barmode='group', title='Sentimiento por temas')
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
    fig = px.line(filtered_df, x='date', y='sentiment', title='Sentimiento a traves del tiempo')
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
    fig = px.histogram(filtered_df, x='source', color='sentiment', barmode='group', title='Sentimiento por red social')
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
    
    filtered_df['comments'] = filtered_df['comments'].apply(lambda x: preprocess_comment(x))
    sentiment_scores = compute_sentiment_scores(filtered_df['comments'])
    text = ' '.join(filtered_df.comments)
    wc = generate_word_cloud(text, sentiment_scores)
    fig = go.Figure(data=[go.Image(z=wc)])
    fig.update_layout(title='Nube de comentarios', xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# Callback for customer 2 subset plot graph
@app.callback(
    Output('customer2-subset-plot', 'figure'),
    Input('customer2-dropdown', 'value')
)
def update_customer2_subset_plot(selected_sources):
    if selected_sources:
        filtered_df = df_customer2[df_customer2['source'].isin(selected_sources)].copy()
    else:
        filtered_df = df_customer2.copy()

    # Apply preprocessing function to comments
    filtered_df['processed_comment'] = filtered_df['comments'].apply(preprocess_comment)

    # Tokenize the processed comments
    tokenized_comments = [comment.split() for comment in filtered_df['processed_comment']]

    # Create a dictionary from the tokenized comments
    dictionary = corpora.Dictionary(tokenized_comments)

    # Create a bag-of-words representation of the comments
    corpus = [dictionary.doc2bow(comment) for comment in tokenized_comments]

    # Perform topic modeling using LDA
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # Assign the main topic to each comment
    filtered_df['main_topic'] = [max(lda_model[comment], key=lambda x: x[1])[0] for comment in corpus]

    fig = px.histogram(filtered_df, x='main_topic', color='sentiment', barmode='group', title='Sentimiento por temas')
    return fig


@app.callback(Output('table-O', 'children'), Input('dropdown-week', 'value'))
def update_debilidades(selected_week):
    comments = df_DOFA['O']
    # Intersperse the list with line breaks
    comments_with_breaks = []
    for comment in comments:
        comments_with_breaks.append(comment)
        comments_with_breaks.append(html.Br())
    return comments_with_breaks

@app.callback(Output('table-F', 'children'), Input('dropdown-week', 'value'))
def update_debilidades(selected_week):
    comments = df_DOFA['F']
    # Intersperse the list with line breaks
    comments_with_breaks = []
    for comment in comments:
        comments_with_breaks.append(comment)
        comments_with_breaks.append(html.Br())
    return comments_with_breaks

@app.callback(Output('table-A', 'children'), Input('dropdown-week', 'value'))
def update_debilidades(selected_week):
    comments = df_DOFA['A']
    # Intersperse the list with line breaks
    comments_with_breaks = []
    for comment in comments:
        comments_with_breaks.append(comment)
        comments_with_breaks.append(html.Br())
    return comments_with_breaks

@app.callback(Output('table-D', 'children'), Input('dropdown-week', 'value'))
def update_debilidades(selected_week):
    comments = df_DOFA['D']
    # Intersperse the list with line breaks
    comments_with_breaks = []
    for comment in comments:
        comments_with_breaks.append(comment)
        comments_with_breaks.append(html.Br())
    return comments_with_breaks
# @app.callback(Output('time-series-comparison', 'figure'),
#               Input('dropdown-bank', 'value'))
# def update_time_series_comparison(selected_sources):
#     df3['date'] = pd.to_datetime(df3['date']) 
#     if selected_sources:
#         filtered_df = df3[df3.bank.isin(selected_sources+[selected_bank])]
#     else:
#         filtered_df=df3[df3.bank==selected_bank]

#     filtered_df=filtered_df.groupby(['bank',pd.Grouper(key='date', freq='W')])['sentiment'].mean().reset_index()
#     #print(filtered_df.columns)
#     fig = px.line(filtered_df, x='date', y='sentiment', color='bank', title='Sentiment over time')
#     #fig.add_trace(px.line(filtered_df2, x='date', y='sentiment',color_discrete_sequence=['red']).data[0])
#     fig.update_layout(plot_bgcolor='#B0E0E6', paper_bgcolor='#B0E0E6')

#     return fig

# # Define the callback for the barplot graph
# @app.callback(Output('barplot-comparison', 'figure'),
#               Input('dropdown-bank', 'value'))
# def update_barplot_comparison(selected_sources):
#     if selected_sources:
#         filtered_df = df[df.source.isin(selected_sources)]
#         filtered_df2 = df2[df2.source.isin(selected_sources)]
#     else:
#         filtered_df = df
#         filtered_df2 = df2

#     fig = sp.make_subplots(rows=1, cols=2)#, subplot_titles=('Dataframe 1', 'Dataframe 2'))
#     fig.add_trace(px.histogram(filtered_df,x='source', color='sentiment').data[0], row=1, col=1)
#     fig.add_trace(px.histogram(filtered_df,x='source', color='sentiment').data[1], row=1, col=1)
#     fig.add_trace(px.histogram(filtered_df2,x='source',color='sentiment').data[0], row=1, col=2)
#     fig.add_trace(px.histogram(filtered_df2,x='source',color='sentiment').data[1], row=1, col=2)
#     fig.update_layout(plot_bgcolor='#B0E0E6', paper_bgcolor='#B0E0E6')

#     return fig

# # Define the callback for the word cloud
# @app.callback(Output('word-cloud-bank', 'figure'),
#               Input('dropdown-bank', 'value'))
# def word_cloud_bank(selected_sources):
#     filtered_df = df3[df3.bank == selected_bank]
#     text = ' '.join(filtered_df.comments)
#     wc = wordcloud.WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
#     fig = go.Figure(go.Image(z=wc.to_array()))
#     fig.update_layout(title='Word cloud of bank')
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)
#     fig.update_layout(plot_bgcolor='#B0E0E6', paper_bgcolor='#B0E0E6')

#     return fig 

# @app.callback(Output('word-cloud-comparison', 'figure'),
#               Input('dropdown-bank', 'value'))
# def update_word_cloud_comparison(selected_sources):
#     if selected_sources:
#         filtered_df = df3[df3.bank.isin(selected_sources)]
#     else:
#         filtered_df = df3
#     text = ' '.join(filtered_df.comments)
#     wc = wordcloud.WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
#     fig = go.Figure(go.Image(z=wc.to_array()))
#     fig.update_layout(title='Word cloud of another')
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)
#     fig.update_layout(plot_bgcolor='#B0E0E6', paper_bgcolor='#B0E0E6')

#     return fig 
# Define the callback for the subset plot
# @app.callback(Output('subset-plot-comparison', 'figure'),
#               Input('dropdown-comparison', 'value'))
# def update_subset_plot_comparison(selected_sources):
#     if selected_sources:
#         filtered_df = df2[df2.source.isin(selected_sources)]
#     else:
#         filtered_df = df2
#     fig = px.histogram(filtered_df, x='comments', color='sentiment', barmode='group', title='Sentiment by comment')
#     return fig


# Run the app
if __name__ == '__main__':
  app.run_server(debug=True)
