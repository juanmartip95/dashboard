Customer Sentiment Analysis
This code provides a customer sentiment analysis dashboard using Dash, Plotly, and NLTK. The dashboard allows you to analyze the sentiment of customer comments from two different sources and compare them.

Getting Started
To get started, make sure you have the following libraries installed:

pandas
plotly
dash
wordcloud
nltk
dash_bootstrap_components
openai
numpy
You can install these libraries using pip:

Copy code
pip install pandas plotly dash wordcloud nltk dash_bootstrap_components openai numpy
Data Preparation
The code assumes that you have two CSV files (comments.csv and bank_comments2.csv) containing customer comments from two different sources. The code reads the data from these CSV files and combines them into a single DataFrame (df_combined).

Running the Dashboard
To run the dashboard, execute the code and open a web browser to the provided URL. The dashboard consists of several tabs:

Customer 1: Displays time series, bar plot, subset plot, and word cloud for customer 1 comments.
Customer 2: Displays time series, bar plot, subset plot, and word cloud for customer 2 comments.
Comparative Analysis: Displays a heatmap comparing the sentiment of comments from both customers, as well as strengths and weaknesses analysis.
Functionality
The dashboard provides the following functionality:

Time Series: Displays the sentiment over time for the selected sources.
Bar Plot: Displays the average sentiment for each source.
Subset Plot: Displays the distribution of sentiment for each comment.
Word Cloud: Generates a word cloud of the comments, with words colored based on sentiment.
Comparative Analysis: Displays a heatmap comparing the sentiment of comments from both customers. Also provides an analysis of strengths and weaknesses based on the selected comments.
Customization
You can customize the dashboard by modifying the code. For example, you can change the color scheme, add more tabs, or modify the preprocessing and analysis functions to suit your specific needs.

Note: The code uses the OpenAI API for strengths and weaknesses analysis. Make sure you have an API key and set it in the openai.api_key variable.

Feel free to explore and analyze your customer sentiment using this dashboard!
