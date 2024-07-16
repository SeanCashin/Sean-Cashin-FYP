import sqlite3
import re
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from scipy.special import softmax
from typing import Tuple, Union
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from matplotlib.ticker import MultipleLocator
from wordcloud import WordCloud
from typing import Tuple
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from matplotlib.ticker import MultipleLocator
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from scipy.stats import pearsonr

# ----------------------------------------------------------------------------------
#  Manipulating Data and getting dfs - (get_)
# ----------------------------------------------------------------------------------

def get_attack_from_merged(posts_df: pd.DataFrame, comments_df: pd.DataFrame, replies_df: pd.DataFrame,attack_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    posts = posts_df[posts_df['attack_id'] == attack_id]
    comments = comments_df[comments_df['post_id'].isin(posts['post_id'])]
    replies = replies_df[replies_df['post_id'].isin(posts['post_id'])]
    return posts, comments, replies

def get_all_attacks(conn) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    posts = pd.read_sql(sql = f"select * from reddit_posts", con = conn)
    comments = pd.read_sql(sql = f"select * from reddit_parent_comments", con = conn)
    replies = pd.read_sql(sql = f"select * from reddit_replies", con = conn)
    posts.set_index(posts.columns[0], inplace=True, drop=True)
    comments.set_index(comments.columns[0], inplace=True, drop=True)
    replies.set_index(replies.columns[0], inplace=True, drop=True)
    return posts, comments, replies

def get_all_comments_df(posts: pd.DataFrame, parent_comments: pd.DataFrame, reddit_replies: pd.DataFrame) -> pd.DataFrame:
    # Merge dfs
    comments_df = pd.concat([parent_comments, reddit_replies], ignore_index=False)
    # Handle empty values in p_c_id (from replies)
    comments_df['parent_comment_id'] = comments_df['parent_comment_id'].fillna(comments_df['reply_id'])
    # Rename p_c_id, remove now irrelevant comments
    comments_df.rename(columns={'parent_comment_id': 'comment_id'}, inplace=True)
    comments_df.drop(columns=['reply_id'], inplace=True)
    comments_df.drop(columns=['parent_id'], inplace=True)
    # Reset the index of the DataFrame
    comments_df.reset_index(drop=True, inplace=True)
    # Add relevant subreddit values
    post_subreddit_dict = posts.set_index('post_id')['subreddit'].to_dict()
    comments_df['subreddit'] = comments_df['post_id'].map(post_subreddit_dict)
    # Do the same for attack id
    post_attack_dict = posts.set_index('post_id')['attack_id'].to_dict()
    comments_df['attack_id'] = comments_df['post_id'].map(post_attack_dict)
    # Remove empty comments
    comments_df = comments_df[comments_df['content'] != '']
    # Ensure no duplicates
    comments_df = comments_df.drop_duplicates(subset = 'comment_id')
    return comments_df

def get_top_10_active_posts(posts, comments):
    # Count occurrences of each post_id in comments
    post_counts = comments['post_id'].value_counts().reset_index()
    post_counts.columns = ['post_id', 'comment_count']
    
    # Calculate average sentiment_score for each post
    avg_sentiment_scores = comments.groupby('post_id')['sentiment_score'].mean().reset_index()
    avg_sentiment_scores.rename(columns={'sentiment_score': 'average_sentiment_score'}, inplace=True)
    
    # Merge post counts with average sentiment scores and posts DataFrame to get titles
    top_10_posts = pd.merge(post_counts.head(10), avg_sentiment_scores, on='post_id', how='left')
    top_10_posts = pd.merge(top_10_posts, posts[['post_id', 'title']], on='post_id', how='left')
    # Adjust display settings to show full text in DataFrame cells
    return top_10_posts

# ----------------------------------------------------------------------------------
# Process data - _process
# ----------------------------------------------------------------------------------

def process_comment_sentiment(text, sentiment_tokenizer, sentiment_config, sentiment_model) -> pd.DataFrame:   
    # Tokenize input to the format suitable for the model
    encoded_input = sentiment_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Generate predictions
    with torch.no_grad():
        output = sentiment_model(**encoded_input)
    # Process scores
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    # Extract positive, neutral, and negative scores
    positive_score = scores[sentiment_config.label2id['positive']]
    neutral_score = scores[sentiment_config.label2id['neutral']]
    negative_score = scores[sentiment_config.label2id['negative']]
    # Calculate overall sentiment score 
    overall_score = (1 * positive_score + 0 * neutral_score - 1 * negative_score) 
    # Determine class with highest score
    classes = ['positive', 'neutral', 'negative']
    max_class_index = np.argmax([positive_score, neutral_score, negative_score])
    determined_sentiment = classes[max_class_index]
    # Create DataFrame
    data = {'sentiment_score': overall_score, 'determined_sentiment': determined_sentiment}
    df = pd.DataFrame([data])
    return df

def process_df(df:pd.DataFrame, column:str, sentiment_tokenizer, sentiment_config, sentiment_model)-> pd.DataFrame: 
    result_df = pd.DataFrame()
    # Iterate over each row 
    for index, row in df.iterrows():
        # Get sentiment scores for the content in the current row
        sentiment_scores = process_comment_sentiment(row[f'{column}'], sentiment_tokenizer, sentiment_config, sentiment_model)
        # Append the sentiment scores to the result DataFrame
        result_df = pd.concat(objs = [result_df, sentiment_scores], ignore_index=True)
    result_df.index = df.index
    df = df.join(result_df)
    return df

# ----------------------------------------------------------------------------------
# Graphs - display_ start of name
# ----------------------------------------------------------------------------------

def display_sentiment_distribution_hist(df:pd.DataFrame, attack):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sentiment_score'], bins = 'auto', kde=True)
    plt.title(f'Distribution of Sentiment Scores ({attack})')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()  
    
def display_determined_sentiment_distribution_countplot(df:pd.DataFrame, attack_name):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='determined_sentiment', order=['negative', 'neutral', 'positive'])
    plt.title(f'Distribution of Sentiment Classes ({attack_name})')
    plt.xlabel('Determined Class')
    plt.ylabel('Frequency')
    plt.show()
    
def display_subreddit_sentiment_over_time_line(df:pd.DataFrame, attack_name, relevant_dates=None, filter_period=None, top_subreddits=10):
    # Convert comment_datetime to datetime
    df['comment_datetime'] = pd.to_datetime(df['comment_datetime'], unit='s')
    # Extract date from datetime
    df['date'] = df['comment_datetime'].dt.date
    # Calculate cutoff date based on filter_period
    if filter_period:
        num, period = filter_period.split()
        num = int(num)
        if period.lower() in ['month', 'months']:
            cutoff_date = df['comment_datetime'].min() + pd.DateOffset(months=num)
        elif period.lower() in ['day', 'days']:
            cutoff_date = df['comment_datetime'].min() + pd.DateOffset(days=num)
        else:
            raise ValueError("Invalid filter period. Please specify 'Month(s)' or 'Day(s)'.")
        # Filter data to include only dates within the specified period
        df = df[df['comment_datetime'] <= cutoff_date]
    # Get the top subreddits by occurrences
    top_subreddits = df['subreddit'].value_counts().nlargest(top_subreddits).index.tolist()
    # Filter data to include only top subreddits
    df = df[df['subreddit'].isin(top_subreddits)]
    # Calculate average sentiment score for each day and subreddit
    avg_sentiment = df.groupby(['date', 'subreddit'])['sentiment_score'].mean().reset_index()
    # Plotting sentiment trends over time for each subreddit
    plt.figure(figsize=(20,12))
    sns.lineplot(x='date', y='sentiment_score', hue='subreddit', data=avg_sentiment, marker='o')
     # Adding vertical lines for relevant dates with staggered horizontal text annotations
    if relevant_dates:
        y_position = 0.8
        for date, label in relevant_dates.items():
            date = pd.to_datetime(date).date()
            plt.axvline(x=date, color='gray', linestyle='--')
            plt.text(date, y_position, label, color='black', ha='center', va='bottom', rotation='horizontal')
            y_position -= 0.05  # Adjust the vertical position
    plt.title(f'{attack_name}: Average Sentiment Over Time for Top {np.size(top_subreddits)} Most Active Subreddits')
    plt.xlabel('Date')
    plt.grid(True)
    plt.ylabel('Average Sentiment Score')
    plt.legend(title='Subreddit', loc='upper right')
    plt.show()

def display_relationship_sentiment_v_score(df:pd.DataFrame):
    # Calculate correlation coefficient
    correlation_coefficient, _ = pearsonr(df['sentiment_score'], df['score'])
    # Group the DataFrame by sentiment category
    grouped_by_category = df.groupby('determined_sentiment')
    # Calculate mean entry score by sentiment category
    mean_entry_score_by_category = grouped_by_category['score'].mean().round(3)
    # Calculate standard deviation of entry scores by sentiment category
    std_dev_entry_score_by_category = grouped_by_category['score'].std().round(3)

    # Prepare the data for the table
    headers = ['Metric', 'Value']
    data = [
        ('Correlation Coefficient', correlation_coefficient),
        ('Mean Entry Score by Sentiment Category', mean_entry_score_by_category.to_dict()),
        ('Standard Deviation of Entry Scores by Sentiment Category', std_dev_entry_score_by_category.to_dict())
    ]

    # Display the results in a table using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')  # Turn off axis
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='left', cellColours=[['#f2f2f2', '#f2f2f2']]*len(data))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Scale table to fit better
    table.auto_set_column_width([0, 1])  # Adjust column width automatically
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # Set cell border width
    plt.title('Relationship between Sentiment Score and Entry Score')

    plt.show()
    # Interpretation
    print("\nInterpretation:")
    if abs(correlation_coefficient) < 0.3:
        if mean_entry_score_by_category['positive'] > mean_entry_score_by_category['negative']:
            print("In conclusion, there is a weak or no relationship between sentiment score and entry score. "
                "However, entries with positive sentiment tend to have slightly higher average scores.")
        else:
            print("In conclusion, there is a weak or no relationship between sentiment score and entry score. "
                "However, entries with negative sentiment tend to have slightly higher average scores.")
    elif abs(correlation_coefficient) < 0.7:
        if mean_entry_score_by_category['positive'] > mean_entry_score_by_category['negative']:
            print("In conclusion, there is a moderate relationship between sentiment score and entry score. "
                "Entries with positive sentiment tend to have higher average scores.")
        else:
            print("In conclusion, there is a moderate relationship between sentiment score and entry score. "
                "Entries with negative sentiment tend to have higher average scores.")
    else:
        if std_dev_entry_score_by_category['positive'] < std_dev_entry_score_by_category['negative']:
            print("In conclusion, there is a strong relationship between sentiment score and entry score. "
                "Entries with positive sentiment tend to have less variability in scores.")
        else:
            print("In conclusion, there is a strong relationship between sentiment score and entry score. "
                "Entries with negative sentiment tend to have less variability in scores.")

def display_wordcloud(df:pd.DataFrame, cloud_subject, posts_or_comments: str):
    all_content = ''
    if posts_or_comments.lower() == "posts":
        all_titles = ' '.join(df['title'].astype(str))
        all_content = ' '.join(df['content'].astype(str))
        all_content = all_titles + all_content
    if posts_or_comments.lower() == "comments":
        all_content = ' '.join(df['content'].astype(str))
    # Create word cloud
    wordcloud = WordCloud(width=1000, height=600, background_color='white').generate(all_content)
    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud of {cloud_subject}')
    plt.axis('off')
    plt.show() 
    
# ----------------------------------------------------------------------------------
# Comparison graphs - compare_ start of name
# ----------------------------------------------------------------------------------

def compare_avg_sentiment_three_months_line(df:pd.DataFrame, start_dates=None):
    # Convert comment_datetime to datetime
    df['comment_datetime'] = pd.to_datetime(df['comment_datetime'], unit='s')
    
    # Group data by attack_id and calculate the start date of each attack
    attack_start_dates = df.groupby('attack_id')['comment_datetime'].min()
    
    # Initialize lists to store data for plotting
    attack_ids = []
    relative_dates = []
    sentiment_scores = []
     # Mapping of attack IDs to attack names
    attack_names = {1: 'Insomniac', 2: 'Wannacry', 3: 'SolarWinds'}
    
    # Iterate over each attack to extract sentiment data based on start dates
    for attack_id, start_date in attack_start_dates.items():
        # If start_dates parameter is provided, use it to filter data for each attack
        if start_dates and attack_id in start_dates:
            filter_start_date = start_dates[attack_id]
        else:
            filter_start_date = start_date
        
        # Filter data for the current attack based on start date
        attack_data = df[(df['attack_id'] == attack_id) & 
                         (df['comment_datetime'] >= filter_start_date) & 
                         (df['comment_datetime'] < filter_start_date + pd.DateOffset(months=3))]
        
        # Calculate relative dates starting from 1 for each attack
        attack_data['relative_date'] = (attack_data['comment_datetime'] - filter_start_date).dt.days + 1
        
        # Store data for plotting
        attack_ids.extend(attack_data['attack_id'])
        relative_dates.extend(attack_data['relative_date'])
        sentiment_scores.extend(attack_data['sentiment_score'])
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({'attack_id': attack_ids, 'relative_date': relative_dates, 'sentiment_score': sentiment_scores})
    
    # Plotting sentiment trends over time for each attack
    plt.figure(figsize=(20, 12))
    sns.lineplot(x='relative_date', y='sentiment_score', hue='attack_id', data=plot_data, marker='o', palette='Set1', ci=None)
    plt.grid(True)
    plt.title('Average Sentiment Over Time For All Attacks')
    plt.xlabel('Relative Attack Day')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
   # Update legend labels with attack names
    legend_labels = [attack_names.get(attack_id, attack_id) for attack_id in plot_data['attack_id'].unique()]
    plt.legend(title='Attack', labels=legend_labels, loc='upper right')
    plt.show()
    
    
# ----------------------------------------------------------------------------------
# Test dataset functions 
# ----------------------------------------------------------------------------------

def filter_english(dataset):
    for row in dataset:
        text = row['text']
        try:
            language = detect(text)
            # Check if language is English
            if language != 'en':  
                row['text'] = '' 
        except:
            pass  # Skip rows where language detection fails
    return dataset

def clean_text(dataset):
    for row in dataset:
        text = row['text']
        # Convert text to string
        text = str(text)
        # Remove HTML tags
        cleaned_text = re.sub(r'<[^>]+>', '', text)
        # Remove Reddit-specific markup
        cleaned_text = re.sub(r'(\*|_)(.*?)\1', r'\2', cleaned_text)  # Markdown for italics
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)   # Markdown for bold
        cleaned_text = re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1', cleaned_text)  # Markdown for hyperlinks
        cleaned_text = re.sub(r'> (.*?)\n', r'\1', cleaned_text)       # Markdown for quoting text
        # Remove new lines
        cleaned_text = cleaned_text.replace('\n', ' ')
        # Remove links
        cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        row['text'] = cleaned_text
    return dataset

def remove_blank(dataset, text_column="text"):
    # Filter out examples with empty text
    dataset = dataset.filter(lambda row: row[text_column] != '')
    return dataset


def chunk_text(text, max_words_per_chunk=20):
    # Split data up into chunks of 20
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def autocorrect_text(dataset):
    spell = Speller(fast = True) 
    # Without this setting, it would not complete in a reasonable timeframe 
    for row in dataset:
        # Split the text into smaller chunks
        chunks = chunk_text(row['text'])
        # Process each chunk separately
        corrected_chunks = []
        for chunk in chunks:
            corrected_chunks.append(str(spell(chunk)))   
        # Combine the corrected chunks into a single string
        row['text'] = " ".join(corrected_chunks)
    return dataset

def lemmatise_remove_stopwords(dataset):
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    nltk_stopwords = set(stopwords.words('english'))
    # https://gist.github.com/sebleier/554280 - source of list of stopwords
    removed_stopwords = ["all", "any", "both", "each", "few", "more", "most", 
                    "other", "some", "such", "only", "own", "same", "too", 
                    "very", "s", "t", "just", "now", "not", "in"] 
    for word in removed_stopwords:
            if word in nltk_stopwords:
                nltk_stopwords.remove(word)
    for row in dataset:
        text = row['text']
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Lemmatize and remove stopwords
        processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token.isalnum()]
        # Help the model handle larger inputs
        if len(processed_tokens) > 512:
            # Truncate the tokens list to contain only the first 512 tokens
            processed_tokens = processed_tokens[:512]
        # Concatenate tokens into a string
        processed_text = ' '.join(processed_tokens)
        row['text'] = processed_text
    return dataset

def preprocess_test_dataset(dataset):
    dataset = filter_english(dataset)
    dataset = clean_text(dataset)
    dataset = remove_blank(dataset)
    dataset = autocorrect_text(dataset)
    dataset = lemmatise_remove_stopwords(dataset)
    return dataset

def evaluate_test_dataset(test_labels_mapped, pred_labels):
   # Calculate evaluation metrics for validation data
    accuracy = round(accuracy_score(test_labels_mapped, pred_labels), 5)
    precision = round(precision_score(test_labels_mapped, pred_labels, average='weighted'), 5)
    recall = round(recall_score(test_labels_mapped, pred_labels, average='weighted'), 5)
    f1 = round(f1_score(test_labels_mapped, pred_labels, average='weighted'), 5)
    headers = ['Metric', 'Value']
    scores_table = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1-score", f1)
    ]
     # Display the results in a table using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis('off')  # Turn off axis
    table = ax.table(cellText=scores_table, colLabels=headers, loc='center', cellLoc='left', cellColours=[['#f2f2f2', '#f2f2f2']]*len(scores_table))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Scale table to fit better
    table.auto_set_column_width([0, 1])  # Adjust column width automatically
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # Set cell border width
    plt.title('Evaluation Metrics:')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(test_labels_mapped, pred_labels)
    plt.show()