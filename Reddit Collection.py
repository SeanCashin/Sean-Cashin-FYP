# Reddit - praw https://praw.readthedocs.io/en/stable/index.html#

import datetime
import praw
import sqlite3
import numpy as np

# Set up

reddit_client_id = ''
reddit_client_secret = ''
reddit_user_agent = 'Python: Project Research Application: Praw Version 7.7.1, App Version 1.0 (by /u/SeanCashinResearch)'
reddit_username = 'SeanCashinResearch'
reddit_password = ''
reddit_redirect_uri = '  '

conn = sqlite3.connect('Collected.db')
conn.cursor()

queries_insomniac = ["Insomniac Games AND ransomware",
                     "Insomniac Games AND cyber attack",
                     "Insomniac Games AND data breach",
                     "Insomniac Games AND Rhysida",
                     "Insomniac Games AND data leak",
                     "Insomniac Games AND ransom demand",
                     "Insomniac Games AND cyber extortion",
                     "Insomniac Games AND breach",
                     "Insomniac Games AND security incident",
                     "Insomniac Games AND ransomware group",
                     "Insomniac Games AND cyber security",
                     "Insomniac Games AND hacker group",
                     "Insomniac Games AND data encryption",
                     "Insomniac Games AND ransom negotiation",
                     "Insomniac Games AND breach impact"
                     "Insomniac Games AND cybersecurity response",
                     "Insomniac Games AND data protection",
                     "Insomniac Games AND ransom",
                     "Insomniac Games AND ransomware recovery",
                     "Insomniac Games cyber attack"]


queries_wannacry = ["WannaCry AND attack",
                    "WannaCry AND hack",
                    "WannaCry AND ransomware",
                    "WannaCry AND malware",
                    "WannaCry AND vulnerability",
                    "WannaCry AND exploit",
                    "WannaCry AND prevention",
                    "WannaCry AND mitigation",
                    "WannaCry AND impact",
                    "WannaCry AND consequences",
                    "WannaCry AND recovery",
                    "WannaCry AND remediation",
                    "WannaCry AND cybersecurity",
                    "WannaCry AND defense",
                    "WannaCry AND analysis",
                    "WannaCry AND assessment",
                    "WannaCry AND incident",
                    "WannaCry AND breach",
                    "WannaCry AND ransom",
                    "WannaCry AND ransomware",
                    "WannaCry cyber attack"] 

queries_solarwinds = ["SolarWinds AND attack",
                      "SolarWinds AND breach",
                      "SolarWinds AND hack",
                      "SolarWinds AND compromised",
                      "SolarWinds AND cyber attack",
                      "SolarWinds AND intrusion",
                      "SolarWinds AND incident",
                      "SolarWinds AND cyber espionage",
                      "SolarWinds AND infiltration",
                      "SolarWinds AND malicious activity",
                      "SolarWinds AND vulnerability exploit",
                      "SolarWinds AND data breach",
                      "SolarWinds AND compromise detection",
                      "SolarWinds AND threat intelligence",
                      "SolarWinds AND forensics",
                      "SolarWinds AND investigation",
                      "SolarWinds AND remediation",
                      "SolarWinds AND recovery",
                      "SolarWinds AND vulnerability",
                      "SolarWinds hack"]


def store_post(submission, attack_id):
    sql_query = "INSERT INTO reddit_posts(post_id, subreddit, title, content, score, upvote_ratio, post_datetime, attack_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    subreddit = submission.subreddit.display_name
    conn.execute(sql_query, (submission.id, subreddit, submission.title, submission.selftext, submission.score, submission.upvote_ratio, submission.created_utc, attack_id))
    conn.commit()

def store_comment(comment, post_id):
    sql_query = "INSERT INTO reddit_parent_comments(parent_comment_id, content, score, comment_datetime, post_id) VALUES (?, ?, ?, ?, ?)" 
    conn.execute(sql_query, (comment.id, comment.body, comment.score, comment.created_utc, post_id))
    conn.commit()

def store_reply(reply, comment_id, post_id):
    sql_query = "INSERT INTO reddit_replies(reply_id, content, score, comment_datetime, parent_id, post_id) VALUES (?, ?, ?, ?, ?, ?)"
    conn.execute(sql_query, (reply.id, reply.body, reply.score, reply.created_utc, comment_id, post_id))
    conn.commit()


def search_for_information(reddit, attack_id, queries):
    search = reddit.subreddit("all")
    for search_query in queries:
        search_results = search.search(search_query, sort='relevance', limit = 50)
        # Search for posts based on the query
        for submission in search_results:
            print(f"Title: {submission.title}")
            # Store the post
            store_post(submission, attack_id)
            # Search for comments in the post
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                # Store the comment
                store_comment(comment, submission.id)
                for reply in comment.replies.list():
                    #Store replies
                    store_reply(reply, comment.id, submission.id)
            print(f"Saved for submission {submission.id}")
        print(f"\nSaved query {search_query}")
        print("\n---\n")
    return

def startup():
    # Establish connection to reddit
    try:
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
            username=reddit_username,
            password=reddit_password,
            redirect_uri = reddit_redirect_uri,
        )

        # Print information about the Reddit instance, to prove it worked.
        print(f"Reddit Instance Information:")
        print(f"Authenticated: {not reddit.read_only}")  # If True, it means the instance is read-only (non-authenticated)
        print(f"User: {reddit.user.me()}")  # Print the authenticated user's information

    except praw.exceptions.RedditAPIException as e:
        print(f"Authentication failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    #Set up search, and begin    
    search_or_test = input("searching or testing? Default is testing ").lower()
    if(search_or_test == "search" or search_or_test == "searching"):
        attack = (input("Which attack is this? Insomniac(I), WannaCry(W), or Solarwinds(S)? ").lower())
        if (attack == "i" or attack == "insomniac"):
            search_for_information(reddit, 1, queries_insomniac)
            print("Saved Insomniac Queries")
        if (attack == "w" or attack == "wannacry"):
            search_for_information(reddit, 2, queries_wannacry)
            print("Saved WannaCry Queries")
        if (attack == "s" or attack == "solarwinds"):    
            search_for_information(reddit, 3, queries_solarwinds)
            print("Saved SolarWinds Queries")
    else:
        # Get rate limit information
        rate_limit = reddit.auth.limits
        # Print rate limit information        
        print(rate_limit)
        reset_timestamp = rate_limit['reset_timestamp']
        reset_datetime = datetime.datetime.fromtimestamp(reset_timestamp)
        # Print the reset datetime
        print("Reset datetime:", reset_datetime)
        print("Test done")

startup()