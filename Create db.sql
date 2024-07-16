CREATE TABLE IF NOT EXISTS attacks (
    attack_id   INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
    attack_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS reddit_posts (
    rp_index INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
    post_id   TEXT  NOT NULL,
    subreddit TEXT,
    title     TEXT      NOT NULL,
    content   TEXT,
    score     INTEGER,
    upvote_ratio FLOAT,
    post_datetime TIMESTAMP,
    attack_id INTEGER,
    FOREIGN KEY (attack_id)
    REFERENCES attacks (attack_id) 
);

CREATE TABLE IF NOT EXISTS reddit_parent_comments (
    rc_index INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
    parent_comment_id TEXT NOT NULL,
    content TEXT, 
    score INTEGER DEFAULT 0,
    comment_datetime TIMESTAMP NOT NUll,
    post_id TEXT NOT NULL,
    FOREIGN KEY (post_id) REFERENCES reddit_posts(post_id)
);

CREATE TABLE IF NOT EXISTS reddit_replies (
    rr_index INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
    reply_id TEXT NOT NULL,
    content TEXT, 
    score INTEGER,
    comment_datetime TIMESTAMP NOT NUll,
    parent_id TEXT NOT NULL, 
    post_id   TEXT  NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES reddit_parent_comments(parent_comment_id),
    FOREIGN KEY (post_id) REFERENCES reddit_posts(post_id)    
);

INSERT INTO attacks (attack_name) 
VALUES ("Insomniac Games"), ("WannaCry"), ("SolarWinds");