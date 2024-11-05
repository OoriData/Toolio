Reddit-sourced agent-guided newsletter generator (`demo/reddit_newsletter`)

# Basic flow

Agent 1: Asks user main topic of interest, with follow-ups to narrow down

Agent 2: Checks that the post contains no anti-social material

Agent 1: confirms with the user (who is in effect agent 3) before posting

Additions?

* Sentiment analysis ("I only want to select positively toned articles")

# Running

Set up your environment & secrets. [tips, if needed](https://huggingface.co/blog/ucheog/separate-env-setup-from-code)

## Reddit API setup

Requires [Reddit app ID & Secrets](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps)

Give the app a name. Set to script-type.

# Launch demo

Following assumes using 1passwd for secrets

```sh
pip install -r requirements.txt

op run --env-file=.env -- python cli.py summarize_hot
```

# Sample input/session

* Making music
* music production
* How about production in general, and different production software?
* Actually, can we just get on to writing the newsletter?

Leads to: ['/r/WeAreTheMusicMakers', '/r/ableton', '/r/musictheory', '/r/edmproduction', '/r/electronicmusic', '/r/hiphopheads', '/r/Metal', '/r/popheads', '/r/guitar', '/r/piano', '/r/bass', '/r/drums']

# Related intrest

* Lists & DBs of subreddits
  * [awesome-subreddits](https://github.com/iCHAIT/awesome-subreddits) (programming/tech)
  * [r/ListOfSubreddits](https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits/) (community curated)

