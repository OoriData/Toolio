Reddit-sourced agent-guided newsletter generator (`demo/reddit_newsletter`)

# Basic flow

Agent 0: The human user

Agent 1 (local LLM, e.g. Mistral Nemo): Interviews the user turn by turn to build an understanding of a topic of interest for the newsletter, with follow-ups to narrow down the focus

Agent 2 (Can be the same local LLM as Agent 1, e.g. Mistral Nemo): Uses the topics the user raised to select a list of relevant subreddits

Agent 3 (Code tool): Reads the hottest, most recent 10 posts from each of the selected subreddits

Agent 4 (OpenAI GPT 4o-mini): Takes the raw posts from Agent 3 and generates a full newsletter

Agent 5 (different local LLM such as LlamaGuard): Checks that the resulting newsletter contains no anti-social material (remember the source is Reddit)

Agent 6 (Code tool): publishes the newsletter


Agent 6 (Code tool): confirms with the user (who is in effect agent 3) before posting

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


Cooking
Asian cuisine
Vietnam and Cambodia
More a genre: street food
Street food
I think that's enough. Let's write!

# Related intrest

* Lists & DBs of subreddits
  * [awesome-subreddits](https://github.com/iCHAIT/awesome-subreddits) (programming/tech)
  * [r/ListOfSubreddits](https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits/) (community curated)

