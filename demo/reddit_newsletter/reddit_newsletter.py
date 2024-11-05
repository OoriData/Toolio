'''
demo/reddit_newsletter/cli.py

Agent 1: Asks user main topic of interest, with follow-ups to narrow down

Agent 2: Checks that the post contains no anti-social material

Agent 1: confirms with the user (who is in effect agent 3) before posting

Additions?

* Sentiment analysis ("I only want to select positively toned articles")



First set up your environment & secrets. Useful resource: https://huggingface.co/blog/ucheog/separate-env-setup-from-code

You need [Reddit app ID & Secrets](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps)

Give the app a name. Easiest to just make it script-type.

Following assumes using 1passwd for secrets

```sh
pip install -r requirements.txt

op run --env-file=.env -- python cli.py summarize_host
'''
import os
import json
import asyncio
import warnings
import ssl

import aiohttp
import asyncpraw

# from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

# from toolio.tool import tool
from toolio.llm_helper import model_manager
from toolio.common import response_text

# Need this crap to avoid self-signed cert errors (really annoying, BTW!)
ssl_ctx = ssl.create_default_context(cafile=os.environ.get('CERT_FILE'))

# Requires OPENAI_API_KEY in environment
# llm_api = openai_chat_api(model='gpt-4o-mini')

MLX_MODEL_PATH = 'mlx-community/Mistral-Nemo-Instruct-2407-4bit'

USER_AGENT = 'Python:Arkestra Agent:v0.1.0 (by u/CodeGriot'

MAX_SUBREDDITS = 3

AGENT_1_SCHEMA = '''\
{
  "type": "object",
  "description": "Respond with a \\"question\\" response, or \\"ready\\" BUT NEVER BOTH.",
  "anyOf": [
    {"required" : ["question"]},
    {"required" : ["ready"]}
  ],
  "properties": {
    "question": {
      "description": "Initial or follow-up question to the user. DO NOT USE IF YOU ALSO USE ready.",
      "type": "string"
    },
    "ready": {
      "description": "Use this when you have a decent amount of info from the user. Contains a summarized bullet list of to user's desired topics, in a way that's easy to look up by tags in a public forum. DO NOT USE IF YOU ALSO USE question.",
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "additionalProperties": false
}
''' #  noqa E501

AGENT_2_SCHEMA = '{"type": "array", "items": { "type": "string" } }'

AGENT_3_UPROMPT = '''\
Here is a list of the top ten hottest post titles in a selection of subreddits: 

=== BEGIN POST TITLES
{titles}
=== END POST TITLES

The source subreddits are: {subreddits}

Based on this, please provide a nicely-written newletter summary of the interesting topics of the day across these
forums. Try to make it more than just a random list of topics, and more a readable take on the latest points of intrest.
'''

toolio_mm = model_manager(MLX_MODEL_PATH)

# System prompt will be used to direct the LLM's tool-calling
agent_1_sysprompt = '''\
You are a newsletter ghost-writer who interviews a user asking for a topic of interest, with a couple
of follow-up questions to narrow down the discussion. The user will start the process by saying they want to
create a newsletter post. Your responses are always in JSON, either using the
`question` object key to ask the user an inital or follow-up question, or when you think you have
a reasonable amount of information, and after no more than a few turns, you will respond using
the `ready` object key, with a list of topics in a form that can be looked up as simple topic tags.
Here is the schema for you to use: {json_schema}
'''
userprompt_1 = 'Hi! I\'d like to create a newsletter post'

agent_2_sysprompt = '''\
You are a helpful assistant who specializes in finding online forums relevant
to a user's informational needs. You always respond with a simple list of forums
to check, relevant to a given list of topics.
'''

agent_2_uprompt = '''\
Given the following list of topics:

### BEGIN TOPICS
{topics}
### END TOPICS

Select which of the following subreddits might be useful for research.

### BEGIN SUBREDDITS
{subreddits}
### END SUBREDDITS

Respond using the following schema: {{json_schema}}
'''

async def async_main(tmm):
    print('Stage 1: Initial interview (user & agent 1)')
    msgs = [ {'role': 'system', 'content': agent_1_sysprompt},
             {'role': 'user', 'content': userprompt_1} ]
    interview_finished = False
    ready = None
    while not interview_finished:
        resp = await response_text(tmm.complete(msgs, json_schema=AGENT_1_SCHEMA, max_tokens=2048))
        resp = json.loads(resp)
        # print(resp)
        question = resp.get('question')
        ready = resp.get('ready')
        if question and ready:
            warnings.warn(f'The LLM got a bit confused and returned a question and a ready list, but we\'ll just use the latter: {resp}')
        if ready:
            interview_finished = True
        elif question:
            msgs.append({'role': 'assistant', 'content': question})
            user_msg = input(question + '    ')
            print()
            msgs.append({'role': 'user', 'content': user_msg})

    print('Stage 2: Look up subreddits (agents 1 & 2)')
    with open('subreddits.json') as fp:
        subreddits = json.load(fp)

    # Type maniac linters get on my nerves 🤬
    msgs = [ {'role': 'system', 'content': agent_2_sysprompt},
             {'role': 'user', 'content': agent_2_uprompt.format(topics='\n* '.join(ready), subreddits=subreddits)} ]  # pyright: ignore[reportCallIssue, reportArgumentType] noqa: 501
    resp = await response_text(tmm.complete(msgs, json_schema=AGENT_2_SCHEMA, max_tokens=2048))
    subreddits = json.loads(resp)
    subreddits = subreddits[:MAX_SUBREDDITS]
    # print(resp)

    subreddits = [ s.replace('/r/', '') for s in subreddits]
    print('Stage 3: Summarize hottest 10 titles from the listed subreddits')
    with aiohttp.TCPConnector(ssl=ssl_ctx) as conn:
        session = aiohttp.ClientSession(connector=conn)

        reddit = asyncpraw.Reddit(
            client_id=os.environ.get('REDDIT_CLIENT_ID'),
            client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
            user_agent=USER_AGENT,
            requestor_kwargs={'session': session},  # Custom HTTP session
        )

        posts = []
        for subreddit in subreddits:
            subreddit_api = await reddit.subreddit(subreddit, fetch=True)
            async for submission in subreddit_api.hot(limit=20):  # , time_filter="day"
                # print(submission.title)
                if submission.stickied:  # Pinned post
                    continue
                posts.append(submission)
                # post.id  # id
                # post.selftext  # body

    titles = [post.title for post in posts]

    titles = '\n*  '.join(titles).strip()
    uprompt = AGENT_3_UPROMPT.format(subreddits=subreddits, titles=titles)
    msgs = [ {'role': 'user', 'content': uprompt} ]
    resp = await response_text(tmm.complete(msgs, max_tokens=8096))

    # resp = llm_api(messages=msgs)
    # resp = resp.first_re


    print(resp)

asyncio.run(async_main(toolio_mm))
