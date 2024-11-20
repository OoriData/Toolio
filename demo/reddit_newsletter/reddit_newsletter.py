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

op run --env-file=.env -- python reddit_newsletter.py
'''
import os
import json
import asyncio
import warnings
import ssl
import webbrowser
import tempfile

import aiohttp
import asyncpraw
import markdown
from joblib import Memory, expires_after

from utiloori.ansi_color import ansi_color
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

# from toolio.tool import tool
from toolio.llm_helper import model_manager
from toolio.common import response_text

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Shut up Tokenizers lib warning

# Set up cache
CACHE_AGING = expires_after(hours=1)  # Web cache ages out after an hour
CACHEDIR = './.cache'
job_memory = Memory(CACHEDIR, verbose=0)

# Need this crap to avoid self-signed cert errors (really annoying, BTW!)
ssl_ctx = ssl.create_default_context(cafile=os.environ.get('CERT_FILE'))

# Requires OPENAI_API_KEY in environment
llm_api = openai_chat_api(model='gpt-4o-mini')


def display_html_string(html=None, md=None):
    '''
    Displays the given HTML string in a web browser.

    Args:
        html_string: The HTML string to display.
    '''
    if not any((html, md)):
        raise ValueError('Requires either an HTML or Markdown string')
    if md:
        html = markdown.markdown(md)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        f.write(html.encode('utf-8'))  # reportOptionalMemberAccess
        webbrowser.open('file://' + f.name)


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
  }
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
Where possible, cite your sources using the Reddit links.
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
Here is the schema for you to use: #!JSON_SCHEMA!#
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

Respond using the following schema: #!JSON_SCHEMA!#
'''


async def console_throbber(frame_time: float=0.15):
    'Prints a spinning throbber to console with specified frame time'
    THROBBER_GFX = ["◢", "◣", "◤", "◥"]
    while True:
        for frame in THROBBER_GFX:
            print(f" [{frame}]", end="\r", flush=True)
            await asyncio.sleep(frame_time)


# @job_memory.cache(cache_validation_callback=CACHE_AGING)
async def gather_reddit(topics, all_subreddits):
    # Type maniac linters get on my nerves 🤬
    msgs = [ {'role': 'system', 'content': agent_2_sysprompt},
             {'role': 'user', 'content': agent_2_uprompt.format(topics='\n* '.join(topics), subreddits=all_subreddits)} ]  # pyright: ignore[reportCallIssue, reportArgumentType] noqa: 501
    resp = await response_text(toolio_mm.complete(msgs, json_schema=AGENT_2_SCHEMA, max_tokens=2048))
    subreddits = json.loads(resp)
    subreddits = subreddits[:MAX_SUBREDDITS]
    # print(resp)

    subreddits = [ s.replace('/r/', '') for s in subreddits]
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
            added_posts = []
            async for submission in subreddit_api.hot(limit=15):  # , time_filter="day"
                # print(submission.title)
                if submission.stickied:  # Pinned post
                    continue
                added_posts.append(submission)
                await asyncio.sleep(0.1)
                # post.id  # id
                # post.selftext  # body
            posts.extend(added_posts[:10])
    return posts


async def async_main():
    print('Stage 1: Initial interview (user & agent 1)', flush=True)
    msgs = [ {'role': 'system', 'content': agent_1_sysprompt},
             {'role': 'user', 'content': userprompt_1} ]
    interview_finished = False
    ready = None
    while not interview_finished:
        resp = await response_text(toolio_mm.complete(msgs, json_schema=AGENT_1_SCHEMA, max_tokens=2048))
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
            user_msg = input(ansi_color(question, 'cyan') + '\n')
            print()
            msgs.append({'role': 'user', 'content': user_msg})

    print('Stage 2: Look up subreddits (agents 1 & 2)', flush=True)
    with open('subreddits.json') as fp:
        subreddits = json.load(fp)

    done, _ = await asyncio.wait((
        asyncio.create_task(console_throbber()),
        asyncio.create_task(gather_reddit(ready, subreddits))), return_when=asyncio.FIRST_COMPLETED)

    posts = list(done)[0].result()
    titles = '\n*  '.join([post.title for post in posts]).strip()

    print('Stage 3: Summarize hottest 10 titles from the listed subreddits', flush=True)
    uprompt = AGENT_3_UPROMPT.format(subreddits=subreddits, titles=titles)
    msgs = [ {'role': 'user', 'content': uprompt} ]
    done, _ = await asyncio.wait((
        asyncio.create_task(console_throbber()),
        asyncio.create_task(response_text(toolio_mm.complete(msgs, max_tokens=4096)))),
            return_when=asyncio.FIRST_COMPLETED)

    resp = list(done)[0].result()

    resp = await llm_api(messages=msgs)
    resp = resp.first_choice_text

    print(resp)
    display_html_string(md=resp)


asyncio.run(async_main())
