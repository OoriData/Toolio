# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/demo/researcher/main
'''
Demo using Toolio for structured interaction between parts of a multi-agent research
pipeline. The researcher combines LLM and web search to investigate a query.

1. Uses async/await throughout for efficient I/O
2. Toolio-guaranteed structured JSON schemas for LLM outputs
3. Chain-of-thought research planning upfront
4. Comprehensive tracing system recording every step
5. Configurable rigor level to control research depth
6. SearXNG integration (with optional Toolio tool features)
7. Clean separation of concerns between components

Research process:

1. Takes initial query and plans research steps using a structured schema
2. Executes planned steps combining web search and analysis
3. Can adaptively add steps based on findings
4. Uses rigor parameter to control depth/thoroughness
5. Maintains detailed trace of all operations

Usage:

```bash
python demo/tee_seek/main.py "Impact of regenerative agriculture on soil health" --rigor 0.6
python demo/tee_seeker/main.py "What's so valuable about DeepSeek's GRPO technique?" --rigor 0.5
```
'''
import os
import json
import asyncio
# from pathlib import Path
from datetime import datetime
import logging

import fire
import httpx

# from toolio import load_or_connect
from toolio.llm_helper import local_model_runner
from toolio.tool import tool, param

# Settings that could be moved to config
SEARXNG_ENDPOINT = os.getenv('SEARXNG_ENDPOINT', 'http://localhost:8888/search')
RESULTS_PER_QUERY = 3  # Number of results to process per search
MAX_STEPS = 10  # Maximum steps before forcing termination
MIN_STEPS = 2  # Minimum steps before allowing early termination
DEFAULT_TRACE_FILE = 'tee_seek_trace.json'

# MODEL_DEFAULT = 'mlx-community/deepseek-r1-distill-qwen-1.5b-4bit'
# MODEL_DEFAULT = 'mlx-community/deepseek-r1-distill-qwen-1.5b'  # Think this is the same as mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16
# MODEL_DEFAULT = 'mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit'
# MODEL_DEFAULT = 'mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit'
# MODEL_DEFAULT = 'mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit'
MODEL_DEFAULT = 'mlx-community/Mistral-Nemo-Instruct-2407-4bit'

LAUNCH_PROMPT_TEMPLATE = '''\
You are a research assistant tasked with investigating the following main query:
{query}
Your task is to plan a series of research steps to thoroughly investigate this query.
Consider the following:
1. What initial information do we have?
2. What additional information do we need?
3. What steps will you take to gather this information?

Start by responding with a list of tasks, and only a list of tasks. You can work on additional steps, and the conclusion later.
Keep it brief!
'''
# Keep your response short, because you'll have a chance to elaborate as we go along.

# Simplified schema for research planning response
PLAN_SCHEMA = {
    'type': 'object', 
    'properties': {
        'research_steps': {
            'type': 'array',
            'maxItems': 5,  # Limit number of steps
            'items': {
                'type': 'object',
                'properties': {
                    'step_type': {'type': 'string', 'enum': ['web_search', 'analysis']},
                    'query': {'type': 'string', 'maxLength': 100},  # Limit query length
                    'purpose': {'type': 'string', 'maxLength': 150}  # Brief explanation
                },
                'required': ['step_type', 'query', 'purpose']
            }
        },
        'goal': {'type': 'string', 'maxLength': 200}  # Brief success criteria
    },
    'required': ['research_steps', 'goal']
}

# Add a system prompt to encourage conciseness
PLAN_SYSPROMPT = '''Create a focused research plan with specific search queries.
Keep explanations brief and precise. Limit to 3-5 key search steps.'''


# Schema for research planning response
PLAN_PLUS_SCHEMA = {
    'type': 'object', 
    'properties': {
        'initial_thoughts': {'type': 'string'},
        'research_steps': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'step_type': {'type': 'string', 'enum': ['web_search', 'analysis', 'conclusion']},
                    'query': {'type': 'string'},
                    'reasoning': {'type': 'string'}
                },
                'required': ['step_type', 'reasoning']
            }
        },
        'success_criteria': {'type': 'string'}
    },
    'required': ['initial_thoughts', 'research_steps', 'success_criteria']
}


# Schema for analyzing search results
ANALYSIS_SCHEMA = {
    'type': 'object',
    'properties': {
        'key_findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'finding': {'type': 'string'},
                    'sources': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'title': {'type': 'string'},
                                'url': {'type': 'string'},
                                'relevance': {'type': 'string', 'enum': ['primary', 'supporting', 'related']}
                            },
                            'required': ['title', 'url', 'relevance']
                        }
                    }
                },
                'required': ['finding', 'sources']
            }
        },
        'additional_questions': {
            'type': 'array', 
            'items': {'type': 'string'}
        },
        'research_complete': {'type': 'boolean'},
        'completion_reasoning': {'type': 'string'}
    },
    'required': ['key_findings', 'research_complete', 'completion_reasoning']
}


#Note: Not actually calling this via LLM tool-calling, but no harm declaring it that way
@tool('searxng_search',
      desc='Run Web search query and get relevant results',
      params=[param('query', str, 'search query text', True)])
async def searxng_search(query):
    '''Execute a SearXNG search and return results'''
    qparams = {
        'q': query,
        'format': 'json',
        'categories': ['general'],
        'engines': ['qwant', 'duckduckgo', 'bing']
    }

    async with httpx.AsyncClient() as client:
        # Execute search
        resp = await client.get(SEARXNG_ENDPOINT, params=qparams)
        results = resp.json()['results'][:RESULTS_PER_QUERY]

        # Get content for each result
        processed = []
        for r in results:
            try:
                content_resp = await client.get(r['url'])
                content = content_resp.text
                processed.append({
                    'title': r['title'],
                    'url': r['url'],
                    'snippet': r.get('snippet', ''),
                    'published_date': r.get('publishedDate'),
                    'content': content[:2000],  # Truncate for LLM context
                    'engine': r.get('engine'),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logging.warning(f'Error fetching {r["url"]}: {e}')

        return processed


class tee_seeker:
    def __init__(self, llm, trace_file=DEFAULT_TRACE_FILE):
        self.llm = llm
        self.trace_file = trace_file
        self.trace = []
        self.sources = {}

    async def research(self, query, rigor=0.5):
        '''
        Main research loop
        rigor: 0.0-1.0 controls how thorough the research should be
        '''
        # Plan initial research steps
        # full_query = LAUNCH_PROMPT_TEMPLATE.format(query=query)
        # plan = await self.llm(full_query, json_schema=PLAN_SCHEMA, max_tokens=8192)
        full_query = f'Plan research steps to answer: {query}'
        plan = await self.llm(full_query, 
                            json_schema=PLAN_SCHEMA,
                            sysprompt=PLAN_SYSPROMPT, 
                            max_tokens=2048)  # Reduced token limit
        print(plan)
        plan = json.loads(plan)
        self._trace('plan', plan)

        completed_steps = 0
        findings_with_sources = []

        for step in plan['research_steps']:
            if completed_steps >= MAX_STEPS:
                break

            self._trace('step_start', step)

            if step['step_type'] == 'web_search':
                results = await searxng_search(step['query'])
                self._trace('search_results', results)

                # Track sources
                for r in results:
                    self.sources[r['url']] = {
                        'title': r['title'],
                        'url': r['url'],
                        'snippet': r.get('snippet', ''),
                        'published_date': r.get('published_date'),
                        'engine': r.get('engine'),
                        'accessed': r.get('timestamp')
                    }

                # Analyze results, encouraging proper citation
                analysis_prompt = f'''Analyze these search results. For each key finding:
                1. State the finding clearly and concisely
                2. Link it to specific sources that support it
                3. Indicate how each source relates to the finding (primary evidence, supporting detail, or related context)

                Search results: {json.dumps(results)}'''
                # "Keep it succinct, with just 3-5 main takeaways"

                analysis = await self.llm(analysis_prompt, json_schema=ANALYSIS_SCHEMA, max_tokens=8192)
                try:
                    analysis = json.loads(analysis)
                except json.JSONDecodeError as e:
                    raise
                self._trace('analysis', analysis)

                findings_with_sources.extend(analysis['key_findings'])

                # Possibly finish early if we have enough info
                if (completed_steps >= MIN_STEPS and 
                    analysis['research_complete'] and
                    rigor < 0.8):
                    break

            completed_steps += 1

        # Synthesis. Preserve citations
        synthesis_prompt = f'''Create a detailed research summary that:
        1. Presents key findings with their supporting evidence
        2. Cites specific sources for each major claim
        3. Notes where multiple sources corroborate findings
        4. Includes a "Sources Cited" section at the end

        Research findings: {json.dumps(findings_with_sources)}'''

        summary = await self.llm(synthesis_prompt, max_tokens=8192)
        print('-'*80)
        self._trace('summary', {
            'text': summary,
            'sources': self.sources
        })

        return summary

    def _trace(self, step_type, data):
        '''Record a trace entry with source tracking'''
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': step_type,
            'data': data
        }
        self.trace.append(entry)

        # Include source index in trace file
        trace_data = {
            'steps': self.trace,
            'sources': self.sources
        }

        # Write updated trace
        with open(self.trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)


def main(query, model: str = MODEL_DEFAULT, rigor: float = 0.5, trace_file: str=DEFAULT_TRACE_FILE):
    '''
    Research tool combining LLMs and web search

    Command line args:
    query: Research query
    model: Model ID or Toolio server URL
    trace-file: JSON file to record execution trace
    rigor: Research rigor level (0.0-1.0)
    '''
    # llm = load_or_connect(model)
    llm = local_model_runner(model)
    researcher = tee_seeker(llm, trace_file)
    summary = asyncio.run(researcher.research(query, rigor))
    print(f'\nResearch Summary:\n{summary}')


if __name__ == '__main__':
    fire.Fire(main)
