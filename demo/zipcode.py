'''
demo/zipcode.py

I asked Nous Hermes Theta about the zip code where I live, and it proceded to get the city wrong and tell me a whole lot about the wrong place.

```sh
❯ toolio_request --apibase="http://localhost:8000" \                                           
--prompt='Tell me something about life in zip code 80027'
The zip code 80027 is located in the city of Westminster, Colorado. Westminster is a suburban city situated in the northern part of the Denver-Aurora-Lakewood, CO Metropolitan Statistical Area. Life in zip code 80027 offers a mix of urban and suburban amenities, with easy access to the Denver metropolitan area.

[SNIP 7 more paragraphs of irrelevant info]
```

Let's make sure it has up to date info.

```sh
❯ python demo/zipcode.py
⚙️ Calling tool zip_code_info with args {'code': '80027'}
⚙️ Tool call result: LOUISVILLE, CO, US
Here are some details about life in the 80027 zip code, which is located in Louisville, Colorado:

**Population:** As of the 2020 Census, the population of Louisville, CO (which includes the 80027 zip code) was 19,955.

**Median Age:** The median age in Louisville is 41.5 years, which is slightly higher than the national median age of
38.5 years.

**Gender Ratio:** The gender ratio in Louisville is nearly equal, with 50.3% of the population being male and 
```

Cuts off for max length, but we've accomplished what we wanted. Well, except that I actually live in Superior
(which shares a zip with Louisville), but that's a nitpick for another project. 😉
'''
import asyncio
from math import sqrt
from toolio.tool import tool, param
from toolio.llm_helper import model_manager, extract_content

import httpx

# Yup. Still HTTP
ZIPCODE_ENDPOINT = 'http://ZiptasticAPI.com/'

@tool('zip_code_info', params=[param('code', str, 'The zip code to look up. Must be only the 5 numbers', True)])
async def zip_code_info(code=None):
    'Look up city, state and country from a zip code'
    assert code.isdigit()  # Only allow numbers
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.get(ZIPCODE_ENDPOINT + str(code))
        info = resp.json()  # e.g. {"country":"US","state":"CO","city":"LOUISVILLE"}

    # return f'{info['city']}, {info['state']}, {info['country']}'
    return f'{info["city"]}, {info["state"]}, {info["country"]}'

# Had a problem using Hermes-2-Theta-Llama-3-8B-4bit 😬
# MLX_MODEL_PATH = 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'
MLX_MODEL_PATH = 'mlx-community/Mistral-Nemo-Instruct-2407-4bit'

toolio_mm = model_manager(MLX_MODEL_PATH, tool_reg=[zip_code_info], trace=True)

PROMPT = 'Tell me something about life in zip code 80027'
async def async_main(tmm):
    msgs = [ {'role': 'user', 'content': PROMPT} ]
    async for chunk in extract_content(tmm.complete_with_tools(msgs)):
        print(chunk, end='')

asyncio.run(async_main(toolio_mm))
