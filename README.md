AI Assistant framework for OpenAI compatible llms. Uses MCP tools (fastmcp)


config.ini:


```
[assistant]
server_url=http://192.168.1.40:11434
apikey=fake
model=qwen
mcp_servers=[
    "http://192.168.1.40:11000/mcp"
    ]

```


1. use AIKun directly:


```
from ai_kun_kosci.aikun import AIKun
from node_listener.service.config import Config
import asyncio
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = Config('config.ini')


async def main(content):
    assistant = AIKun(
        config.get('assistant.server_url'),
        config.get('assistant.server_key'),
        config.get('assistant.model')
    )

    await assistant.load_mcps(config.get_list('assistant.mcp_servers'))

    response = await assistant.query(content)

    print(dir(response))
    print(response['message'])


if __name__ == "__main__":
    # user_msg = "Hi, what's up?"
    user_msg = "What is the weather in Bielsko?"
    # user_msg = "What is current time ?"
    asyncio.run(main(user_msg))
```


2. use with FastAPI server


```
from ai_kun_kosci.fastapi import init_server, app
from service.config import Config
import asyncio
import uvicorn


async def main():
    config = Config('config.ini')
    await init_server(
        config.get('assistant.server_url'),
        config.get('assistant.server_key'),
        config.get('assistant.model'),
        config.get_list('assistant.mcp_servers')
    )

asyncio.run(main())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

```