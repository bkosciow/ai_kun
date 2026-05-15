from ollama_mcp_kun_kosci.aikun import AIKun
import asyncio
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def main(content):
    assistant = AIKun(
        'http://192.168.1.106:11434',
        'fake_key',
        'qwen'
    )

    await assistant.load_mcps([
        "http://192.168.1.40:11000/mcp"
    ])

    msg = await assistant.query(content)

    print("Role:", msg.role)
    print("Content:", msg.content)
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print("Tool calls:", msg.tool_calls)


if __name__ == "__main__":
    # user_msg = "Hi, what's up?"
    user_msg = "What is current time ?"
    asyncio.run(main(user_msg))
