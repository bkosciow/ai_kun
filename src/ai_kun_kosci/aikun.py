from pathlib import Path

from fastmcp import Client as MCPClient
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
from fastmcp.client.client import CallToolResult
from mcp.types import ImageContent, TextContent
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)


def _make_client(transport: str, token: str | None = None):
    if Path(transport).exists() and not transport.endswith(('.py', '.js')):
        return MCPClient(StdioTransport(command=transport, args=[]))
    if token is not None:
        if transport.endswith('/sse'):
            return MCPClient(SSETransport(url=transport, auth=token))
        return MCPClient(StreamableHttpTransport(url=transport, auth=token))
    return MCPClient(transport, )


def _parse_tool_result(result):
    """Convert a fastmcp CallToolResult into OpenAI message content blocks.

    Returns a list of content dicts suitable for the 'content' field of a message.
    Images become image_url blocks, text stays as text.
    """
    if not isinstance(result, CallToolResult):
        return [{
                "type": "text",
                "text": json.dumps(result) if isinstance(result, dict) else str(result),
            }]

    blocks = []
    for item in result.content:
        if isinstance(item, ImageContent):
            data_uri = f"data:{item.mimeType};base64,{item.data}"
            blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
        elif isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        else:
            blocks.append({"type": "text", "text": str(item)})

    return blocks if blocks else [{"type": "text", "text": str(result)}]


class AIKun:
    def __init__(self, server_url: str, api_key, model: str, session_manager=None):
        self.server_url = server_url
        self.model = model
        self.session_manager = session_manager
        self.client = OpenAI(base_url=server_url, api_key=api_key)
        self.tools = []
        self.url_to_tool = {}
        self.url_to_token = {}
        self.mcps = []

    async def load_mcps(self, mcps: list=[]):
        for mcp in mcps:
            await self.load_mcp(mcp)
        self.mcps = mcps

    async def load_mcp(self, mcp_cfg: str | dict):
        mcp_token = None
        if isinstance(mcp_cfg, str):
            mcp_url = mcp_cfg
        else:
            mcp_url = mcp_cfg['url']
            mcp_token = mcp_cfg['token'] if 'token' in mcp_cfg else None

        logger.debug(f"Loading MCP {mcp_url} (token={'set' if mcp_token else 'none'})")
        async with _make_client(mcp_url, mcp_token) as mcp:
            tools_list = await mcp.list_tools()
            self.url_to_token[mcp_url] = mcp_token
            for tool in tools_list:
                self.url_to_tool[tool.name] = mcp_url
                self.tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })

    async def clear_mcps(self):
        self.tools = []
        self.url_to_tool = {}
        self.url_to_token = {}

    async def handle_tools(self, tool_calls: list):
        messages = []
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_response = await self.call_tool(
                self.url_to_tool[tool_name],
                tool_name,
                tool_args
            )
            content = _parse_tool_result(tool_response)
            # Use string for text-only, list for multimodal
            if len(content) == 1 and content[0]["type"] == "text":
                content = content[0]["text"]
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": content,
            })
        return messages

    async def call_tool(self, url: str, tool_name: str, args: dict):
        try:
            token = self.url_to_token.get(url)
            async with _make_client(url, token) as mcp:
                result = await mcp.call_tool(tool_name, args)
                return result
        except Exception as e:
            logger.error(f"Call failed for {tool_name}: {e}")
            return {"error": str(e)}

    async def parse_response(self, response, session: str=None):
        return response

    async def query(self, prompt: str, session: str=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
        )

        msg = response.choices[0].message
        if not msg.tool_calls:
            return await self.parse_response(msg, session)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls}
        ]

        messages += await self.handle_tools(msg.tool_calls)

        followup_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return await self.parse_response(followup_response.choices[0].message, session)

    async def get_models(self):
        try:
            models = self.client.models.list().data
            return [m.id for m in models]
        except Exception as e:
            logger.error(f"Failed to fetch models from server: {e}")
            return []
