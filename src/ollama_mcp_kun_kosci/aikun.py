from fastmcp import Client as MCPClient
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)


class AIKun:
    def __init__(self, server_url: str, api_key, model: str, session_manager=None):
        self.server_url = server_url
        self.model = model
        self.session_manager = session_manager
        self.client = OpenAI(base_url=server_url, api_key=api_key)
        self.tools = []
        self.url_to_tool = {}
        self.mcps = []

    async def load_mcps(self, mcps: list=[]):
        for mcp in mcps:
            await self.load_mcp(mcp)
        self.mcps = mcps

    async def load_mcp(self, mcp_url: str):
        async with MCPClient(mcp_url) as mcp:
            tools_list = await mcp.list_tools()
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
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": json.dumps(tool_response) if isinstance(tool_response, dict) else str(tool_response),
            })
        return messages

    async def call_tool(self, url: str, tool_name: str, args: dict):
        try:
            async with MCPClient(url) as mcp:
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
