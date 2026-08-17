# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

`ai_kun_kosci` (v0.0.2) is an AI assistant framework that bridges an **OpenAI-compatible API** (e.g. Ollama's `/v1` endpoint) with MCP (Model Context Protocol) tool servers. It dynamically discovers MCP tools, maps them to the OpenAI function calling schema, and dispatches tool calls at runtime. Supports **multimodal tool responses** (images + text) and both **network and stdio MCP transports**.

Two usage modes:
- **Direct:** instantiate `AIKun` and call `query()` in an async context
- **FastAPI server:** call `init_server()` then run with `uvicorn`

## Source Structure

| File | Purpose |
|------|---------|
| `src/ai_kun_kosci/aikun.py` | Core `AIKun` class — loads MCP tools via `fastmcp.Client`, maps them to OpenAI function calling schema, handles tool call dispatch with multimodal support |
| `src/ai_kun_kosci/fastapi.py` | FastAPI app with single `POST /chat` endpoint, `init_server()` for startup |
| `src/simple_ask.py` | Standalone CLI example — instantiates `AIKun`, loads MCPs, queries, and prints the response |

## Key Dependencies

Declared in `pyproject.toml`: `openai>=1.0.0`, `fastmcp>=0.4.0`, `fastapi>=0.100.0`, `uvicorn>=0.20.0`. Requires **Python >= 3.13**. No `ollama` dependency — the framework uses the OpenAI SDK against any compatible endpoint.

## Running the FastAPI Server

```bash
uvicorn ai_kun_kosci.fastapi:app --host 0.0.0.0 --port 8001
```

Before requests work, `init_server(server_url, apikey, model, mcps)` must be called. `server_url` should point to an OpenAI-compatible `/v1` endpoint (e.g. Ollama at `http://host:11434/v1`). The endpoint accepts `POST /chat` with body `{"prompt": "...", "session": "..."}`.

## Architecture Notes

- `AIKun` constructor: `AIKun(server_url, api_key, model, session_manager=None)` — uses `openai.OpenAI(base_url=server_url, api_key=api_key)`
- **MCP transport**: `_make_client(transport, token=None)` auto-detects the transport type — a local file path (not `.py`/`.js`) uses `StdioTransport`, everything else (URLs) uses the default HTTP transport. With a token it builds `StreamableHttpTransport(url, auth=token)` (or `SSETransport` for `/sse` URLs), which sends `Authorization: Bearer <token>`
- **Tool loading**: `load_mcps(mcps)` accepts plain URL strings or `{url, token}` dicts, delegates to per-server `load_mcp()`, which uses `async with` for proper connection lifecycle. Per-URL tokens are stored in `url_to_token` so `call_tool()` re-authenticates when it opens a fresh connection per call. `clear_mcps()` resets tools, mappings, and tokens
- **Multimodal tool responses**: `_parse_tool_result()` converts `fastmcp CallToolResult` into OpenAI message content blocks — `ImageContent` becomes `image_url` data URI blocks, `TextContent` stays as text. `handle_tools()` uses a plain string for single-text results and a list for multimodal content
- **`query()` flow**: send prompt via OpenAI SDK with tool definitions → if the model requests tool calls, execute each via `fastmcp.Client` (creates a new connection per call at `aikun.py:99`) → parse results as multimodal content → send follow-up call with full conversation context → return the message object
- **`query()` return**: the raw OpenAI message object with `.role`, `.content`, and optionally `.tool_calls`
- Tool-to-MCP mapping: `url_to_tool` dict (tool name → MCP server URL/path)
- `AIKun.get_models()` lists available models via `self.client.models.list().data`
- `session` parameter on `query()` and `parse_response()` is accepted but not yet used
- `session_manager` constructor parameter is stored but not yet used
- `parse_response()` is a no-op override point for subclasses
- `config.ini` is in `.gitignore` — configuration is expected per-env
- No tests exist in the project currently
