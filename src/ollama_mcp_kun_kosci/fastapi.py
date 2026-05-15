from fastapi import FastAPI, HTTPException, status
from ollama_mcp_kun_kosci.aikun import AIKun
from pydantic import BaseModel
from typing import Optional


class PromptRequest(BaseModel):
    prompt: str
    session: Optional[str] = None


app = FastAPI()
assistant = None


async def init_server(server_url: str, apikey: str, model: str, mcps: list):
    global assistant
    assistant = AIKun(server_url, apikey, model)
    await assistant.load_mcps(mcps)


@app.post("/chat")
async def chat(request: PromptRequest):
    if not assistant:
        raise HTTPException(status_code=500, detail="Assistant not initialized")
    try:
        msg = await assistant.query(request.prompt)
        return {"response": {"role": msg.role, "content": msg.content}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
