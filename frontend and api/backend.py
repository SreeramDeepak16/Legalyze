# backend.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import uvicorn
from fastapi import File, UploadFile
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

# Create LLM normally (do NOT set stream=True here)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------------------------------------------------
# STREAMING ENDPOINT
# ---------------------------------------------------

async def gemini_stream(prompt: str):
    """Correct streaming implementation using astream()."""
    try:
        # astream() gives an async generator of chunks
        async for chunk in llm.astream(prompt):
            text = chunk.content or ""
            tokens = text.split(" ")

            for i in range(0, len(tokens), 3):  
                yield " ".join(tokens[i:i+3]) + " "
                await asyncio.sleep(0)
    except Exception as e:
        yield f"[Error in streaming]: {str(e)}"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stream")
async def stream(message: str = ""):
    return StreamingResponse(
        gemini_stream(message),
        media_type="text/plain"
    )

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        content = await f.read()
        with open(f"uploads/{f.filename}", "wb") as w:
            w.write(content)
        saved.append(f.filename)
    return {"saved": saved}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
