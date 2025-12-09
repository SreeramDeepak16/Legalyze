# backend.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import List, Optional
from dotenv import load_dotenv



load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# 👇 import your query agent + workflow here
# adjust these imports to match your project structure
from query_agent import QueryAgent     # instance that has get_complete_query()
from workflow import workflow            # function that runs the full LMARS workflow

query_agent = QueryAgent()
app = FastAPI()

# Create LLM normally (do NOT set stream=True here)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------------------------------------------------
# STREAMING ENDPOINT (existing)
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

# ---------------------------------------------------
# LMARS QUERY ENDPOINT
# ---------------------------------------------------

class LMARSQueryRequest(BaseModel):
    query: str
    # Only sent on the SECOND call, after follow-ups are answered
    followup_questions: Optional[List[str]] = None
    followup_answers: Optional[List[str]] = None


@app.post("/lmars/query")
async def lmars_query(payload: LMARSQueryRequest):
    """
    Main LMARS endpoint.

    First call:
      - Takes a raw user query.
      - Uses QueryAgent to decide if it's sufficient.
      - If NOT sufficient -> returns follow-up questions.
      - If sufficient -> runs the workflow and returns final summary.

    Second call:
      - Takes original query + followup_questions + followup_answers.
      - Uses QueryAgent.generate_new_query() to build final query.
      - Runs workflow(final_query) and returns final summary.
    """
    user_query = payload.query

    # ---------- Phase 2: we already have follow-up questions & answers ----------
    if payload.followup_questions and payload.followup_answers:
        # Use your QueryAgent to build the final query
        final_query = query_agent.generate_new_query(
            prev_query=user_query,
            ques_list=payload.followup_questions,
            ans_list=payload.followup_answers,
        )

        summary = workflow(final_query)

        return {
            "status": "answer",
            "sufficiency": "Enough",
            "final_query": final_query,
            "summary": summary,
        }

    # ---------- Phase 1: just the original query ----------
    result = query_agent.get_complete_query(user_query)
    sufficiency = result.get("sufficiency")

    if sufficiency == "Not Enough":
        follow_ups = result.get("follow_up_questions", [])
        return {
            "status": "followup_needed",
            "sufficiency": sufficiency,
            "follow_up_questions": follow_ups,
        }

    # Sufficient -> run full LMARS workflow and return summary
    summary = workflow(user_query)

    return {
        "status": "answer",
        "sufficiency": sufficiency,
        "final_query": user_query,
        "summary": summary,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
