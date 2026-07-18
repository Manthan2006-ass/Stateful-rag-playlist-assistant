from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.responses import StreamingResponse
# FIXED: Corrected file name to 'incomprocess' and function to 'query_rag_assistant_stream'
from incompress import query_rag_assistant_stream

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]]

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/query")
def query(req: QueryRequest):
    # Pass the question and history parameters into the stream generator
    return StreamingResponse(
        query_rag_assistant_stream(req.question, req.history), 
        media_type="text/plain"
    )