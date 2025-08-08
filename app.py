import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBearer()

class RequestBody(BaseModel):
    documents: str
    questions: list[str]

def process_with_ragflow(document_url: str, questions: list[str]) -> list[str]:
    ragflow_url = "https://ragflow-service.up.railway.app/api/v1/process"  # Railway RAGFlow URL
    payload = {
        "document_url": document_url,
        "questions": questions,
        "embedding_model": "all-mpnet-base-v2",
        "vector_store": "faiss"
    }
    try:
        response = requests.post(ragflow_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("answers", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"RAGFlow error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"RAGFlow error: {str(e)}")

@app.post("/hackrx/run")
async def run(body: RequestBody, auth: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = "f724ae04b606169085d4253d601b61078628048f18963e18daed3844e0a976dd"
    if auth.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    answers = process_with_ragflow(body.documents, body.questions)
    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
