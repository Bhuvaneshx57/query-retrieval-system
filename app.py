import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests
import traceback

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBearer()

class RequestBody(BaseModel):
    documents: str
    questions: list[str]

@app.get("/health")
async def health():
    return {"status": "healthy"}

def process_with_ragflow(document_url: str, questions: list[str]) -> list[str]:
    ragflow_url = "https://ragflow-service.onrender.com/api/v1/process"
    payload = {
        "document_url": document_url,
        "questions": questions,
        "embedding_model": "all-mpnet-base-v2",
        "vector_store": "faiss"
    }
    logger.info(f"Sending payload to RAGFlow: {payload}")
    try:
        response = requests.post(ragflow_url, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received response from RAGFlow: {result}")
        return result.get("answers", [])
    except requests.exceptions.RequestException as e:
        logger.error("RAGFlow RequestException: " + str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAGFlow error: {str(e)}")

@app.post("/hackrx/run")
async def run(body: RequestBody, request: Request, auth: HTTPAuthorizationCredentials = Depends(security)):
    logger.info(f"Incoming request from {request.client.host}")
    logger.info(f"Request body: {body}")
    
    expected_token = "f724ae04b606169085d4253d601b61078628048f18963e18daed3844e0a976dd"
    if auth.credentials != expected_token:
        logger.warning("Invalid token attempt")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        answers = process_with_ragflow(body.documents, body.questions)
        return {"answers": answers}
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")
