from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector_store import VectorStore
from llm_handler import LLMHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()
llm = LLMHandler()

class ChatRequest(BaseModel):
    message: str

class KnowledgeRequest(BaseModel):
    text: str
    metadata: dict = None

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests"""
    try:
        search_results = vector_store.search(request.message)
        
        context = "\n".join([
            match['metadata']['text'] 
            for match in search_results['matches']
        ]) if search_results['matches'] else ""
        
        response = llm.generate_response(request.message, context)
        
        return {
            "response": response,
            "context_used": bool(context)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge")
async def add_knowledge(request: KnowledgeRequest):
    """Add knowledge to the vector database"""
    try:
        vector_id = vector_store.add_knowledge(request.text, request.metadata)
        return {"status": "success", "vector_id": vector_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)