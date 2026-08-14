from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.file_processor import FileProcessor
from backend.dynamic_rag import DynamicRAG
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Dynamic PDF RAG Assistant",
    description="Upload PDFs/Text files and ask questions",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize components
file_processor = FileProcessor()
rag = DynamicRAG()

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    status: str
    document: str
    chunks: int
    message: str

class QueryResponse(BaseModel):
    answer: str
    document: str
    sources: list

# ============ ENDPOINTS ============

@app.get("/")
def home():
    """Home endpoint"""
    return {
        "title": "Dynamic PDF RAG Assistant",
        "endpoints": {
            "upload": "POST /api/upload",
            "query": "POST /api/query",
            "status": "GET /api/status"
        },
        "docs": "Visit /docs for interactive API"
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF or TXT file and initialize RAG"""
    try:
        # Validate file type
        if file.filename.endswith(('.pdf', '.txt')):
            # Save file
            file_path = file_processor.save_uploaded_file(file, file.filename)
            print(f"File saved: {file_path}")
            
            # Process file
            chunks = file_processor.process_file(file_path)
            
            # Build RAG pipeline
            rag.build_pipeline(chunks, file.filename)
            
            return UploadResponse(
                status="success",
                document=file.filename,
                chunks=len(chunks),
                message=f"Document '{file.filename}' uploaded and processed. Ready to answer questions!"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """Ask a question about the uploaded document"""
    if not rag.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Upload a PDF or TXT file first using /api/upload"
        )
    
    try:
        result = rag.query(request.question)
        return QueryResponse(
            answer=result["answer"],
            document=result["document"],
            sources=result["source_chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def status():
    """Check RAG pipeline status"""
    return {
        "status": "ready" if rag.is_ready() else "waiting_for_upload",
        "current_document": rag.current_document_name,
        "message": "Upload a document to get started" if not rag.is_ready() else f"Ready to answer questions about {rag.current_document_name}"
    }

@app.get("/api/help")
def help():
    """Get usage instructions"""
    return {
        "steps": [
            "1. Upload a PDF or TXT file using POST /api/upload",
            "2. Wait for 'Document uploaded and processed' message",
            "3. Ask questions using POST /api/query",
            "4. Get answers from the LLM based on document content"
        ],
        "supported_formats": ["PDF", "TXT"],
        "example_question": "What is the main topic of this document?",
        "note": "Ensure Ollama is running locally (ollama serve)"
    }

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)