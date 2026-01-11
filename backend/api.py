from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
from typing import List, Optional
from agent import get_agent
import httpx
from PIL import Image
from io import BytesIO

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UrlRequest(BaseModel):
    url: str
    agent_type: str = "exploratory-v4"
    n_colors: int = 5

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    agent_type: str = "exploratory-v4",
    n_colors: int = 5
):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        agent = get_agent(agent_type, n_colors)
        result = agent.extract_palette(file_path)
        
        # We don't need to keep the file
        os.remove(file_path)
        
        return format_result(result)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-url")
async def process_url(request: UrlRequest):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_download.jpg")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Could not download image from URL")
            
            with open(file_path, "wb") as f:
                f.write(response.content)
        
        agent = get_agent(request.agent_type, request.n_colors)
        result = agent.extract_palette(file_path)
        
        os.remove(file_path)
        return format_result(result)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

def format_result(result):
    # Convert result to JSON-serializable format
    # The result has 'image', 'colors', 'metadata'
    # We need to return colors and maybe the resized image as base64 or just dimensions
    return {
        "colors": result["colors"],
        "metadata": {
            "agent": result["metadata"]["agent"],
            "image_size": result["metadata"]["image_size"],
            "method": result["metadata"]["method"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

