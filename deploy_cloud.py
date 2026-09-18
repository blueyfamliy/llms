import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi, upload_folder
from pydantic import BaseModel

from llms.sample import generate_text, load_model
from llms.tokenizer import BPETokenizer

app = FastAPI(title="LLMS Cloud API")

# Global variables for model and tokenizer
MODEL = None
CONFIG = None
TOKENIZER = None

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 50

def init_model(ckpt_path="out/tiny/ckpt.pt"):
    global MODEL, CONFIG, TOKENIZER
    print(f"Loading model from {ckpt_path}...")
    MODEL, CONFIG = load_model(ckpt_path, torch.device("cpu"))
    TOKENIZER = BPETokenizer.load(CONFIG.data.tokenizer_path)
    print("Model loaded successfully.")

@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/generate")
async def generate(request: GenerationRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        result = generate_text(
            MODEL,
            TOKENIZER,
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k
        )
        return {"generated_text": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

def upload_to_hf(repo_id, folder_path, token):
    """Uploads model checkpoints and tokenizer to Hugging Face Hub."""
    print(f"Uploading {folder_path} to HF repo {repo_id}...")
    HfApi()
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print("Upload complete!")

if __name__ == "__main__":
    import sys

    import uvicorn

    if len(sys.argv) > 1:
        if sys.argv[1] == "upload":
            # Usage: python deploy_cloud.py upload <repo_id> <folder> <token>
            upload_to_hf(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "serve":
            # Usage: python deploy_cloud.py serve
            uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Usage:")
        print("  python deploy_cloud.py upload <repo_id> <folder> <token>")
        print("  python deploy_cloud.py serve")
