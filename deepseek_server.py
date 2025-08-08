import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from llama_cpp import Llama

# CUDA setup
os.environ["LLAMA_CUDA_DEV"] = "0"
os.environ["LLAMA_CUDA_FORCE_MGPU"] = "0"

# Load model
llm = Llama(
    model_path="../models/Deepseek/DeepSeek-R1-Distill-Llama-8B-F16.gguf",
    n_gpu_layers=-1,
    verbose=False
)

# FastAPI app
app = FastAPI()

# Request schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

# Route to call LLaMA
@app.post("/generate")
async def generate(request: PromptRequest):
    result = llm(request.prompt, max_tokens=request.max_tokens)
    return {"response": result["choices"][0]["text"].strip()}

@app.post("/v1/chat/completions")
async def openai_compatible(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = "\n".join([m["content"] for m in messages if m["role"] != "system"])
    max_tokens = body.get("max_tokens", 128)

    print()
    print(messages)
    print()
    
    
    result = llm(prompt, max_tokens=max_tokens)
    
    print(result)
    print("-"*60)
    print()
    
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["choices"][0]["text"].strip()},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
