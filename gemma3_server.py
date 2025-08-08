import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


# Disable TorchDynamo completely (runs model in eager mode)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Local model path
model_path = "../models/Gemma-3-12b-it/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    )

# Initialize FastAPI
app = FastAPI()

# Schema for simple prompt-based text generation
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
async def generate(request: PromptRequest):
    result = pipe(request.prompt, max_new_tokens=request.max_new_tokens)
    return {"response": result[0]["generated_text"].strip()}


# OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def openai_compatible(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Build prompt by concatenating messages (excluding system)
    prompt = "\n".join([m["content"] for m in messages if m["role"] != "system"])
    max_new_tokens = body.get("max_new_tokens", 128)
    print()
    print(prompt)
    print()
    result = pipe(prompt, max_new_tokens=max_new_tokens)
    generated_text = result[0]["generated_text"].strip()
    print(result)
    print("-"*60)
    print()
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,  # Optional to implement token counting
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
