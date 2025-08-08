#  LLM Model Server

Run your own model-serving powerhouse. This repository lets you spin up a fully OpenAI-compatible server using FastAPI and GPU scheduling, delivering JSON-safe outputs and health-monitoring endpoints.

---

##  What This Does

| Feature | Description |
|---------|-------------|
|  Model Host | Serves LLMs (e.g., Gemma, DeepSeek) via HTTP endpoints |
|  JSON Guardrails | Ensures safe, structured JSON outputs |
|  GPU Routing | Intelligent routing across server-level GPUs |
|  OpenAI-Compatible | Uses same input/output conventions as OpenAI API for seamless integration |
|  Health & Metrics | Built-in monitoring endpoints for observability |

---

##  Quick Start

```bash
git clone https://github.com/sidhu66/LLM_Model_Server
cd LLM_Model_Server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Choose server:
# For DeepSeek:
./run_deepseek_server.sh

# For Gemma:
./run_gemma3_server.sh
