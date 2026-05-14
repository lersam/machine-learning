[[__TOC__]]
# lang_chain

Practical starter examples for building local LangChain agents and chat flows with Ollama models.

This folder contains small, runnable scripts for:

- basic tool-calling agents
- memory and structured responses
- middleware patterns (custom hooks, dynamic prompts, dynamic model selection)
- local RAG-style retrieval with FAISS + Ollama embeddings
- multimodal image description

## Prerequisites

- Python 3.10+ (the code uses `match` statements and modern typing style)
- `pip` and `venv`
- Ollama installed and running locally
- Ollama models referenced by the scripts (for example):
	- `qwen3:14b`
	- `nemotron3:33b`
	- `gemma3:4b-it-qat`
	- `qwen3-embedding:8b`

Install/pull models with Ollama as needed:

```powershell
ollama pull qwen3:14b
ollama pull nemotron3:33b
ollama pull gemma3:4b-it-qat
ollama pull qwen3-embedding:8b
```

## Setup

From `machine-learning` root:

### Windows (PowerShell)

```powershell
cd .\lang_chain
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Command Prompt)

```bat
cd lang_chain
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
cd lang_chain
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run each script from the `lang_chain` directory.

### 1) Basic weather agent with tool calling

```powershell
python main.py
```

Uses `wttr.in` via `httpx` and an Ollama-backed agent.

### 2) Standalone chat model invocation/streaming

```powershell
python stand_alone.py
```

### 3) Agent with memory/context and structured response

```powershell
python memory.py
```

### 4) Dynamic prompt middleware example

```powershell
python middleware_examples.py
```

### 5) Dynamic model selection middleware

```powershell
python select_model_dynamically.py
```

### 6) Custom middleware lifecycle hooks

```powershell
python custom_middleware.py
```

### 7) Local retrieval + embeddings (FAISS + Ollama)

```powershell
python rag_module.py
```

### 8) Multimodal image description

```powershell
python multimodal.py
```

This script reads `lang_chain/ext/dog-breed-height-comparison.jpg`.

## Project Structure

- `main.py`: Minimal agent + weather tool using `create_agent`.
- `stand_alone.py`: Direct chat model usage with regular and streaming responses.
- `memory.py`: Agent context schema, user-location tool, and structured output pattern.
- `middleware_examples.py`: `@dynamic_prompt` middleware based on user role.
- `select_model_dynamically.py`: `@wrap_model_call` middleware that switches models by message count.
- `custom_middleware.py`: Custom `AgentMiddleware` hook lifecycle timing example.
- `rag_module.py`: Ollama embeddings + FAISS similarity search and retriever tool usage.
- `multimodal.py`: Image understanding example using URL/base64 content.
- `ext/`: Local assets used by examples (currently one demo image).
- `requirements.txt`: Python dependencies for this folder.

## Troubleshooting

- Ollama model not found:
	- Pull the missing model (`ollama pull <model_name>`) and re-run the script.
- Cannot connect to Ollama/local model:
	- Confirm Ollama is running locally before running Python scripts.
- `ModuleNotFoundError` for LangChain/FAISS/httpx:
	- Re-activate your virtual environment and run `pip install -r requirements.txt`.
- FAISS install issues on some environments:
	- Ensure you are using a supported Python version and reinstall from `requirements.txt`.
- Weather tool failures in `main.py`/`memory.py`:
	- Check internet connectivity to `wttr.in`.

## Next Steps

- Add a shared config layer for model names to avoid hardcoding in each script.
- Replace print statements in example scripts with structured `logging` consistently.
- Add `.env`-driven settings for model, temperature, and endpoints.
- Add pytest smoke tests for each script entrypoint.
- Expand the RAG example with external document loading and chunking.

## References

- https://docs.langchain.com/oss/python/langchain/middleware/overview
- https://www.youtube.com/watch?v=J7j5tCB_y4w