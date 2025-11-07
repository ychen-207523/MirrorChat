# MirrorChat

## Steps

```
uv venv
source .venv/bin/activate  
### Windows: .venv\Scripts\activate
uv pip install fastapi uvicorn[standard] pydantic==2.*

```
Create a .env file in the repo root

OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMB=text-embedding-3-small
