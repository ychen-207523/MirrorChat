import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title="MirrorChat Backend")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("OPENAI_EMB", "text-embedding-3-small")
NAME = os.getenv("NAME", "User")

PERSONA = (
    f"You are MirrorChat, a concise, practical assistant that speaks like {NAME}. "
    "Keep sentences short and clear. No emojis."
)

emb = OpenAIEmbeddings(model=EMB_MODEL)
vs = Chroma.from_texts(["Seed knowledge for MirrorChat."], embedding=emb)

class IngestItem(BaseModel):
    text: str
    metadata: Optional[dict] = None

class IngestBatch(BaseModel):
    items: List[IngestItem]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

SESSION_HISTORY = {}

def add_history(sid: str, role: str, content: str):
    SESSION_HISTORY.setdefault(sid, []).append((role, content))
    if len(SESSION_HISTORY[sid]) > 20:
        SESSION_HISTORY[sid] = SESSION_HISTORY[sid][-20:]

def history_msgs(sid: str):
    return [{"role": r, "content": c} for r, c in SESSION_HISTORY.get(sid, [])]

llm = ChatOpenAI(model=MODEL, temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", PERSONA),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
    ("system", "Use this context if helpful:\n{context}")
])

def chain():
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    return (
        {
            "question": RunnablePassthrough(),
            "history": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
        }
        | prompt
        | llm
    )

@app.get("/health")
def health():
    return {"ok": True, "service": "mirrorchat", "version": 1}

@app.post("/ingest")
def ingest(batch: IngestBatch):
    docs = [Document(page_content=i.text, metadata=i.metadata or {}) for i in batch.items]
    vs.add_documents(docs)
    return {"added": len(docs)}

@app.post("/chat")
def chat(req: ChatRequest):
    add_history(req.session_id, "user", req.message)
    resp = chain().invoke({
        "question": req.message,
        "history": history_msgs(req.session_id)
    })
    text = getattr(resp, "content", str(resp))
    add_history(req.session_id, "assistant", text)
    return {"reply": text, "session_id": req.session_id}
