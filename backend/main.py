import os
import io
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from supabase import create_client, Client
from openai import OpenAI


# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dim


def require_env():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {', '.join(missing)}")


# clients (creati se env ok)
oa = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
sb: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY) else None


# -------------------------
# APP
# -------------------------
app = FastAPI(title="Rail Copilot API", version="1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        chunks.append(text[i:end])
        if end == n:
            break
        i = max(0, end - overlap)
    return [c for c in chunks if c.strip()]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if oa is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized (missing OPENAI_API_KEY)")
    resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


@app.get("/health")
def health():
    ok = bool(OPENAI_API_KEY and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    return {"ok": ok, "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}


@app.post("/ingest_pdf")
async def ingest_pdf(
    title: str = "Documento",
    source: str = "upload",
    file: UploadFile = File(...),
):
    require_env()
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported")

    if sb is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages_text = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages_text.append(t)

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF (if scanned, need OCR)")

    doc_res = sb.table("documents").insert({"title": title, "source": source}).execute()
    doc_id = doc_res.data[0]["id"]

    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunking produced no text")

    embeddings = embed_texts(chunks)

    rows = [{
        "document_id": doc_id,
        "chunk_index": i,
        "content": c,
        "embedding": e
    } for i, (c, e) in enumerate(zip(chunks, embeddings))]

    B = 100
    for i in range(0, len(rows), B):
        sb.table("chunks").insert(rows[i:i + B]).execute()

    return {"document_id": doc_id, "chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    require_env()
    if sb is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    q_emb = embed_texts([q])[0]

    res = sb.rpc("match_chunks", {"query_embedding": q_emb, "match_count": int(req.top_k)}).execute()
    hits = res.data or []

    context = "\n\n---\n\n".join([h["content"] for h in hits])

    system = (
        "Sei un Engineering Copilot specializzato in ingegneria ferroviaria. "
        "Rispondi in italiano, in modo tecnico, strutturato e prudente. "
        "Usa prioritariamente il CONTENUTO del contesto fornito. "
        "Se il contesto non contiene la risposta, dichiaralo chiaramente e indica quale documento servirebbe."
    )

    user = f"DOMANDA:\n{q}\n\nCONTESTO (estratti):\n{context}"

    chat = oa.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = chat.choices[0].message.content
    return {"answer": answer, "sources": hits}
