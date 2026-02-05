from fastapi import FastAPI, UploadFile, File, Query
from app.ingest import ingest_pdf

app = FastAPI(title="Rail Assistant API")

@app.get("/health")
def health():
    return {
        "ok": True,
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small"
    }

@app.post("/ingest_pdf")
async def ingest_pdf_endpoint(
    title: str = Query(...),
    source: str = Query("upload"),
    file: UploadFile = File(...)
):
    return await ingest_pdf(title=title, source=source, file=file)
