from app.embeddings import embed_text
from app.supabase_client import supabase
from PyPDF2 import PdfReader
import uuid

async def ingest_pdf(title: str, source: str, file):
    reader = PdfReader(file.file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        supabase.table("chunks").insert({
            "document_id": str(uuid.uuid4()),
            "chunk_index": idx,
            "content": chunk,
            "embedding": embedding
        }).execute()

    return {
        "status": "ok",
        "chunks": len(chunks)
    }
