import io
from pypdf import PdfReader
from app.embeddings import embed_text
from app.supabase_client import supabase

async def ingest_pdf(file, title: str, source: str):
    content = await file.read()
    if not content:
        raise ValueError("Empty file")

    reader = PdfReader(io.BytesIO(content))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    if not text.strip():
        raise ValueError("No extractable text from PDF")

    # 1) Inserisci documento e RICHIEDI esplicitamente il record inserito
    doc_res = (
        supabase
        .table("documents")
        .insert({"title": title, "source": source})
        .select("id")
        .single()
        .execute()
    )

    if not doc_res.data or "id" not in doc_res.data:
        raise RuntimeError(f"documents insert failed or returned no id: {doc_res}")

    document_id = doc_res.data["id"]

    # 2) Embedding
    embedding = embed_text(text[:8000])  # limito per costo/tempo

    # 3) Inserisci chunk col document_id corretto
    chunk_res = (
        supabase
        .table("chunks")
        .insert({
            "document_id": document_id,
            "chunk_index": 0,
            "content": text[:8000],
            "embedding": embedding
        })
        .execute()
    )

    return {
        "status": "ok",
        "document_id": document_id,
        "chunks_inserted": len(chunk_res.data or [])
    }
