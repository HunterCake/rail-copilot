import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Rail Engineering Copilot API",
    version="0.1.0",
)

# -----------------------------------------------------------------------------
# CORS (necessario per chiamate dal browser - Vercel)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # OK per MVP, restringere in produzione
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# OpenAI client
# -----------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# -----------------------------------------------------------------------------
# Schemi
# -----------------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------------------------------------------------------
# Chat endpoint
# -----------------------------------------------------------------------------
@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "Sei un assistente tecnico di ingegneria ferroviaria. "
                    "Rispondi in italiano, in modo tecnico e conciso."
                ),
            },
            {
                "role": "user",
                "content": payload.message,
            },
        ],
    )

    return {"answer": response.output_text}
