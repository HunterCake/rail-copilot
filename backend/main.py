import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class ChatIn(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn):
    resp = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "system",
                "content": "Sei un assistente tecnico di ingegneria ferroviaria. Rispondi in italiano, in modo tecnico e conciso."
            },
            {
                "role": "user",
                "content": payload.message
            }
        ],
    )
    return {"answer": resp.output_text}
