from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed_text(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding
