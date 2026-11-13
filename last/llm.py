"""One small helper to talk to a local Ollama model (no API key needed)."""
import os

from dotenv import load_dotenv

load_dotenv()


def ask(prompt: str, temperature: float = 0.0) -> str:
    """Send one prompt to Ollama and return the text reply."""
    from ollama import Client

    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    reply = client.chat(
        model=os.getenv("MODEL_NAME", "llama3.2"),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},  # 0 = steady, repeatable answers
    )
    return reply["message"]["content"].strip()
