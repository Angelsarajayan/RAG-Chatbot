from config import JINA_URL,JINA_API,CHUNKS
import requests
import json
import logging

logger = logging.getLogger(__name__)

def generate_embeddings(chunks):
    """
    Sends text chunks to Jina AI embedding API.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[Dict]: Embeddings response.
    """
    data = {
        "input": chunks,
        "model": "jina-embeddings-v3"
    }
    headers = {
        "Authorization": f"Bearer {JINA_API}",
        "Content-Type": "application/json"
    }
    response = requests.post(JINA_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        raise ValueError(f"API Error {response.status_code}: {response.text}")

try:
    with open(CHUNKS, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = [chunk.strip() for chunk in content.split('***') if chunk.strip()]

    embeddings = generate_embeddings(chunks)

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    logger.info("Embeddings saved to 'embeddings.json'.")

except Exception as e:
    logger.critical(f"Embedding generation failed: {e}")