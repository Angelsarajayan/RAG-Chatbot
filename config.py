import os
from dotenv import load_dotenv


load_dotenv()


JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./mtech_chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MTECH_PROSPECTUS")

TOP_K = int(os.getenv("TOP_K", 5))  

LLM_BACKEND = os.getenv("LLM_BACKEND", "groq") 
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "jina") 
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jina-embeddings-v3")

PDF = os.getenv("PDF", "PROSPECTUS FOR ADMISSION TO M.Tech. PROGRAMMES.pdf")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

JINA_URL = os.getenv("JINA_URL")
JINA_API = os.getenv("JINA_API")

EXTRACTED_TEXT = os.getenv("EXTRACTED_TEXT")
CHUNKS = os.getenv("CHUNKS")
EMBEDDINGS = os.getenv("EMBEDDINGS")