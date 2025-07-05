from config import EMBEDDINGS, CHUNKS, CHROMA_DIR
import json
import chromadb
import logging

logger = logging.getLogger(__name__)

def detect_department(text):
    """Infer department name from chunk text."""
    lower = text.lower()
    if "computer science" in lower or "digital image computing" in lower:
        return "Computer Science"
    if "futures studies" in lower or "technology management" in lower:
        return "Futures Studies"
    if "optoelectronics" in lower or "electronics and communication" in lower:
        return "Optoelectronics"
    return "General"

def detect_course(text):
    """Infer specific course name from chunk text."""
    lower = text.lower()
    if "digital image computing" in lower:
        return "M.Tech Computer Science with Specialization in Digital Image Computing"
    if "technology management" in lower:
        return "M.Tech Technology Management"
    if "optoelectronics" in lower or "electronics and communication" in lower:
        return "M.Tech Electronics and Communication (Optoelectronics and Optical Communication)"
    return "General"

def detect_section(text):
    """Identify document section like eligibility, fees, etc."""
    lower = text.lower()
    keywords = {
        "Eligibility": ["eligibility"],
        "Fees": ["fee", "tuition"],
        "Reservation": ["reservation"],
        "Important Dates": ["important dates", "notification"],
        "Application Process": ["application"],
        "Admission Procedure": ["admission", "how to apply"],
        "Entrance Exam": ["entrance"],
        "Rank List": ["rank list"]
    }
    for label, keys in keywords.items():
        if any(k in lower for k in keys):
            return label
    return "General"

def detect_topic_type(text):
    """Classify chunk as instruction vs department-specific."""
    lower = text.lower()
    if any(keyword in lower for keyword in [
        "online application", "entrance", "admit card", "admission memo", "about the university",
        "important information", "admision activities", "fee payment", "instructions", "apply online",
        "upload", "hall ticket", "how to apply", "rank list", "reservation"
    ]):
        return "Instruction"
    return "Department-Specific"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="MTECH_PROSPECTUS")

    with open(CHUNKS, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split('***') if c.strip()]

    with open(EMBEDDINGS, "r", encoding="utf-8") as f:
        embeddings = [e["embedding"] for e in json.load(f)]

    assert len(chunks) == len(embeddings), "Mismatch in chunk and embedding counts!"

    records = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "department": detect_department(chunk),
            "course": detect_course(chunk),
            "section": detect_section(chunk),
            "topic_type": detect_topic_type(chunk),
            "source": "MTech Prospectus 2024",
            "index": i
        }
        records.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "embedding": embeddings[i],
            "metadata": metadata
        })

    collection.add(
        documents=[r["text"] for r in records],
        embeddings=[r["embedding"] for r in records],
        metadatas=[r["metadata"] for r in records],
        ids=[r["id"] for r in records]
    )

    with open("chromadb_metadata.json", "w", encoding="utf-8") as f:
        json.dump([r["metadata"] for r in records], f, indent=2)

    logger.info("Chunks added to ChromaDB with metadata.")

except Exception as e:
    logger.critical(f"ChromaDB operation failed: {e}")
