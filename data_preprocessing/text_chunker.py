from config import HUGGINGFACE_TOKEN,EXTRACTED_TEXT
import logging
from huggingface_hub import login
from transformers import AutoTokenizer
import re

logger = logging.getLogger(__name__)
login(token=HUGGINGFACE_TOKEN)

def custom_sent_tokenize(text):
    """
    Splits text into sentences based on punctuation.

    Args:
        text (str): Full paragraph text.

    Returns:
        List[str]: List of sentence strings.
    """
    sentence_endings = r'(?<=[.!?]) +'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_chunk_by_tokens(sentences, encoder, max_tokens=512, overlap=50):
    """
    Chunks a list of sentences into groups based on token limits.

    Args:
        sentences (List[str]): Tokenized sentences.
        encoder: Tokenizer model.
        max_tokens (int): Max token limit per chunk.
        overlap (int): Overlap tokens between chunks.

    Returns:
        List[str]: Chunked sections of text.
    """

    all_chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        tokenized = encoder.encode(sentence, add_special_tokens=False)
        sent_len = len(tokenized)
        if current_len + sent_len > max_tokens:
            all_chunks.append(" ".join(current_chunk))
            if overlap > 0:
                overlap_tokens = []
                total_tokens = 0
                for sent in reversed(current_chunk):
                    t = encoder.encode(sent, add_special_tokens=False)
                    total_tokens += len(t)
                    if total_tokens >= overlap:
                        break
                    overlap_tokens.insert(0, sent)
                current_chunk = overlap_tokens
                current_len = sum(len(encoder.encode(s, add_special_tokens=False)) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(sentence)
        current_len += sent_len

    if current_chunk:
        all_chunks.append(" ".join(current_chunk))

    return all_chunks


try:
    with open(EXTRACTED_TEXT, "r", encoding="utf-8") as f:
        full_text = f.read()

    full_text = re.sub(r'\n{2,}', '\n', full_text).strip()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True, use_fast=False)

    section_pattern = r"(?=\n\d+\.\s[A-Z])"
    sections = re.split(section_pattern, full_text)

    processed_sections = []
    for section in sections:
        cleaned = section.strip()
        if not cleaned:
            continue
        if re.match(r"^\d+\.\s[A-Z]", cleaned):
            processed_sections.append(cleaned)
        elif processed_sections:
            processed_sections[-1] += " " + cleaned
        else:
            processed_sections.append(cleaned)

    final_chunks = []
    for section in processed_sections:
        sentences = custom_sent_tokenize(section)
        chunks = split_chunk_by_tokens(sentences, tokenizer)
        final_chunks.extend(chunks)

    with open("Chunks.txt", "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(chunk.strip() + "\n\n***\n\n")

    logger.info(f"Chunking complete. Total chunks: {len(final_chunks)}")

except Exception as e:
    logger.critical(f"Chunking failed: {e}")
