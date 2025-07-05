import pdfplumber
import re
from config import PDF
import logging
logger = logging.getLogger(__name__)

def extract_text_and_tables(pdf_path):
    """
    Extracts text from a PDF, removing standalone page numbers.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Cleaned full text extracted from the PDF.
    """

    full_text = ""
   
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
                    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
                    full_text += f"\n\n{text}"
        logger.info("PDF extraction completed.")
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise

    return full_text

if __name__ == "__main__":
    try:
        pdf = PDF
        text = extract_text_and_tables(pdf)
        with open("Extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved extracted text to 'Extracted_text.txt'.")
    except Exception as e:
        logger.critical(f"Failed to extract or save PDF content: {e}")