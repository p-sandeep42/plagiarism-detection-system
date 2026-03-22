import os
import io
import fitz  # PyMuPDF
import docx

def parse_pdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def parse_file(file_filename: str, file_bytes: bytes) -> str:
    ext = file_filename.split(".")[-1].lower()
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext == "docx":
        return parse_docx(file_bytes)
    elif ext in ["txt", "py", "js", "md", "csv", "json"]:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")
    else:
        # Default fallback
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return str(file_bytes)
