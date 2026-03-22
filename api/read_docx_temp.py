from docx import Document
import sys

def read_docx(path):
    doc = Document(path)
    for para in doc.paragraphs:
        print(para.text)

if __name__ == "__main__":
    read_docx(sys.argv[1])
