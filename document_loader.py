
import tempfile
from pypdf import PdfReader

def load_uploaded_pdfs(uploaded_file, source_type: str):
    documents = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "metadata": {
                    "source": uploaded_file.name,
                    "page": page_num + 1,
                    "type": source_type
                }
            })

    return documents
