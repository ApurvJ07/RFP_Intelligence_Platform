import os
import re
from bs4 import BeautifulSoup
import pdfplumber

def detect_doc_type(filename: str):
    lower = filename.lower()
    if lower.endswith((".html", ".htm")):
        return "bid_page", None
    elif "addendum" in lower:
        match = re.search(r'addendum\s*(\d+)', lower)
        add_num = int(match.group(1)) if match else 1
        return "addendum", add_num
    elif "spec" in lower:
        return "specs", None
    elif "affidavit" in lower:
        return "affidavit", None
    else:
        return "rfp", None

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_html(filepath: str, bid_id: str, filename: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = clean_text(soup.get_text(separator="\n"))
    doc_type, add_num = detect_doc_type(filename)
    return [{
        "bid_id": bid_id,
        "file_name": filename,
        "doc_type": doc_type,
        "addendum_number": add_num,
        "page_number": 1,
        "text": text
    }]

def format_table_as_markdown(table: list[list]) -> str:
    if not table:
        return ""
    md_rows = []
    for row in table:
        cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        if any(cleaned_row):
            md_rows.append("| " + " | ".join(cleaned_row) + " |")
    return "\n".join(md_rows)

def parse_pdf(filepath: str, bid_id: str, filename: str) -> list[dict]:
    pages_data = []
    doc_type, add_num = detect_doc_type(filename)
    try:
        with pdfplumber.open(filepath) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                tables = page.extract_tables() or []
                table_texts = [format_table_as_markdown(t) for t in tables if t]
                combined = clean_text(raw_text)
                if table_texts:
                    combined += "\n\n[EXTRACTED TABLES]:\n" + "\n\n".join(table_texts)
                if combined:
                    pages_data.append({
                        "bid_id": bid_id,
                        "file_name": filename,
                        "doc_type": doc_type,
                        "addendum_number": add_num,
                        "page_number": idx,
                        "text": combined
                    })
    except Exception as e:
        print(f"[ERROR] Failed to parse {filename}: {e}")
    return pages_data

def ingest_bid_folder(folder_path: str) -> list[dict]:
    bid_id = os.path.basename(os.path.normpath(folder_path))
    all_pages = []
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return all_pages
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith((".html", ".htm")):
            all_pages.extend(parse_html(fpath, bid_id, fname))
        elif fname.lower().endswith(".pdf"):
            all_pages.extend(parse_pdf(fpath, bid_id, fname))
    return all_pages

if __name__ == "__main__":
    sample_pages = ingest_bid_folder("data/Bid1")
    print(f"Successfully extracted {len(sample_pages)} pages from Bid1!")
    if sample_pages:
        print("Sample Metadata:", {k: v for k, v in sample_pages[0].items() if k != "text"})