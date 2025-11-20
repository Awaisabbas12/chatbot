import os
import math
import time
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs, unquote_plus

import requests
from langchain_community.document_loaders import PyPDFLoader

# ----------
# CONSTANTS
# ----------

BASE_OUTPUT_DIR = "legal_rag_data"
ARCHIVE_RAW_DIR = os.path.join(BASE_OUTPUT_DIR, "archive_raw")
os.makedirs(ARCHIVE_RAW_DIR, exist_ok=True)

ADV_SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/"
DOWNLOAD_BASE_URL = "https://archive.org/download/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LegalRAGArchiveBot/1.0; +https://example.com/bot-info)"
}


# ----------
# HELPERS
# ----------

def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_", ".", "+", " ")).strip() or "file"


def extract_query_from_search_url(search_url: str) -> Optional[str]:
    """
    From a URL like:
        https://archive.org/search?query=%22service+agreement%22
    extract the value of 'query' and return a decoded string:
        '"service agreement"'
    """
    parsed = urlparse(search_url)
    qs = parse_qs(parsed.query)
    q = qs.get("query", [""])[0]
    if not q:
        return None
    return unquote_plus(q)


def search_archive_docs(query: str, max_items: int = 300, rows: int = 50, sleep: float = 1.0) -> List[Dict[str, Any]]:
    """
    Use Internet Archive Advanced Search API to get a list of docs for the query.
    Returns metadata docs (identifier, title, year, creator, mediatype, etc.)
    """
    all_docs: List[Dict[str, Any]] = []
    page = 1

    while len(all_docs) < max_items:
        params = {
            "q": query,
            "output": "json",
            "rows": rows,
            "page": page,
            "fl[]": [
                "identifier",
                "title",
                "creator",
                "year",
                "mediatype",
                "collection",
            ],
        }

        print(f"[ARCHIVE] AdvancedSearch page {page}")
        resp = requests.get(ADV_SEARCH_URL, params=params, headers=HEADERS, timeout=40)
        resp.raise_for_status()
        data = resp.json().get("response", {})
        docs = data.get("docs", [])
        num_found = data.get("numFound", 0)

        if not docs:
            break

        for d in docs:
            all_docs.append(d)
            if len(all_docs) >= max_items:
                break

        total_pages = math.ceil(num_found / rows)
        if page >= total_pages or len(all_docs) >= max_items:
            break

        page += 1
        time.sleep(sleep)

    print(f"[ARCHIVE] Collected {len(all_docs)} docs (max_items={max_items}).")
    return all_docs


def fetch_metadata(identifier: str) -> Dict[str, Any]:
    resp = requests.get(METADATA_URL + identifier, headers=HEADERS, timeout=40)
    resp.raise_for_status()
    return resp.json()


def choose_text_and_pdf(files: List[Dict[str, Any]]) -> (Optional[Dict[str, Any]], Optional[Dict[str, Any]]):
    """
    Choose best text-like file and a possible PDF file.
    Preference for text:
        DjvuTXT > Text > any .txt
    PDF:
        Text PDF / PDF / any .pdf
    """
    text_file = None
    pdf_file = None

    # text-like
    for f in files:
        fmt = (f.get("format") or "").lower()
        name = (f.get("name") or "").lower()
        if "djvutxt" in fmt or "text" in fmt or name.endswith(".txt"):
            text_file = f
            break

    # pdf-like
    for f in files:
        fmt = (f.get("format") or "").lower()
        name = (f.get("name") or "").lower()
        if "pdf" in fmt or name.endswith(".pdf"):
            pdf_file = f
            break

    return text_file, pdf_file


def download_file(identifier: str, fileinfo: Dict[str, Any]) -> str:
    """
    Download a single file from archive.org/download/{identifier}/{name}
    Returns local file path under ARCHIVE_RAW_DIR.
    """
    name = fileinfo["name"]
    url = f"{DOWNLOAD_BASE_URL}{identifier}/{name}"
    fname = safe_filename(f"{identifier}_{name}")
    path = os.path.join(ARCHIVE_RAW_DIR, fname)

    if os.path.exists(path):
        print(f"[ARCHIVE] Already downloaded {path}")
        return path

    print(f"[ARCHIVE] Downloading {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Use LangChain's PyPDFLoader to read pages and join text.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        print(f"[ARCHIVE] PDF extraction failed for {pdf_path}: {e}")
        return ""


# ----------
# MAIN API
# ----------

def scrape_archive_search_to_items(search_url: str, max_items: int = 300) -> List[Dict[str, Any]]:
    """
    Public function used by scrapper.py.

    For a given archive search URL:
        - resolve Advanced Search query
        - collect up to max_items documents
        - for each document, download OCR text or PDF and extract text
        - return a list of item dicts suitable to be embedded in one JSONL record

    Each returned item has structure roughly:

    {
        "item_url": "https://archive.org/details/...",
        "identifier": "...",
        "title": "...",
        "creator": "...",
        "year": "...",
        "mediatype": "...",
        "collection": [...],
        "file_type": "text" | "pdf" | "none",
        "source_file": "/path/to/file" | null,
        "text": "full extracted text or ''"
    }
    """
    query = extract_query_from_search_url(search_url)
    if not query:
        raise ValueError(f"Could not extract 'query' parameter from URL: {search_url}")

    docs = search_archive_docs(query, max_items=max_items)
    items: List[Dict[str, Any]] = []

    for idx, d in enumerate(docs, start=1):
        identifier = d["identifier"]
        print(f"[ARCHIVE] [{idx}/{len(docs)}] Processing identifier={identifier}")

        try:
            meta = fetch_metadata(identifier)
        except Exception as e:
            print(f"[ARCHIVE] Failed to fetch metadata for {identifier}: {e}")
            items.append({
                "item_url": f"https://archive.org/details/{identifier}",
                "identifier": identifier,
                "title": d.get("title"),
                "creator": d.get("creator"),
                "year": d.get("year"),
                "mediatype": d.get("mediatype"),
                "collection": d.get("collection"),
                "file_type": "none",
                "source_file": None,
                "text": "",
                "error": f"metadata_failed: {e}",
            })
            continue

        files = meta.get("files", []) or []
        text_file, pdf_file = choose_text_and_pdf(files)

        text_content = ""
        file_type = "none"
        src_path = None

        try:
            if text_file:
                src_path = download_file(identifier, text_file)
                with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                file_type = "text"

            elif pdf_file:
                src_path = download_file(identifier, pdf_file)
                text_content = extract_text_from_pdf(src_path)
                file_type = "pdf"

        except Exception as e:
            print(f"[ARCHIVE] Error downloading/extracting for {identifier}: {e}")
            # keep text_content = "" / file_type = "none"

        item_record = {
            "item_url": f"https://archive.org/details/{identifier}",
            "identifier": identifier,
            "title": d.get("title"),
            "creator": d.get("creator"),
            "year": d.get("year"),
            "mediatype": d.get("mediatype"),
            "collection": d.get("collection"),
            "file_type": file_type,
            "source_file": src_path,
            "text": text_content,
        }

        items.append(item_record)

    return items


# if __name__ == "__main__":
#     # Quick manual test (optional)
#     TEST_URL = "https://archive.org/search?query=%22service+agreement%22"
#     data = scrape_archive_search_to_items(TEST_URL, max_items=5)
#     print(json.dumps({"items": data}, ensure_ascii=False)[:2000])
