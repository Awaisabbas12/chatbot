import os
import time
import json
import mimetypes
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from config import BRAINS


BASE_OUTPUT_DIR = "legal_rag_data"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LegalRAGBot/1.0; +https://example.com/bot-info)"
}


def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_", ".", "+", "=")).strip() or "file"


def get_extension_from_content_type(ct: str | None) -> str:
    if not ct:
        return ".bin"
    ct = ct.split(";")[0].strip().lower()
    if "html" in ct:
        return ".html"
    if "pdf" in ct:
        return ".pdf"
    if "json" in ct:
        return ".json"
    return mimetypes.guess_extension(ct) or ".bin"


def fetch(url: str, timeout: int = 40) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def get_domain(url: str) -> str:
    return urlparse(url).netloc


def get_binary_output_path(brain: str, url: str, ext: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace(":", "_") or "unknown_domain"
    path_part = parsed.path or "index"
    if path_part.endswith("/"):
        path_part += "index"
    path_part = path_part.replace("/", "_") or "file"
    filename = safe_filename(f"{domain}_{path_part}{ext}")
    out_dir = os.path.join(BASE_OUTPUT_DIR, brain, "raw_files")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def append_jsonl_record(brain: str, record: dict):
    jsonl_path = os.path.join(BASE_OUTPUT_DIR, f"{brain}.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Archive.org SPECIAL HANDLERS
# =========================

def is_archive_search(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "archive.org" and parsed.path == "/search"


def scrape_archive_item(brain: str, item_url: str, fetched_at: str | None = None):
    """
    Given a single item page like https://archive.org/details/XYZ,
    download its HTML text and any direct files (PDF/TXT/DOC/DOCX) linked from it.
    """
    print(f"    [ITEM] {item_url}")
    resp = fetch(item_url)
    if resp is None:
        append_jsonl_record(brain, {
            "brain": brain,
            "url": item_url,
            "domain": get_domain(item_url),
            "status": "error",
            "error": "item_request_failed",
            "fetched_at": fetched_at or (datetime.utcnow().isoformat() + "Z"),
        })
        return

    ct = (resp.headers.get("Content-Type") or "").lower()
    domain = get_domain(item_url)
    ts = fetched_at or (datetime.utcnow().isoformat() + "Z")

    # 1) Save the item page HTML as text (for context / metadata)
    if "text/html" in ct or "<html" in resp.text[:500].lower():
        text = html_to_text(resp.text)
        append_jsonl_record(brain, {
            "brain": brain,
            "url": item_url,
            "domain": domain,
            "status": "ok",
            "type": "html",
            "content_type_header": ct,
            "fetched_at": ts,
            "text": text,
            "source": "archive_item_page",
        })
        print(f"      [OK] Saved item HTML text")

        # 2) Parse download links from the item page
        soup = BeautifulSoup(resp.text, "html.parser")
        file_links: set[str] = set()

        # Archive.org item pages usually have links to actual files.
        # We'll be generic: look for hrefs pointing to common doc extensions.
        for a in soup.find_all("a", href=True):
            href = a["href"]
            lower = href.lower()
            if lower.endswith((".pdf", ".txt", ".doc", ".docx")):
                file_links.add(urljoin(item_url, href))

        for file_url in sorted(file_links):
            print(f"      [FILE] {file_url}")
            file_resp = fetch(file_url)
            if file_resp is None:
                append_jsonl_record(brain, {
                    "brain": brain,
                    "url": file_url,
                    "domain": get_domain(file_url),
                    "status": "error",
                    "error": "file_request_failed",
                    "fetched_at": ts,
                    "parent_item": item_url,
                })
                continue

            file_ct = (file_resp.headers.get("Content-Type") or "").lower()
            ext = get_extension_from_content_type(file_ct)
            out_path = get_binary_output_path(brain, file_url, ext)
            try:
                with open(out_path, "wb") as f:
                    f.write(file_resp.content)
                append_jsonl_record(brain, {
                    "brain": brain,
                    "url": file_url,
                    "domain": get_domain(file_url),
                    "status": "ok",
                    "type": "binary",
                    "content_type_header": file_ct,
                    "fetched_at": ts,
                    "file_path": out_path,
                    "parent_item": item_url,
                    "source": "archive_item_file",
                })
                print(f"        [OK] Saved file → {out_path}")
            except Exception as e:
                print(f"        [ERROR] Saving file failed: {e}")
                append_jsonl_record(brain, {
                    "brain": brain,
                    "url": file_url,
                    "domain": get_domain(file_url),
                    "status": "error",
                    "error": f"save_file_failed: {e}",
                    "content_type_header": file_ct,
                    "fetched_at": ts,
                    "parent_item": item_url,
                })
    else:
        # Rare: item page is non-HTML, just save raw
        ext = get_extension_from_content_type(ct)
        out_path = get_binary_output_path(brain, item_url, ext)
        try:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            append_jsonl_record(brain, {
                "brain": brain,
                "url": item_url,
                "domain": domain,
                "status": "ok",
                "type": "binary",
                "content_type_header": ct,
                "fetched_at": ts,
                "file_path": out_path,
                "source": "archive_item_raw",
            })
            print(f"      [OK] Saved raw item content → {out_path}")
        except Exception as e:
            print(f"      [ERROR] Saving raw item failed: {e}")
            append_jsonl_record(brain, {
                "brain": brain,
                "url": item_url,
                "domain": domain,
                "status": "error",
                "error": f"save_item_failed: {e}",
                "content_type_header": ct,
                "fetched_at": ts,
                "source": "archive_item_raw",
            })


def scrape_archive_search(brain: str, search_url: str, max_items: int = 50, delay: float = 1.5):
    """
    Crawl Archive.org search results:
    - Go through result pages (&page=2,3,...)
    - For each result item (/details/...), scrape the item page + files.
    """
    print(f"[ARCHIVE] Starting search crawl for {search_url}")
    fetched_at = datetime.utcnow().isoformat() + "Z"

    parsed = urlparse(search_url)
    base_query = parsed.query or ""
    base_path = parsed.path
    base_root = f"{parsed.scheme}://{parsed.netloc}{base_path}"

    page = 1
    scraped_items = 0
    seen_items: set[str] = set()

    while scraped_items < max_items:
        if base_query:
            page_url = f"{base_root}?{base_query}&page={page}"
        else:
            page_url = f"{base_root}?page={page}"

        print(f"[ARCHIVE] Page {page}: {page_url}")
        resp = fetch(page_url)
        if resp is None:
            print("[ARCHIVE] Stopping: failed to fetch page.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for any link to /details/...
        item_links: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/details/"):
                full = urljoin("https://archive.org", href)
                item_links.add(full)

        if not item_links:
            print("[ARCHIVE] No more item links found, stopping.")
            break

        for item_url in sorted(item_links):
            if scraped_items >= max_items:
                break
            if item_url in seen_items:
                continue
            seen_items.add(item_url)
            scraped_items += 1

            scrape_archive_item(brain, item_url, fetched_at=fetched_at)
            time.sleep(delay)

        page += 1
        time.sleep(delay)

    print(f"[ARCHIVE] Done. Scraped {scraped_items} items for brain={brain}")


# =========================
# CORE SCRAPING LOGIC
# =========================

def scrape_url_for_brain(brain: str, url: str):
    # Special handling for archive.org search pages:
    if is_archive_search(url):
        # You can tune max_items as needed
        scrape_archive_search(brain, url, max_items=50, delay=1.5)
        return

    print(f"\n[INFO] ({brain}) Fetching: {url}")
    resp = fetch(url)
    fetched_at = datetime.utcnow().isoformat() + "Z"

    if resp is None:
        record = {
            "brain": brain,
            "url": url,
            "domain": get_domain(url),
            "status": "error",
            "error": "request_failed",
            "fetched_at": fetched_at,
        }
        append_jsonl_record(brain, record)
        return

    ct = (resp.headers.get("Content-Type") or "").lower()
    domain = get_domain(url)

    # HTML – store clean text
    if "text/html" in ct or "<html" in resp.text[:500].lower():
        text = html_to_text(resp.text)
        record = {
            "brain": brain,
            "url": url,
            "domain": domain,
            "status": "ok",
            "type": "html",
            "content_type_header": ct,
            "fetched_at": fetched_at,
            "text": text,
        }
        append_jsonl_record(brain, record)
        print(f"[OK] Saved HTML text record for {url}")

    else:
        # Binary (PDF, JSON, etc.)
        ext = get_extension_from_content_type(ct)
        out_path = get_binary_output_path(brain, url, ext)
        try:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            record = {
                "brain": brain,
                "url": url,
                "domain": domain,
                "status": "ok",
                "type": "binary",
                "content_type_header": ct,
                "fetched_at": fetched_at,
                "file_path": out_path,
            }
            append_jsonl_record(brain, record)
            print(f"[OK] Saved binary file → {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save binary for {url}: {e}")
            record = {
                "brain": brain,
                "url": url,
                "domain": domain,
                "status": "error",
                "error": f"save_binary_failed: {e}",
                "content_type_header": ct,
                "fetched_at": fetched_at,
            }
            append_jsonl_record(brain, record)


def scrape_all_brains(delay: float = 2.0):
    for brain, urls in BRAINS.items():
        print(f"\n========== Scraping brain: {brain} ==========")
        for url in urls:
            scrape_url_for_brain(brain, url)
            time.sleep(delay)


if __name__ == "__main__":
    scrape_all_brains(delay=2.0)
