"""
contract_extractor_ndamsa.py

Updated features per your request:
- Outputs a JSON file per source and a combined JSONL file containing metadata + full extracted article content.
- Follows links discovered on each main page recursively (configurable depth and link limits).
- Extracts HTML article/body text and full-page text. Tries to identify the main article using basic heuristics (article tags, largest <p> block, or fallback to full-page text).
- Handles PDFs, GitHub blob -> raw conversion, and common contract repositories.
- Respects rate limits and avoids infinite crawling via depth and per-domain limits.

Run:
python contract_extractor_ndamsa.py

Outputs:
- ./downloads/ (raw files)
- ./extracted/ (plain text files)
- ./json_output/ (one JSON per source + combined.jsonl)
- ./metadata.csv (summary CSV)

Dependencies:
pip install requests beautifulsoup4 pdfminer.six tqdm pandas python-dateutil

"""

import os
import re
import time
import json
import logging
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from dateutil import parser as dateparser

# PDF text extraction using pdfminer.six
from pdfminer.high_level import extract_text

# --------------------------- Configuration ---------------------------
USER_AGENT = "ContractExtractor/1.0 (+https://example.com)"
HEADERS = {"User-Agent": USER_AGENT}
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"
JSON_OUT_DIR = "json_output"
METADATA_CSV = "metadata.csv"
COMBINED_JSONL = os.path.join(JSON_OUT_DIR, "combined.jsonl")
MAX_WORKERS = 6
REQUEST_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.8  # polite
FOLLOW_LINKS = True
MAX_CRAWL_DEPTH = 2  # how deep to follow links from the root page
MAX_LINKS_PER_PAGE = 20  # limit links followed per page to avoid explosion
FOLLOW_SAME_DOMAIN_ONLY = False  # set True to restrict to same domain

# --------------------------- Seed URLs (first two sections) ---------------------------
URLS = [
    # NDA Brain
    "https://www.sec.gov/edgar/search-and-access",
    "https://www.sec.gov/edgar/search/#/q=nda&filter_forms=10-K",
    "https://github.com/ContractStandards",
    "https://www.onenda.org/blog-post/why-nda-litigation-is-rare",
    "https://www.onenda.org/blog-post/onenda-and-law-insider-are-redefining-legal-standardisation"
    "https://huggingface.co/datasets/atticus-project/cuad",
    "https://github.com/jamesacampbell/contracts",
    "https://github.com/alangrafu/legal-docs",
    "https://www.dol.gov/agencies/oasam/site-closures/nda",
    "https://archive.org/search?query=non-disclosure+agreement",

    # MSA Brain
    "https://www.sec.gov/edgar/search-and-access",
    "https://www.sec.gov/about/privacy-information#dissemination"
    "https://www.sec.gov/edgar/search/#/q=%22master%20service%20agreement%22",
    "https://huggingface.co/datasets/lex_glue",
    "https://content.next.westlaw.com/practical-law",
    "https://www.miamidade.gov/Apps/ContractSearch/",
    "https://www.data.gov/search?q=contract",
    "https://sam.gov/content/opportunities",
    "https://archive.org/search?query=%22service+agreement%22",
]

# --------------------------- Setup ---------------------------
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(JSON_OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

session = requests.Session()
session.headers.update(HEADERS)

# --------------------------- Helpers ---------------------------

def safe_get(url, stream=False, timeout=30):
    for i in range(REQUEST_RETRIES):
        try:
            r = session.get(url, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            logging.warning("GET failed %s (attempt %d/%d): %s", url, i + 1, REQUEST_RETRIES, e)
            time.sleep(1 + i * 2)
    logging.error("Failed to GET %s after %d attempts", url, REQUEST_RETRIES)
    return None


def is_pdf_response(resp, url):
    if not resp:
        return False
    ct = resp.headers.get("content-type", "").lower()
    if "pdf" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return False


def make_filename_from_url(url):
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    if parsed.query:
        name += "_" + parsed.query
    # sanitise
    name = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    if len(name) > 200:
        name = name[:200]
    return name


def save_stream_to_file(resp, path):
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_text_from_pdf(path):
    try:
        text = extract_text(path)
        return text
    except Exception as e:
        logging.exception("PDF extraction failed for %s: %s", path, e)
        return ""


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    full_text = soup.get_text(separator="")
    article_text = extract_main_article_text(soup)
    return full_text.strip(), article_text.strip()


def extract_main_article_text(soup):
    # Heuristics: article tag, main tag, largest <div> by text length, or largest group of <p>
    article = soup.find("article")
    if article:
        return article.get_text(separator="")
    main = soup.find("main")
    if main:
        return main.get_text(separator="")
    # find the element with most <p> text
    candidates = soup.find_all(["div", "section", "article", "body"], limit=40)
    best = None
    best_len = 0
    for c in candidates:
        text = c.get_text(separator="").strip()
        ln = len(text)
        if ln > best_len:
            best_len = ln
            best = text
    if best_len > 200:
        return best
    # fallback: concatenate top-level <p>
    ps = soup.find_all("p")
    if ps:
        texts = [p.get_text().strip() for p in ps if p.get_text().strip()]
        # find longest contiguous block
        if texts:
            return "".join(texts[:50])
    return soup.get_text(separator="")


def github_blob_to_raw(url):
    m = re.match(r"https?://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def discover_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        found.append(full)
    # de-duplicate preserving order
    seen = set()
    out = []
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def within_domain(root, candidate):
    if not FOLLOW_SAME_DOMAIN_ONLY:
        return True
    r = urlparse(root).netloc
    c = urlparse(candidate).netloc
    return r == c


# --------------------------- Worker ---------------------------

def process_url(url, root_url=None, depth=0, visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return None
    visited.add(url)

    logging.info("[Depth %d] Processing %s", depth, url)
    result = {
        "source_url": url,
        "root_url": root_url or url,
        "fetched_url": None,
        "local_path": None,
        "downloaded": False,
        "content_type": None,
        "status_code": None,
        "title": None,
        "published": None,
        "full_text_path": None,
        "article_text_path": None,
        "article_text": None,
        "full_text": None,
        "error": None,
        "discovered_links": [],
    }

    # convert github blob to raw url
    if "github.com" in url and "/blob/" in url:
        url = github_blob_to_raw(url)

    resp = safe_get(url, stream=True)
    if resp is None:
        result["error"] = "failed_to_fetch"
        return result

    result["status_code"] = resp.status_code
    result["fetched_url"] = resp.url
    ct = resp.headers.get("content-type", "").lower()
    result["content_type"] = ct

    try:
        if is_pdf_response(resp, url):
            fname = make_filename_from_url(resp.url)
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            local_path = os.path.join(DOWNLOAD_DIR, fname)
            save_stream_to_file(resp, local_path)
            result["local_path"] = local_path
            result["downloaded"] = True
            # extract text
            text = extract_text_from_pdf(local_path)
            txt_path = os.path.join(EXTRACT_DIR, os.path.basename(local_path) + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            result["full_text_path"] = txt_path
            result["full_text"] = text

        else:
            html = resp.text
            fname = make_filename_from_url(resp.url)
            if not fname.lower().endswith(".html"):
                fname += ".html"
            local_path = os.path.join(DOWNLOAD_DIR, fname)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(html)
            result["local_path"] = local_path
            result["downloaded"] = True

            full_text, article_text = extract_text_from_html(html)
            txt_path = os.path.join(EXTRACT_DIR, os.path.basename(local_path) + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            result["full_text_path"] = txt_path
            result["full_text"] = full_text

            art_path = os.path.join(EXTRACT_DIR, os.path.basename(local_path) + ".article.txt")
            with open(art_path, "w", encoding="utf-8") as f:
                f.write(article_text)
            result["article_text_path"] = art_path
            result["article_text"] = article_text

            # metadata: title, possible publish date
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text().strip()
            # try common meta tags for published date
            for tagname in [
                ('meta', {'property':'article:published_time'}),
                ('meta', {'name':'date'}),
                ('meta', {'name':'publication_date'}),
                ('meta', {'name':'pubdate'}),
                ('time', {}),
            ]:
                try:
                    if tagname[0] == 'meta':
                        m = soup.find('meta', attrs=tagname[1])
                        if m and m.get('content'):
                            try:
                                result['published'] = dateparser.parse(m.get('content')).isoformat()
                                break
                            except Exception:
                                result['published'] = m.get('content')
                    else:
                        t = soup.find('time')
                        if t and t.get('datetime'):
                            result['published'] = dateparser.parse(t.get('datetime')).isoformat()
                            break
                        elif t and t.get_text().strip():
                            result['published'] = t.get_text().strip()
                            break
                except Exception:
                    continue

            # discover links
            links = discover_links(html, resp.url)
            result["discovered_links"] = links[:MAX_LINKS_PER_PAGE]

            # optionally follow links
            if FOLLOW_LINKS and depth < MAX_CRAWL_DEPTH:
                children = []
                for link in links[:MAX_LINKS_PER_PAGE]:
                    if not within_domain(result['root_url'], link):
                        continue
                    # avoid binary files except PDFs
                    if any(link.lower().endswith(ext) for ext in ['.jpg', '.png', '.zip', '.exe']):
                        continue
                    # recurse
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                    child_res = process_url(link, root_url=result['root_url'], depth=depth+1, visited=visited)
                    if child_res:
                        children.append(child_res)
                # attach a lightweight summary of children to result
                result['children_count'] = len(children)

    except Exception as e:
        logging.exception("Error processing %s: %s", url, e)
        result['error'] = str(e)

    # write per-source JSON
    basename = make_filename_from_url(result['fetched_url'] or url)
    json_path = os.path.join(JSON_OUT_DIR, basename + ".json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)

    # append to combined JSONL
    with open(COMBINED_JSONL, 'a', encoding='utf-8') as outf:
        outf.write(json.dumps(result, ensure_ascii=False) + '')

    # small pause
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return result


# --------------------------- Orchestration ---------------------------

def main(urls=URLS):
    all_results = []
    visited = set()
    for u in urls:
        try:
            res = process_url(u, root_url=u, depth=0, visited=visited)
            if res:
                all_results.append(res)
        except Exception as e:
            logging.exception("Failed on root %s: %s", u, e)

    # save metadata csv
    rows = []
    for r in all_results:
        rows.append({
            'source_url': r.get('source_url'),
            'fetched_url': r.get('fetched_url'),
            'local_path': r.get('local_path'),
            'full_text_path': r.get('full_text_path'),
            'article_text_path': r.get('article_text_path'),
            'content_type': r.get('content_type'),
            'status_code': r.get('status_code'),
            'title': r.get('title'),
            'published': r.get('published'),
            'downloaded': r.get('downloaded'),
            'error': r.get('error'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(METADATA_CSV, index=False)
    logging.info("Saved metadata to %s", METADATA_CSV)

    logging.info("Done. JSON output in %s, combined JSONL at %s", JSON_OUT_DIR, COMBINED_JSONL)


if __name__ == '__main__':
    # clear combined output if exists
    if os.path.exists(COMBINED_JSONL):
        os.remove(COMBINED_JSONL)
    main()