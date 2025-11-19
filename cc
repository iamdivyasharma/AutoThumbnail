
from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict

import fitz  # PyMuPDF

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Optional: DOCX loader
try:
    from langchain_community.document_loaders import Docx2txtLoader
    _HAS_DOCX_LOADER = True
except Exception:
    _HAS_DOCX_LOADER = False

# ===================== LOCAL PATH CONFIG =====================

# Local root where SageMaker sees ce/legal and ce/finance
LOCAL_DOC_ROOT = Path("ce").resolve()

CATEGORY_ROOTS: Dict[str, Path] = {
    "legal":   LOCAL_DOC_ROOT / "legal",
    "finance": LOCAL_DOC_ROOT / "finance",
}

# Where to write FAISS indices on SageMaker
FAISS_INDEX_DIR = Path("/opt/ml/faiss_indices").resolve()

# ===================== S3 DOC CONFIG (for Textract) =====================

# BUCKET that holds your source PDFs for Textract
S3_DOC_BUCKET = "YOUR_SOURCE_DOC_BUCKET"       # TODO: change this

# Prefix inside that bucket that corresponds to LOCAL_DOC_ROOT
# Example: if local path is ce/legal/ClientA/file.pdf and S3 key is
#          user/d/ce/legal/ClientA/file.pdf then S3_DOC_PREFIX = "user/d/ce"
S3_DOC_PREFIX = "user/d/ce"                    # TODO: adjust to your layout

# ===================== AWS / BEDROCK / TEXTRACT CONFIG =====================

REGION      = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

import boto3
from boto3 import Session

if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    SESSION = Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
    )
else:
    # Use default chain (e.g. SageMaker IAM role)
    SESSION = Session(region_name=REGION)

TEXTRACT_CLIENT = SESSION.client("textract")

# Poll interval for Textract async jobs (seconds)
TEXTRACT_POLL_INTERVAL = 5

# ===================== CHUNKING CONFIG =====================

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150

# ===================== HELPERS =====================

def clean_text(t: str) -> str:
    """Light cleanup; keep punctuation and newlines."""
    return re.sub(r"[ \t]+\n", "\n", t or "").strip()

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Tuple[int, int, str]]:
    """
    Split `text` into chunks with overlap.
    Returns list of (start_idx, end_idx, chunk_text).
    """
    text = text or ""
    n = len(text)
    if n <= chunk_size:
        return [(0, n, text)]
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@dataclass
class DocInfo:
    policy_id: str
    source_file: str
    pages: int
    modified_time: int

# ===================== LOCAL → S3 KEY MAPPING =====================

def local_path_to_s3_key(path: Path) -> Optional[str]:
    """
    Map a local path under LOCAL_DOC_ROOT to the S3 key used for Textract.
    Example:
      LOCAL_DOC_ROOT = /opt/ml/input/ce
      path           = /opt/ml/input/ce/legal/ClientA/file1.pdf
      relative       = legal/ClientA/file1.pdf
      S3_DOC_PREFIX  = "user/d/ce"
      => S3 key      = "user/d/ce/legal/ClientA/file1.pdf"
    """
    try:
        rel = path.resolve().relative_to(LOCAL_DOC_ROOT.resolve())
    except ValueError:
        print(f"[WARN] Path {path} is not under LOCAL_DOC_ROOT {LOCAL_DOC_ROOT}; cannot map to S3 key.")
        return None

    rel_key = rel.as_posix()
    if S3_DOC_PREFIX:
        return f"{S3_DOC_PREFIX.rstrip('/')}/{rel_key}"
    else:
        return rel_key

# ===================== CLIENT SCAN (LOCAL FOLDERS) =====================

def scan_category_client_files() -> Dict[str, Dict[str, List[Path]]]:
    """
    Walk ce/legal and ce/finance and build:
        {
          "legal":   { "clienta": [list of Paths], ... },
          "finance": { "clientx": [list of Paths], ... }
        }
    Each immediate subdirectory under each category root is treated
    as a client folder (client_id = folder name lowercased).
    """
    out: Dict[str, Dict[str, List[Path]]] = {}

    for category, root in CATEGORY_ROOTS.items():
        if not root.exists():
            print(f"[WARN] Category root does not exist: {root}")
            continue

        cat_map: Dict[str, List[Path]] = {}

        for client_dir in root.iterdir():
            if not client_dir.is_dir():
                continue

            client_id = client_dir.name.strip().lower()

            pdfs  = list(client_dir.rglob("*.pdf"))
            docs  = list(client_dir.rglob("*.doc")) + list(client_dir.rglob("*.docx"))
            files = sorted(pdfs + docs, key=lambda p: p.name.lower())

            if not files:
                continue

            cat_map.setdefault(client_id, []).extend(files)

        if cat_map:
            out[category] = cat_map

    return out

# ===================== EMBEDDINGS =====================

def embedder():
    return BedrockEmbeddings(
        model_id=EMBED_MODEL,
        client=SESSION.client("bedrock-runtime")
    )

# ===================== SCANNED PDF DETECTION =====================

def is_scanned_pdf(pdf_path: Path, sample_pages: int = 3, char_threshold: int = 50) -> bool:
    """
    Heuristic:
      - Open PDF with PyMuPDF.
      - Sample up to `sample_pages` pages.
      - If total extracted text length < char_threshold * sampled_pages,
        treat as scanned.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = doc.page_count
            if n_pages == 0:
                return True
            to_sample = min(n_pages, sample_pages)
            total_chars = 0
            for i in range(to_sample):
                page = doc.load_page(i)
                txt = page.get_text("text") or ""
                total_chars += len(txt.strip())
            if total_chars < char_threshold * to_sample:
                return True
            return False
    except Exception as e:
        print(f"[WARN] Failed to inspect {pdf_path} for scanned detection: {e}")
        # If we can't read text at all, treat as scanned to push through Textract
        return True

# ===================== TEXTRACT VIA S3 (ASYNC JOB) =====================

def textract_start_job(bucket: str, key: str) -> Optional[str]:
    """
    Start a Textract DocumentTextDetection job for a PDF in S3.
    Returns JobId or None on failure.
    """
    try:
        resp = TEXTRACT_CLIENT.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        job_id = resp["JobId"]
        print(f"[TEXTRACT] Started job {job_id} for s3://{bucket}/{key}")
        return job_id
    except Exception as e:
        print(f"[WARN] Failed to start Textract job for s3://{bucket}/{key}: {e}")
        return None

def textract_wait_for_job(job_id: str) -> bool:
    """
    Poll Textract until job is SUCCEEDED or FAILED.
    Returns True if SUCCEEDED, else False.
    """
    while True:
        try:
            resp = TEXTRACT_CLIENT.get_document_text_detection(
                JobId=job_id,
                MaxResults=1
            )
        except Exception as e:
            print(f"[WARN] get_document_text_detection failed for JobId {job_id}: {e}")
            return False

        status = resp.get("JobStatus")
        if status == "SUCCEEDED":
            print(f"[TEXTRACT] Job {job_id} SUCCEEDED")
            return True
        if status in ("FAILED", "PARTIAL_SUCCESS"):
            print(f"[WARN] Textract job {job_id} ended with status {status}")
            return False

        print(f"[TEXTRACT] Job {job_id} status {status}, waiting...")
        time.sleep(TEXTRACT_POLL_INTERVAL)

def textract_collect_blocks(job_id: str) -> List[dict]:
    """
    After the job has SUCCEEDED, collect all Blocks by paginating.
    """
    blocks: List[dict] = []
    next_token: Optional[str] = None

    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token

        resp = TEXTRACT_CLIENT.get_document_text_detection(**kwargs)
        blocks.extend(resp.get("Blocks", []))
        next_token = resp.get("NextToken")
        if not next_token:
            break

    return blocks

def textract_pdf_to_page_texts_s3(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Call Textract on the S3 version of this PDF and return per-page text.
    """
    s3_key = local_path_to_s3_key(pdf_path)
    if not s3_key:
        return []

    job_id = textract_start_job(S3_DOC_BUCKET, s3_key)
    if not job_id:
        return []

    if not textract_wait_for_job(job_id):
        return []

    blocks = textract_collect_blocks(job_id)
    if not blocks:
        return []

    page_lines: Dict[int, List[str]] = defaultdict(list)
    for b in blocks:
        if b.get("BlockType") == "LINE":
            text = (b.get("Text") or "").strip()
            if not text:
                continue
            page_num = int(b.get("Page") or 1)
            page_lines[page_num].append(text)

    page_texts: List[Tuple[int, str]] = []
    for page_num in sorted(page_lines.keys()):
        combined = "\n".join(page_lines[page_num])
        combined = clean_text(combined)
        if combined.strip():
            page_texts.append((page_num, combined))

    return page_texts

# ===================== PDF CHUNKING =====================

def make_chunks_for_pdf(pdf_path: Path, category: str, client_id: str) -> Iterable[Document]:
    """
    Decide between:
      - PyMuPDF for text-based PDFs.
      - Textract (via S3) for scanned PDFs.
    No Tesseract is used.
    """
    policy_id = pdf_path.name
    scanned = is_scanned_pdf(pdf_path)

    # ===== SCANNED PDF → TEXTRACT VIA S3 WITH PAGE RANGES =====
    if scanned:
        page_texts = textract_pdf_to_page_texts_s3(pdf_path)
        if not page_texts:
            print(f"[WARN] No Textract text for scanned PDF {pdf_path}; skipping.")
            return

        concatenated = ""
        page_ranges: List[Tuple[int, int, int]] = []  # (start_char, end_char, page_num)
        cursor = 0
        for page_num, txt in page_texts:
            start = cursor
            concatenated += txt + "\n"
            cursor = len(concatenated)
            page_ranges.append((start, cursor, page_num))

        for c_start, c_end, chunk in chunk_text(concatenated):
            pages = sorted({
                page_num
                for p_start, p_end, page_num in page_ranges
                if not (p_end <= c_start or p_start >= c_end)
            })
            if not pages:
                continue

            meta = {
                "category": category,
                "client_id": client_id,
                "policy_id": policy_id,
                "page_start": pages[0],
                "page_end": pages[-1],
                "source_file": str(pdf_path.resolve()),
                "extracted_by": "textract_s3",
            }
            yield Document(page_content=chunk, metadata=meta)

        return

    # ===== NON-SCANNED PDF → SIMPLE PYMUPDF PAGE-BY-PAGE =====
    try:
        with fitz.open(str(pdf_path)) as doc:
            for pnum in range(doc.page_count):
                page = doc.load_page(pnum)
                text = clean_text(page.get_text("text") or "")
                if not text.strip():
                    continue
                for _, _, chunk in chunk_text(text):
                    meta = {
                        "category": category,
                        "client_id": client_id,
                        "policy_id": policy_id,
                        "page": pnum + 1,
                        "source_file": str(pdf_path.resolve()),
                        "extracted_by": "pymupdf",
                    }
                    yield Document(page_content=chunk, metadata=meta)
    except Exception as e:
        print(f"[WARN] Failed to process PDF {pdf_path} via PyMuPDF: {e}")

# ===================== DOC/DOCX CHUNKING =====================

def make_chunks_for_docx(doc_path: Path, category: str, client_id: str) -> Iterable[Document]:
    """
    Extract text from DOC/DOCX using Docx2txtLoader, then chunk.
    Since DOCX has no stable 'page' concept, we assign a logical page
    number based on chunk order: page 1, 2, 3, ...
    """
    if not _HAS_DOCX_LOADER:
        print(f"[WARN] DOCX loader not available; skipping {doc_path}")
        return

    policy_id = doc_path.name
    try:
        loader = Docx2txtLoader(str(doc_path))
        docs = loader.load()
    except Exception as e:
        print(f"[WARN] Failed to load DOC/DOCX {doc_path}: {e}")
        return

    for d in docs:
        text = clean_text(d.page_content or "")
        if not text.strip():
            continue

        chunks = chunk_text(text)  # [(start, end, chunk_text), ...]
        for logical_page, (_, _, chunk) in enumerate(chunks, start=1):
            meta = {
                "category": category,
                "client_id": client_id,
                "policy_id": policy_id,
                "page": logical_page,  # logical page index (chunk number)
                "source_file": str(doc_path.resolve()),
                "extracted_by": "docx2txt",
            }
            yield Document(page_content=chunk, metadata=meta)

# ===================== FILE DISPATCH =====================

def make_chunks_for_file(path: Path, category: str, client_id: str) -> Iterable[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        yield from make_chunks_for_pdf(path, category, client_id)
    elif suf in (".doc", ".docx"):
        yield from make_chunks_for_docx(path, category, client_id)
    else:
        print(f"[INFO] Skipping unsupported file type: {path}")

# ===================== MANIFEST / INDEX BUILD =====================

def build_manifest_entry(path: Path) -> DocInfo:
    pages = 0
    if path.suffix.lower() == ".pdf":
        try:
            with fitz.open(str(path)) as doc:
                pages = doc.page_count
        except Exception:
            pages = 0
    stat = path.stat()
    return DocInfo(
        policy_id=path.name,
        source_file=str(path.resolve()),
        pages=pages,
        modified_time=int(stat.st_mtime),
    )

def index_client(category: str, client_id: str, files: List[Path]) -> None:
    client_dir = FAISS_INDEX_DIR / category / client_id
    client_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    manifest_rows: List[DocInfo] = []

    for path in files:
        print(f"[{category}/{client_id}] Processing {path}")
        manifest_rows.append(build_manifest_entry(path))
        for doc in make_chunks_for_file(path, category, client_id):
            docs.append(doc)

    if not docs:
        print(f"[{category}/{client_id}] No extractable text; skipping index.")
        return

    print(f"[{category}/{client_id}] Building FAISS with {len(docs)} chunks…")
    vs = FAISS.from_documents(docs, embedder())
    vs.save_local(str(client_dir))

    manifest = {
        "category": category,
        "client_id": client_id,
        "docs": [
            {
                "policy_id": m.policy_id,
                "source_file": m.source_file,
                "pages": m.pages,
                "modified_time": m.modified_time,
            }
            for m in manifest_rows
        ],
        "generated_time": int(time.time()),
    }
    (client_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[{category}/{client_id}] Done. Index at {client_dir}")

# ===================== ENTRYPOINT =====================

def main():
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    by_cat_client = scan_category_client_files()
    if not by_cat_client:
        print("No files (PDF/DOC/DOCX) found under any category root.")
        return

    for category, clients in by_cat_client.items():
        for client_id, files in clients.items():
            print(f"\n==== Indexing {category}/{client_id} with {len(files)} file(s) ====")
            index_client(category, client_id, files)

if __name__ == "__main__":
    main()
