

from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import fitz 

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Optional: Textract PDF loader (for scanned PDFs)
try:
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    _HAS_TEXTRACT_LOADER = True
except Exception:
    _HAS_TEXTRACT_LOADER = False

# Optional: DOCX loader
try:
    from langchain_community.document_loaders import Docx2txtLoader
    _HAS_DOCX_LOADER = True
except Exception:
    _HAS_DOCX_LOADER = False



PDF_ROOTS = [
    Path("ce/finance").resolve(),
    Path("ce/legal").resolve(),
]

# Where to write FAISS indices on SageMaker
FAISS_INDEX_DIR = Path("/opt/ml/faiss_indices").resolve()



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


CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150


def clean_text(t: str) -> str:
    """Light cleanup; keep punctuation and newlines."""
    return re.sub(r"[ \t]+\n", "\n", t or "").strip()

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Tuple[int, int, str]]:
    
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

# ===================== CLIENT SCAN (LOCAL FOLDERS) =====================

def scan_client_folders(roots: List[Path]) -> Dict[str, List[Path]]:
    """
    Walk ce/finance and ce/legal and build:
        { client_id (folder name lowercased): [list of Paths (PDF+DOCX)] }

    Example:
      ce/finance/AcmeCorp/file.pdf  -> client_id "acmecorp"
      ce/legal/AcmeCorp/file.docx   -> grouped into same "acmecorp" client
    """
    out: Dict[str, List[Path]] = {}

    for root in roots:
        if not root.exists():
            print(f"[WARN] client root does not exist: {root}")
            continue

        for client_dir in root.iterdir():
            if not client_dir.is_dir():
                continue

            client_id = client_dir.name.strip().lower()

            pdfs  = list(client_dir.rglob("*.pdf"))
            docxs = list(client_dir.rglob("*.docx")) + list(client_dir.rglob("*.doc"))
            files = sorted(pdfs + docxs, key=lambda p: p.name.lower())

            if not files:
                continue

            out.setdefault(client_id, []).extend(files)

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

# ===================== TEXTRACT PATH (SCANNED PDF) =====================

def textract_pdf_to_page_texts(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Use AmazonTextractPDFLoader to extract text per page.
    Returns a list of (page_num, text).
    """
    if not _HAS_TEXTRACT_LOADER:
        print(f"[WARN] AmazonTextractPDFLoader not available; skipping Textract for {pdf_path.name}")
        return []

    try:
        loader = AmazonTextractPDFLoader(str(pdf_path))
        tex_docs = loader.load()  # usually 1 Document per page
    except Exception as e:
        print(f"[WARN] Textract extraction failed for {pdf_path.name}: {e}")
        return []

    page_texts: List[Tuple[int, str]] = []
    for page_idx, d in enumerate(tex_docs, start=1):
        text = clean_text(d.page_content or "")
        if not text.strip():
            continue
        page_texts.append((page_idx, text))

    return page_texts

# ===================== PDF CHUNKING =====================

def make_chunks_for_pdf(pdf_path: Path, client_id: str) -> Iterable[Document]:
    """
    Decide between:
      - PyMuPDF for text-based PDFs.
      - Textract for scanned PDFs.
    No Tesseract is used.
    """
    policy_id = pdf_path.name
    scanned = is_scanned_pdf(pdf_path)

    # ===== SCANNED PDF → TEXTRACT WITH PAGE RANGES =====
    if scanned:
        page_texts = textract_pdf_to_page_texts(pdf_path)
        if not page_texts:
            print(f"[WARN] No Textract text for scanned PDF {pdf_path}; skipping.")
            return

        # 1) Build one concatenated string and map char ranges -> pages
        concatenated = ""
        page_ranges: List[Tuple[int, int, int]] = []  # (start_char, end_char, page_num)
        cursor = 0
        for page_num, txt in page_texts:
            start = cursor
            concatenated += txt + "\n"
            cursor = len(concatenated)
            page_ranges.append((start, cursor, page_num))

        # 2) Chunk over concatenated text and compute page_start/page_end per chunk
        for c_start, c_end, chunk in chunk_text(concatenated):
            # Overlapping pages for this chunk
            pages = sorted({
                page_num
                for p_start, p_end, page_num in page_ranges
                if not (p_end <= c_start or p_start >= c_end)
            })
            if not pages:
                continue

            meta = {
                "client_id": client_id,
                "policy_id": policy_id,
                "page_start": pages[0],
                "page_end": pages[-1],
                "source_file": str(pdf_path.resolve()),
                "extracted_by": "textract",
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
                        "client_id": client_id,
                        "policy_id": policy_id,
                        "page": pnum + 1,
                        "source_file": str(pdf_path.resolve()),
                        "extracted_by": "pymupdf",
                    }
                    yield Document(page_content=chunk, metadata=meta)
    except Exception as e:
        print(f"[WARN] Failed to process PDF {pdf_path} via PyMuPDF: {e}")

# ===================== DOCX CHUNKING =====================

def make_chunks_for_docx(docx_path: Path, client_id: str) -> Iterable[Document]:
    """
    Extract text from DOCX using Docx2txtLoader (if available), then chunk.
    """
    if not _HAS_DOCX_LOADER:
        print(f"[WARN] DOCX loader not available; skipping DOCX {docx_path}")
        return

    policy_id = docx_path.name
    try:
        loader = Docx2txtLoader(str(docx_path))
        docs = loader.load()
    except Exception as e:
        print(f"[WARN] Failed to load DOCX {docx_path}: {e}")
        return

    for d in docs:
        text = clean_text(d.page_content or "")
        if not text.strip():
            continue
        for _, _, chunk in chunk_text(text):
            meta = {
                "client_id": client_id,
                "policy_id": policy_id,
                "page": None,
                "source_file": str(docx_path.resolve()),
                "extracted_by": "docx2txt",
            }
            yield Document(page_content=chunk, metadata=meta)

# ===================== FILE DISPATCH =====================

def make_chunks_for_file(path: Path, client_id: str) -> Iterable[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        yield from make_chunks_for_pdf(path, client_id)
    elif suf in (".doc", ".docx"):
        yield from make_chunks_for_docx(path, client_id)
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

def index_client(client_id: str, files: List[Path]) -> None:
    client_dir = FAISS_INDEX_DIR / client_id
    client_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    manifest_rows: List[DocInfo] = []

    for path in files:
        print(f"[{client_id}] Processing {path}")
        manifest_rows.append(build_manifest_entry(path))
        for doc in make_chunks_for_file(path, client_id):
            docs.append(doc)

    if not docs:
        print(f"[{client_id}] No extractable text; skipping index.")
        return

    print(f"[{client_id}] Building FAISS with {len(docs)} chunks…")
    vs = FAISS.from_documents(docs, embedder())
    vs.save_local(str(client_dir))

    manifest = {
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
    print(f"[{client_id}] Done. Index at {client_dir}")

# ===================== ENTRYPOINT =====================

def main():
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    by_client = scan_client_folders(PDF_ROOTS)
    if not by_client:
        print("No files (PDF/DOCX) found under any client root.")
        return

    total_files = sum(len(v) for v in by_client.values())
    print(f"Found {total_files} files across {len(by_client)} client(s).")

    for client_id, files in by_client.items():
        print(f"\n==== Indexing client '{client_id}' with {len(files)} file(s) ====")
        index_client(client_id, files)

if __name__ == "__main__":
    main()
