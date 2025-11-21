def load_metadata_for_file(doc_path: Path) -> Optional[dict]:
    """
    Given a document path like:
      /.../deal1.pdf
    we look for:
      /.../deal1.pdf.metadata.json
    or (fallback)
      /.../deal1.pdf.metadata

    We then return the metadataAtrributes block (preferred),
    or the whole JSON if that key is missing.
    """
    # Primary: *.metadata.json
    meta_candidates = [
        doc_path.with_name(doc_path.name + ".metadata.json"),
        doc_path.with_name(doc_path.name + ".metadata"),
    ]

    for meta_path in meta_candidates:
        if not meta_path.exists():
            continue

        try:
            text = meta_path.read_text(encoding="utf-8", errors="ignore").strip()
            data = json.loads(text)
        except Exception as e:
            print(f"[WARN] Failed to read metadata for {doc_path} from {meta_path}: {e}")
            continue

        # Your structure: { "metadataAtrributes": { "Account_Name": "..." , ... } }
        meta_block = (
            data.get("metadataAtrributes")    # exact key you mentioned
            or data.get("metadataAttributes") # fallback for spelling variant
            or data                          # fallback: whole dict
        )
        return meta_block

    return None


@dataclass
class DocInfo:
    policy_id: str
    source_file: str
    pages: int
    modified_time: int
    metadata: Optional[dict] = None


def build_manifest_entry(path: Path) -> DocInfo:
    pages = 0
    if path.suffix.lower() == ".pdf":
        try:
            with fitz.open(str(path)) as doc:
                pages = doc.page_count
        except Exception:
            pages = 0

    stat = path.stat()
    meta = load_metadata_for_file(path)  # <-- NEW: load file-wise metadata

    return DocInfo(
        policy_id=path.name,
        source_file=str(path.resolve()),
        pages=pages,
        modified_time=int(stat.st_mtime),
        metadata=meta,
    )




