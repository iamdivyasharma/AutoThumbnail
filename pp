# app.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.metadata_service import MetadataService
from services.rag_search_service import RagSearchService

load_dotenv()

app = FastAPI(title="Structured Metadata + RAG Retrieval API", version="2.1.0")

# ---- Services ----
svc = MetadataService(
    client_master_csv_path=os.getenv("CLIENT_MASTER_CSV", "Excel/Contractextraction list1.csv"),
    faiss_index_dir=os.getenv("FAISS_INDEX_DIR", "/opt/ml/faiss_indices"),
    client_name_col=os.getenv("CLIENT_NAME_COL", "parent_hq"),
    country_col=os.getenv("COUNTRY_COL", "country_code"),
    cid_col=os.getenv("CID_COL", "cid"),
    # optional:
    # client_alias_map={"apmm": "A.P. Moller - Maersk A/S", "maersk": "A.P. Moller - Maersk A/S"},
)

rag = RagSearchService(
    metadata_svc=svc,
    faiss_index_dir=os.getenv("FAISS_INDEX_DIR", "/opt/ml/faiss_indices"),
    bedrock_region=os.getenv("AWS_REGION", "us-east-1"),
    embedding_model_id=os.getenv("BEDROCK_EMBEDDING_MODEL", None),
    default_top_k=int(os.getenv("RAG_TOP_K", "8")),
    default_category=os.getenv("RAG_DEFAULT_CATEGORY", "finance"),
)


# -------------------------
# Structured Metadata Lookup
# -------------------------
class MetadataLookupStructuredRequest(BaseModel):
    client: str
    country: Optional[str] = None
    category: Optional[str] = "finance"
    include_manifest: bool = True
    max_docs_per_manifest: int = 200


class MetadataLookupStructuredResponse(BaseModel):
    client_input: str
    client_resolved: Optional[str] = None
    category: str
    country: Optional[str] = None

    capid: Optional[str] = None
    capids_by_country: List[Dict[str, str]] = Field(default_factory=list)

    manifest: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


@app.post("/api/metadata/lookup", response_model=MetadataLookupStructuredResponse)
def metadata_lookup_structured(payload: MetadataLookupStructuredRequest):
    if not payload.client or not payload.client.strip():
        raise HTTPException(400, "client is required")

    try:
        out = svc.lookup_structured(
            client=payload.client.strip(),
            country=payload.country,
            category=payload.category,
            include_manifest=bool(payload.include_manifest),
            max_docs_per_manifest=int(payload.max_docs_per_manifest),
        )
        return out
    except Exception as e:
        raise HTTPException(500, f"metadata lookup failed: {str(e)}")


# -------------------------
# Structured RAG Retrieval (chunks only)
# -------------------------
class RagRetrieveStructuredRequest(BaseModel):
    query: str
    client: str
    country: Optional[str] = None
    category: Optional[str] = "finance"
    top_k: Optional[int] = None


class RagChunkOut(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagRetrieveStructuredResponse(BaseModel):
    top_chunks: List[RagChunkOut] = Field(default_factory=list)
    client_resolved: str
    country: Optional[str] = None
    category: str


@app.post("/api/rag/search", response_model=RagRetrieveStructuredResponse)
def rag_retrieve_structured(payload: RagRetrieveStructuredRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(400, "query is required")
    if not payload.client or not payload.client.strip():
        raise HTTPException(400, "client is required")

    try:
        out = rag.retrieve_chunks_structured(
            query=payload.query.strip(),
            client=payload.client.strip(),
            country=payload.country,
            category=payload.category,
            top_k=payload.top_k,
        )
        return out
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"rag retrieval failed: {str(e)}")


@app.on_event("startup")
def _startup_load():
    try:
        svc.ensure_loaded()
    except Exception as e:
        print(f"[WARN] startup preload failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}



_________________

# services/rag_search_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from services.metadata_service import MetadataService, canonicalize_name, normalize_country

from langchain_aws import BedrockEmbeddings

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore


@dataclass
class RagChunk:
    # keep rank/score internally for ordering; DO NOT return them in API response
    rank: int
    score: float
    text: str
    metadata: Dict[str, Any]


class RagSearchService:
    """
    STRUCTURED retrieval-only RAG:

    Input:
      { "query": "...", "client": "...", "country": "US|UK|IN|...", "category": "finance|legal" }

    Routing rules (STRICT):
      - If country is provided -> ONLY /<category>/<client>/<country>/
        If missing -> raise "country not in indexer"
      - If country NOT provided -> ONLY /<category>/<client>/Ultimate Parent/ and/or /Corporate Family/
        (use both if exist)

    Output:
      - top_chunks: [{text, metadata}, ...]
      - client_resolved, country, category
    """

    def __init__(
        self,
        metadata_svc: MetadataService,
        faiss_index_dir: str,
        bedrock_region: str = "us-east-1",
        embedding_model_id: Optional[str] = None,
        default_top_k: int = 8,
        default_category: str = "finance",
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: float = 0.5,
    ):
        load_dotenv()
        self.meta = metadata_svc
        self.faiss_index_dir = Path(faiss_index_dir)
        self.bedrock_region = bedrock_region

        self.embedding_model_id = embedding_model_id or os.getenv(
            "BEDROCK_EMBEDDING_MODEL",
            "amazon.titan-embed-text-v1",
        )

        self.default_top_k = int(default_top_k)
        self.default_category = (default_category or "finance").strip().lower()
        if self.default_category not in {"finance", "legal"}:
            self.default_category = "finance"

        self.mmr_fetch_k = int(mmr_fetch_k) if mmr_fetch_k is not None else max(self.default_top_k * 5, 25)
        self.mmr_lambda_mult = float(mmr_lambda_mult)

        self._embeddings = None

    # ---------------- Embeddings ----------------
    def _emb(self):
        if self._embeddings is not None:
            return self._embeddings
        import boto3

        brt = boto3.client("bedrock-runtime", region_name=self.bedrock_region)
        self._embeddings = BedrockEmbeddings(model_id=self.embedding_model_id, client=brt)
        return self._embeddings

    # ---------------- Folder helpers ----------------
    def _match_client_dirs(self, category_dir: Path, client_name: str) -> List[Path]:
        if not category_dir.exists():
            return []
        target = canonicalize_name(client_name)
        if not target:
            return []
        out: List[Path] = []
        for d in category_dir.iterdir():
            if not d.is_dir():
                continue
            dc = canonicalize_name(d.name)
            if dc == target or target in dc or dc in target:
                out.append(d)
        return out

    def _find_subfolder(self, base: Path, want: str) -> Optional[Path]:
        w = canonicalize_name(want)
        if not base.exists() or not base.is_dir():
            return None
        for d in base.iterdir():
            if d.is_dir() and canonicalize_name(d.name) == w:
                return d
        return None

    def _possible_index_dirs_strict(self, client_name: str, category: str, country: Optional[str]) -> List[Path]:
        """
        STRICT:
          - country provided -> ONLY <client>/<country>
          - no country -> ONLY Ultimate Parent / Corporate Family
        """
        cat = (category or self.default_category).strip().lower()
        if cat not in {"finance", "legal"}:
            cat = self.default_category

        cat_dir = self.faiss_index_dir / cat
        client_dirs = self._match_client_dirs(cat_dir, client_name)
        if not client_dirs:
            return []

        ctry = normalize_country(country)

        if ctry:
            existing: List[Path] = []
            for cd in client_dirs:
                # exact
                p = cd / ctry
                if p.exists() and p.is_dir():
                    existing.append(p)
                    continue
                # normalized match (USA / U.S. / etc.)
                for sub in cd.iterdir():
                    if sub.is_dir() and normalize_country(sub.name) == ctry:
                        existing.append(sub)
            return existing  # if empty => "country not in indexer"

        out: List[Path] = []
        for cd in client_dirs:
            up = self._find_subfolder(cd, "Ultimate Parent")
            cf = self._find_subfolder(cd, "Corporate Family")
            if up:
                out.append(up)
            if cf:
                out.append(cf)
        return out

    # ---------------- FAISS ----------------
    def _load_faiss_store(self, folder: Path):
        try:
            return FAISS.load_local(str(folder), self._emb(), allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(str(folder), self._emb())

    def _json_sanitize(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_sanitize(v) for v in obj]
        return str(obj)

    # ---------------- Public: Structured Retrieval ----------------
    def retrieve_chunks_structured(
        self,
        query: str,
        client: str,
        country: Optional[str] = None,
        category: Optional[str] = "finance",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is required")

        client_in = (client or "").strip()
        if not client_in:
            raise ValueError("client is required")

        cat = (category or self.default_category).strip().lower()
        if cat not in {"finance", "legal"}:
            cat = self.default_category

        ctry = normalize_country(country)
        k = int(top_k) if top_k else self.default_top_k

        # Resolve client via CSV if possible; else use alias-mapped input
        self.meta.ensure_loaded()
        client_mapped = self.meta._apply_client_alias_map(client_in)
        row = self.meta._resolve_client_row(client_mapped, country=ctry)
        client_resolved = str(row[self.meta.client_name_col]).strip() if row is not None else client_mapped

        # STRICT routing
        index_dirs = self._possible_index_dirs_strict(client_resolved, cat, ctry)
        if not index_dirs:
            if ctry:
                raise FileNotFoundError(
                    f"Country '{ctry}' was provided, but index folder was not found under "
                    f"{self.faiss_index_dir}/{cat}/{client_resolved}/<{ctry}>/"
                )
            raise FileNotFoundError(
                f"No country provided, but neither Ultimate Parent nor Corporate Family index folders were found under "
                f"{self.faiss_index_dir}/{cat}/{client_resolved}/"
            )

        merged: List[RagChunk] = []

        for folder in index_dirs:
            store = self._load_faiss_store(folder)

            fetch_k = self.mmr_fetch_k
            lambda_mult = self.mmr_lambda_mult

            # 1) MMR docs (no scores)
            mmr_docs = store.max_marginal_relevance_search(q, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

            # 2) scored pool to map scores for ordering
            scored = store.similarity_search_with_score(q, k=fetch_k)

            def _doc_key(d) -> str:
                md = getattr(d, "metadata", {}) or {}
                md_items = sorted((str(k2), str(v2)) for k2, v2 in md.items())
                return (getattr(d, "page_content", "") or "") + "||" + "||".join([f"{k2}={v2}" for k2, v2 in md_items])

            score_map: Dict[str, float] = {}
            for d, s in scored:
                key = _doc_key(d)
                if key not in score_map or float(s) < score_map[key]:
                    score_map[key] = float(s)

            for d in mmr_docs:
                text = (d.page_content or "").strip()
                md = dict(d.metadata or {})
                md.setdefault("_index_folder", str(folder))
                sc = score_map.get(_doc_key(d), 0.0)
                merged.append(RagChunk(rank=0, score=float(sc), text=text, metadata=md))

        if not merged:
            return {
                "top_chunks": [],
                "client_resolved": client_resolved,
                "country": ctry,
                "category": cat,
            }

        # sort by FAISS distance (lower is usually better)
        merged.sort(key=lambda x: x.score)

        # de-dupe
        dedup: List[RagChunk] = []
        seen = set()
        for ch in merged:
            sig = (ch.text[:400], str(ch.metadata.get("policy_id") or ""), str(ch.metadata.get("_index_folder") or ""))
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(ch)

        top = dedup[:k]

        # return chunks (NO answer, NO sources)
        return {
            "top_chunks": [{"text": ch.text, "metadata": self._json_sanitize(ch.metadata)} for ch in top],
            "client_resolved": client_resolved,
            "country": ctry,
            "category": cat,
        }
________________


# services/metadata_service.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Normalization helpers
# -----------------------------
def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def canonicalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _clean_spaces(s)


_LEGAL_SUFFIX_TOKENS = {"inc", "ltd", "llc", "plc", "as", "sa", "ag", "gmbh", "bv", "nv", "s"}


def acronym(name: str) -> str:
    toks = canonicalize_name(name).split()
    while toks and toks[-1] in _LEGAL_SUFFIX_TOKENS:
        toks.pop()
    return "".join(t[0] for t in toks).upper() if toks else ""


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = str(country).strip().upper()

    if c in {"UNITED STATES", "U.S.", "U.S", "US", "USA"}:
        return "US"
    if c in {"UNITED KINGDOM", "GREAT BRITAIN", "GB", "UK"}:
        return "UK"
    if c in {"INDIA", "IN", "IND"}:
        return "IN"

    return c


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ManifestDoc:
    policy_id: str
    source_file: Optional[str]
    metadata: Dict[str, Any]


# -----------------------------
# MetadataService (Structured only)
# -----------------------------
class MetadataService:
    def __init__(
        self,
        client_master_csv_path: str,
        faiss_index_dir: str,
        client_name_col: str = "parent_hq",
        country_col: str = "country_code",
        cid_col: str = "cid",
        client_alias_map: Optional[Dict[str, str]] = None,
        **_ignored_kwargs,  # keeps compatibility if extra args passed accidentally
    ):
        self.client_master_csv_path = Path(client_master_csv_path)
        self.faiss_index_dir = Path(faiss_index_dir)

        self.client_name_col = client_name_col
        self.country_col = country_col
        self.cid_col = cid_col

        self.df: Optional[pd.DataFrame] = None
        self.manifest_rows: List[Tuple[str, str, Optional[str], Path]] = []
        self._canon_choices: List[str] = []

        alias_map = client_alias_map or {}
        self._client_alias_map: Dict[str, str] = {canonicalize_name(k): v for k, v in alias_map.items()}

    # -------------------- loading --------------------
    def ensure_loaded(self) -> None:
        if self.df is None:
            self._load_master_csv()
        if not self.manifest_rows:
            self._scan_manifests()

    def _load_master_csv(self) -> None:
        if not self.client_master_csv_path.exists():
            raise FileNotFoundError(f"Master CSV not found: {self.client_master_csv_path}")

        df = pd.read_csv(self.client_master_csv_path, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]

        for c in [self.client_name_col, self.country_col, self.cid_col]:
            if c not in df.columns:
                raise ValueError(f"Master CSV missing column '{c}'. Found: {list(df.columns)[:80]}")

        df["__canon_name"] = df[self.client_name_col].map(canonicalize_name)
        df["__acronym"] = df[self.client_name_col].map(acronym)
        df["__country"] = df[self.country_col].map(normalize_country)

        self.df = df
        self._canon_choices = df["__canon_name"].tolist()

    def _scan_manifests(self) -> None:
        if not self.faiss_index_dir.exists():
            raise FileNotFoundError(f"FAISS index dir not found: {self.faiss_index_dir}")

        rows: List[Tuple[str, str, Optional[str], Path]] = []
        for manifest_path in self.faiss_index_dir.rglob("manifest.json"):
            try:
                rel = manifest_path.relative_to(self.faiss_index_dir)
            except Exception:
                continue

            parts = rel.parts
            # category/client/manifest.json  OR category/client/<sub>/manifest.json
            if len(parts) < 3:
                continue

            category = str(parts[0]).lower()
            client_folder = str(parts[1])
            third_folder = str(parts[2]) if len(parts) >= 4 else None

            rows.append((category, client_folder, third_folder, manifest_path))

        self.manifest_rows = rows

    # -------------------- alias map --------------------
    def _apply_client_alias_map(self, client_value: str) -> str:
        cm = canonicalize_name(client_value)
        return self._client_alias_map.get(cm, client_value)

    # -------------------- resolve CSV row --------------------
    def _resolve_client_row(self, client_value: str, country: Optional[str]) -> Optional[pd.Series]:
        assert self.df is not None

        cm = _clean_spaces(client_value)
        if not cm:
            return None

        cm_canon = canonicalize_name(cm)
        cm_acr = cm.strip().upper()
        country_norm = normalize_country(country)

        df = self.df

        def pick_best(candidates: pd.DataFrame) -> Optional[pd.Series]:
            if candidates.empty:
                return None
            if country_norm:
                c2 = candidates[candidates["__country"] == country_norm]
                if not c2.empty:
                    return c2.iloc[0]
            return candidates.iloc[0]

        # 1) exact canonical
        exact = df[df["__canon_name"] == cm_canon]
        r = pick_best(exact)
        if r is not None:
            return r

        # 2) acronym exact
        acr = df[df["__acronym"] == cm_acr]
        r = pick_best(acr)
        if r is not None:
            return r

        # 3) substring
        if cm_canon:
            sub = df[df["__canon_name"].str.contains(re.escape(cm_canon), na=False)]
            r = pick_best(sub)
            if r is not None:
                return r

        # 4) fuzzy
        try:
            from rapidfuzz import process, fuzz  # type: ignore

            match = process.extractOne(cm_canon, self._canon_choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= 85:
                return df.iloc[int(match[2])]
        except Exception:
            import difflib

            best = difflib.get_close_matches(cm_canon, self._canon_choices, n=1, cutoff=0.85)
            if best:
                return df.iloc[self._canon_choices.index(best[0])]

        return None

    # -------------------- manifest selection --------------------
    def _client_folder_matches(self, client_folder: str, client_name: str) -> bool:
        a = canonicalize_name(client_folder)
        b = canonicalize_name(client_name)
        return a == b or a in b or b in a

    def _candidate_manifests(
        self,
        client_name: str,
        category: str,
        third_folder: Optional[str],  # country OR Ultimate Parent / Corporate Family OR None
    ) -> List[Path]:
        third_norm = canonicalize_name(third_folder) if third_folder else None
        out: List[Path] = []

        for (cat, client_folder, third, mpath) in self.manifest_rows:
            if cat != category:
                continue
            if not self._client_folder_matches(client_folder, client_name):
                continue

            if third_norm is None:
                out.append(mpath)
            else:
                if third and canonicalize_name(third) == third_norm:
                    out.append(mpath)

        return out

    # -------------------- manifest parsing --------------------
    def _read_manifest_docs(self, manifest_path: Path) -> List[ManifestDoc]:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        docs_out: List[ManifestDoc] = []
        for d in data.get("docs", []) or []:
            policy_id = str(d.get("policy_id") or "").strip()
            source_file = d.get("source_file")

            md = d.get("metadata") or {}
            if isinstance(md, dict) and "metadataAttributes" in md and isinstance(md["metadataAttributes"], dict):
                md = md["metadataAttributes"]
            if isinstance(md, dict) and "metadataAtrributes" in md and isinstance(md["metadataAtrributes"], dict):
                md = md["metadataAtrributes"]

            if isinstance(md, dict):
                docs_out.append(ManifestDoc(policy_id=policy_id, source_file=source_file, metadata=md))

        return docs_out

    # -----------------------------
    # PUBLIC: Structured lookup
    # -----------------------------
    def lookup_structured(
        self,
        client: str,
        country: Optional[str] = None,
        category: Optional[str] = "finance",
        include_manifest: bool = True,
        max_docs_per_manifest: int = 200,
    ) -> Dict[str, Any]:
        self.ensure_loaded()
        assert self.df is not None

        client_in = (client or "").strip()
        if not client_in:
            return {"error": "client is required"}

        cat = (category or "finance").strip().lower()
        if cat not in {"finance", "legal"}:
            cat = "finance"

        ctry = normalize_country(country)

        # alias mapping first
        client_mapped = self._apply_client_alias_map(client_in)

        # resolve CSV row (prefer matching country)
        row = self._resolve_client_row(client_mapped, ctry)
        if row is None:
            return {
                "client_input": client_in,
                "client_resolved": None,
                "category": cat,
                "country": ctry,
                "capid": None,
                "capids_by_country": [],
                "manifest": {},
                "error": f"'{client_mapped}' was not found in master CSV database.",
            }

        client_resolved = str(row[self.client_name_col]).strip()
        capid = str(row[self.cid_col]).strip()

        capids_by_country: List[Dict[str, str]] = []
        if not ctry:
            canon = canonicalize_name(client_resolved)
            matches = self.df[self.df["__canon_name"] == canon]
            if matches.empty:
                matches = self.df[self.df["__canon_name"].str.contains(re.escape(canon), na=False)]

            seen = set()
            for _, r in matches.iterrows():
                cc = normalize_country(r.get(self.country_col)) or (r.get("__country") or "").strip() or "N/A"
                cid_val = str(r.get(self.cid_col) or "").strip()
                if not cid_val:
                    continue
                key = (cc, cid_val)
                if key in seen:
                    continue
                seen.add(key)
                capids_by_country.append({"country": cc, "capid": cid_val})

        manifest_payload: Dict[str, Any] = {
            "manifests_found": 0,
            "manifests": [],
            "metadata_keys": [],
            "docs_truncated": False,
        }

        if include_manifest:
            manifests: List[Path] = []

            if ctry:
                # country folder manifests: category/client/<country>/manifest.json
                manifests = self._candidate_manifests(client_resolved, cat, ctry)
            else:
                # no country: ONLY Ultimate Parent + Corporate Family
                manifests = (
                    self._candidate_manifests(client_resolved, cat, "Ultimate Parent")
                    + self._candidate_manifests(client_resolved, cat, "Corporate Family")
                )

            all_keys = set()
            for mp in manifests[:10]:
                docs = self._read_manifest_docs(mp)
                truncated = False
                if len(docs) > max_docs_per_manifest:
                    docs = docs[:max_docs_per_manifest]
                    truncated = True
                    manifest_payload["docs_truncated"] = True

                docs_out = []
                for d in docs:
                    for k in (d.metadata or {}).keys():
                        all_keys.add(str(k))
                    docs_out.append(
                        {
                            "policy_id": d.policy_id,
                            "source_file": d.source_file,
                            "metadata": d.metadata,
                        }
                    )

                manifest_payload["manifests"].append(
                    {
                        "manifest_path": str(mp),
                        "docs_count_returned": len(docs_out),
                        "docs_truncated": truncated,
                        "docs": docs_out,
                    }
                )

            manifest_payload["manifests_found"] = len(manifest_payload["manifests"])
            manifest_payload["metadata_keys"] = sorted(all_keys)

        return {
            "client_input": client_in,
            "client_resolved": client_resolved,
            "category": cat,
            "country": ctry,
            "capid": capid if ctry else None,
            "capids_by_country": capids_by_country if not ctry else [],
            "manifest": manifest_payload,
            "error": None,
        }




