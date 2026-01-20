from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# LangChain Bedrock (only when use_llm=True)
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Text normalization helpers
# -----------------------------

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def canonicalize_name(s: str) -> str:
    """Lowercase, remove punctuation, normalize spaces."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _clean_spaces(s)


# Optional: remove trailing legal suffix tokens for better acronym match
_LEGAL_SUFFIX_TOKENS = {
    "inc", "ltd", "llc", "plc", "as", "sa", "ag", "gmbh", "bv", "nv", "s",
}


def acronym(name: str) -> str:
    toks = canonicalize_name(name).split()
    if not toks:
        return ""

    # Remove trailing legal suffix tokens (ex: A/S -> tokens: a s)
    while toks and toks[-1] in _LEGAL_SUFFIX_TOKENS:
        toks.pop()

    if not toks:
        return ""
    return "".join(t[0] for t in toks).upper()


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = str(country).strip().upper()
    if c in {"UNITED STATES", "U.S.", "U.S", "US", "USA"}:
        return "US"
    if c in {"UNITED KINGDOM", "GREAT BRITAIN", "GB", "UK"}:
        return "UK"
    return c


def is_country_token(t: str) -> bool:
    t = (t or "").strip().upper()
    return bool(re.fullmatch(r"[A-Z]{2,3}", t)) or t in {"INDIA", "JAPAN", "DENMARK", "UK", "USA"}


STOPWORDS = {
    "tell", "me", "the", "a", "an", "for", "of", "in", "on", "to", "please",
    "what", "is", "are", "give", "show", "list", "get", "all", "files", "file",
    "names", "documents", "under", "about",
}

ATTRIBUTE_HINTS = [
    "agreement type",
    "effective date",
    "termination date",
    "expiry date",
    "governing law",
    "notice period",
    "renewal",
    "auto renewal",
    "payment term",
]


# Terms that should NEVER be treated as client identifiers
ID_WORDS = {
    "capid", "capoid", "cap", "cid", "client", "id", "clientid", "client_id",
}


# -----------------------------
# LLM prompts (deterministic)
# -----------------------------
_INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured search intent for a metadata lookup. "
            "Return ONLY valid JSON with keys: "
            "client_mention (string or null), country (string or null), "
            "category (legal|finance|null), attributes (array of strings), "
            "wants_file_list (boolean). "
            "Do not add extra keys.",
        ),
        (
            "human",
            "Query: {query}\n\n"
            "Examples:\n"
            "- 'capid for google us' => {{\"client_mention\":\"google\",\"country\":\"US\",\"category\":\"finance\",\"attributes\":[\"capid\"],\"wants_file_list\":false}}\n"
            "- 'agreement type of amgen us for all files' => {{\"client_mention\":\"amgen\",\"country\":\"US\",\"category\":\"finance\",\"attributes\":[\"agreement type\"],\"wants_file_list\":true}}\n"
            "- 'list all files under amgen us' => {{\"client_mention\":\"amgen\",\"country\":\"US\",\"category\":\"finance\",\"attributes\":[],\"wants_file_list\":true}}\n"
            "Return JSON now.",
        ),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a metadata assistant. Answer ONLY using the Provided Context. "
            "If the requested info is missing, say: 'Not found in available metadata.' "
            "If user asks for agreement type across files, return per-file results. "
            "Keep it short and direct.",
        ),
        ("human", "User question: {query}\n\nProvided Context:\n{context}\n\nAnswer:"),
    ]
)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ManifestDoc:
    policy_id: str
    source_file: Optional[str]
    metadata: Dict[str, Any]


# -----------------------------
# Main Service
# -----------------------------
class MetadataService:
    """
    Metadata lookup service

    Flow:
    1) Parse query (LLM intent or fallback).
    2) Correct client mention using alias mapping.
    3) Resolve client row from MASTER CSV (exact/acronym/substring/fuzzy).
    4) Resolve country + category.
    5) Locate manifest.json under /faiss_indices/<category>/<client>/<country?>/manifest.json
    6) Answer:
       - CAPID/CID direct
       - file list
       - per-file attribute values
       - final LLM answer (optional)
    """

    def __init__(
        self,
        client_master_csv_path: str,
        schema_json_path: str,
        faiss_index_dir: str,
        client_name_col: str = "parent_hq",
        country_col: str = "country__c",
        cid_col: str = "cid",
        bedrock_region: str = "us-east-1",
        llm_model_id: Optional[str] = None,
        client_alias_map: Optional[Dict[str, str]] = None,
        **_ignored_kwargs,
    ):
        load_dotenv()

        self.client_master_csv_path = Path(client_master_csv_path)
        self.schema_json_path = Path(schema_json_path)
        self.faiss_index_dir = Path(faiss_index_dir)

        self.client_name_col = client_name_col
        self.country_col = country_col
        self.cid_col = cid_col

        self.bedrock_region = bedrock_region
        self.llm_model_id = llm_model_id or os.getenv(
            "BEDROCK_LLM_MODEL",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        )

        # Loaded state
        self.df: Optional[pd.DataFrame] = None
        self.schema: Dict[str, Any] = {}
        self.manifest_rows: List[Tuple[str, str, Optional[str], Path]] = []

        # For fuzzy matching
        self._canon_choices: List[str] = []

        # session memory
        self.sessions: Dict[str, Dict[str, Optional[str]]] = {}

        # -------- Client alias mapping (your dictionary mapping) --------
        # Keys can be anything user types (APMM, maersk, etc.)
        # Values MUST be the EXACT client name as present in CSV parent_hq column.
        default_alias_map = {
            # Example (edit to match your CSV exactly):
            # "apmm": "A.P. Moller - Maersk A/S",
            # "maersk": "A.P. Moller - Maersk A/S",
            # "ap moller maersk": "A.P. Moller - Maersk A/S",
        }
        alias_map = client_alias_map or default_alias_map
        self._client_alias_map: Dict[str, str] = {
            canonicalize_name(k): v for k, v in alias_map.items()
        }

    # -------------------- loading --------------------
    def ensure_loaded(self) -> None:
        if self.df is None:
            self._load_master_csv()
        if not self.schema:
            self._load_schema_json()
        if not self.manifest_rows:
            self._scan_manifests()

    def _load_master_csv(self) -> None:
        if not self.client_master_csv_path.exists():
            raise FileNotFoundError(f"Master CSV not found: {self.client_master_csv_path}")

        df = pd.read_csv(self.client_master_csv_path, dtype=str, keep_default_na=False)
        df.columns = [str(c) for c in df.columns]

        for c in [self.client_name_col, self.country_col, self.cid_col]:
            if c not in df.columns:
                raise ValueError(
                    f"Master CSV missing column '{c}'. Found: {list(df.columns)[:50]}"
                )

        df["__canon_name"] = df[self.client_name_col].map(canonicalize_name)
        df["__acronym"] = df[self.client_name_col].map(acronym)
        df["__country"] = df[self.country_col].map(normalize_country)

        self.df = df
        self._canon_choices = df["__canon_name"].tolist()

    def _load_schema_json(self) -> None:
        # Kept for compatibility (even if not actively used by lookup)
        if not self.schema_json_path.exists():
            self.schema = {"columns": []}
            return
        try:
            self.schema = json.loads(self.schema_json_path.read_text(encoding="utf-8"))
        except Exception:
            self.schema = {"columns": []}

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
            # <category>/<client>/manifest.json OR <category>/<client>/<country>/manifest.json
            if len(parts) < 3:
                continue

            category = str(parts[0]).lower()
            client_folder = str(parts[1])
            country_folder = str(parts[2]) if len(parts) >= 4 else None

            rows.append((category, client_folder, country_folder, manifest_path))

        self.manifest_rows = rows

    # -------------------- alias mapping --------------------
    def _apply_client_alias_map(self, client_mention: str, query: str) -> str:
        """
        If detected client mention is a known alias, map to the correct CSV client name.
        Also scans the full query for any alias phrase (client can be anywhere).
        """
        cm_norm = canonicalize_name(client_mention)
        q_norm = canonicalize_name(query)

        # 1) Direct mapping for detected token/phrase
        if cm_norm and cm_norm in self._client_alias_map:
            return self._client_alias_map[cm_norm]

        # 2) Scan whole query for mapped aliases
        for alias_norm, real_name in self._client_alias_map.items():
            if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", q_norm):
                return real_name

        return client_mention

    # -------------------- LLM helpers --------------------
    def _llm(self) -> ChatBedrockConverse:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=self.bedrock_region)
        return ChatBedrockConverse(
            model_id=self.llm_model_id,
            client=client,
            temperature=0.0,
            max_tokens=700,
        )

    def _parse_intent_llm(self, query: str) -> Dict[str, Any]:
        chain = _INTENT_PROMPT | self._llm() | StrOutputParser()
        raw = (chain.invoke({"query": query}) or "").strip()

        # remove code fences if any
        raw = re.sub(r"^```(json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()

        try:
            data = json.loads(raw)
            data["country"] = normalize_country(data.get("country"))
            if data.get("category"):
                data["category"] = str(data["category"]).strip().lower()
            data["attributes"] = data.get("attributes") or []
            data["wants_file_list"] = bool(data.get("wants_file_list") or False)
            return data
        except Exception:
            return {
                "client_mention": None,
                "country": None,
                "category": None,
                "attributes": [],
                "wants_file_list": False,
            }

    # -------------------- fallback client extraction --------------------
    def _extract_client_guess(self, query: str) -> str:
        """
        Safe fallback: remove stopwords/country/id words, then return best remaining token.
        """
        q = canonicalize_name(query)
        toks = [t for t in q.split() if t and t not in STOPWORDS]
        toks = [t for t in toks if not is_country_token(t)]
        toks = [t for t in toks if t not in ID_WORDS]
        if not toks:
            return ""
        # pick the longest remaining token as a last resort
        return max(toks, key=len)

    def _resolve_country_from_query(self, query: str) -> Optional[str]:
        toks = re.split(r"[^A-Za-z0-9]+", query or "")
        for t in toks[::-1]:
            if is_country_token(t):
                return normalize_country(t)
        return None

    # -------------------- client resolution --------------------
    def _resolve_client_row(
        self,
        client_mention: str,
        country: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """
        Resolves using:
        1) exact canonical match
        2) acronym match (exact or prefix)
        3) substring match
        4) fuzzy match

        If country is provided and there are multiple matches, prefers matching country.
        """
        assert self.df is not None

        cm = _clean_spaces(client_mention)
        if not cm:
            return None

        cm_canon = canonicalize_name(cm)
        cm_acr = cm.strip().upper()
        country_norm = normalize_country(country)

        df = self.df

        # Helper: if multiple candidates, choose by country if possible
        def _pick_best(candidates: pd.DataFrame) -> Optional[pd.Series]:
            if candidates.empty:
                return None
            if country_norm and "__country" in candidates.columns:
                c2 = candidates[candidates["__country"] == country_norm]
                if not c2.empty:
                    return c2.iloc[0]
            return candidates.iloc[0]

        # 1) exact canonical
        exact = df[df["__canon_name"] == cm_canon]
        picked = _pick_best(exact)
        if picked is not None:
            return picked

        # 2) acronym exact
        acr_exact = df[df["__acronym"] == cm_acr]
        picked = _pick_best(acr_exact)
        if picked is not None:
            return picked

        # 2b) acronym prefix (APMM should match APMMAS)
        if len(cm_acr) >= 2:
            acr_prefix = df[df["__acronym"].str.startswith(cm_acr, na=False)]
            picked = _pick_best(acr_prefix)
            if picked is not None:
                return picked

        # 3) substring canonical
        if cm_canon:
            sub = df[df["__canon_name"].str.contains(re.escape(cm_canon), na=False)]
            picked = _pick_best(sub)
            if picked is not None:
                return picked

        # 4) fuzzy match
        try:
            from rapidfuzz import process, fuzz  # type: ignore

            match = process.extractOne(
                cm_canon,
                self._canon_choices,
                scorer=fuzz.token_set_ratio,
            )
            if match and match[1] >= 85:
                idx = self._canon_choices.index(match[0])
                return df.iloc[idx]
        except Exception:
            import difflib
            best = difflib.get_close_matches(cm_canon, self._canon_choices, n=1, cutoff=0.85)
            if best:
                idx = self._canon_choices.index(best[0])
                return df.iloc[idx]

        return None

    # -------------------- manifest selection --------------------
    def _candidate_manifests(
        self,
        client_name: str,
        category: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[Path]:
        cname = canonicalize_name(client_name)
        country_norm = normalize_country(country)

        candidates: List[Path] = []
        for (cat, client_folder, country_folder, mpath) in self.manifest_rows:
            if category and cat != category:
                continue

            folder_canon = canonicalize_name(client_folder)

            # allow folder to partially match client name
            if folder_canon != cname and cname not in folder_canon and folder_canon not in cname:
                continue

            if country_norm:
                if not country_folder:
                    continue
                if normalize_country(country_folder) != country_norm:
                    continue

            candidates.append(mpath)

        return candidates

    # -------------------- manifest parsing --------------------
    def _read_manifest_docs(self, manifest_path: Path) -> List[ManifestDoc]:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        out: List[ManifestDoc] = []
        for d in data.get("docs", []) or []:
            policy_id = str(d.get("policy_id") or "").strip()
            source_file = d.get("source_file")
            md = d.get("metadata") or {}

            # normalize nested metadata formats
            if isinstance(md, dict) and "metadataAttributes" in md and isinstance(md["metadataAttributes"], dict):
                md = md["metadataAttributes"]
            if isinstance(md, dict) and "metadataAtrributes" in md and isinstance(md["metadataAtrributes"], dict):
                md = md["metadataAtrributes"]

            if isinstance(md, dict):
                out.append(ManifestDoc(policy_id=policy_id, source_file=source_file, metadata=md))

        return out

    # -------------------- query detectors --------------------
    def _is_capid_query(self, query: str) -> bool:
        # include "capoid" (your typo case)
        q = (query or "").lower()
        return bool(re.search(r"\b(cap\s*(id|oid)|capid|capoid|client\s*id|cid)\b", q))

    def _is_list_files_query(self, query: str) -> bool:
        q = (query or "").lower()
        return any(p in q for p in [
            "list all files",
            "all the files",
            "file names",
            "filenames",
            "documents under",
            "all files",
        ])

    def _extract_attributes_fallback(self, query: str) -> List[str]:
        q = (query or "").lower()
        attrs: List[str] = []
        for a in ATTRIBUTE_HINTS:
            if a in q:
                attrs.append(a)
        # common shorthand
        if "agreement" in q and "type" in q and "agreement type" not in attrs:
            attrs.append("agreement type")
        return attrs

    def _best_key_match(self, want: str, keys: List[str]) -> Optional[str]:
        want_c = canonicalize_name(want).replace(" ", "")
        if not want_c:
            return None

        best = None
        best_score = 0.0

        for k in keys:
            kc = canonicalize_name(k).replace(" ", "")
            if not kc:
                continue

            if kc == want_c:
                return k

            score = 0.0
            if want_c in kc or kc in want_c:
                score = 0.90
            else:
                want_t = set(canonicalize_name(want).split())
                k_t = set(canonicalize_name(k).split())
                if want_t and want_t.issubset(k_t):
                    score = 0.82

            if score > best_score:
                best_score = score
                best = k

        return best if best_score >= 0.80 else None

    # -------------------- public main lookup --------------------
    def lookup(
        self,
        query: str,
        category: Optional[str] = None,
        country: Optional[str] = None,
        session_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> str:
        """
        Returns ONLY a final answer string.
        """
        self.ensure_loaded()
        assert self.df is not None

        category_norm = (category or "").strip().lower() or None
        country_norm = normalize_country(country)

        # session memory fallback
        if session_id and session_id in self.sessions:
            mem = self.sessions[session_id]
            if not category_norm:
                category_norm = mem.get("category")
            if not country_norm:
                country_norm = mem.get("country")

        # ---- intent parsing ----
        intent = {
            "client_mention": None,
            "country": None,
            "category": category_norm,
            "attributes": [],
            "wants_file_list": False,
        }

        if use_llm:
            intent = self._parse_intent_llm(query)
            if category_norm:
                intent["category"] = category_norm
            if country_norm:
                intent["country"] = country_norm
        else:
            intent["country"] = country_norm or self._resolve_country_from_query(query)
            intent["category"] = category_norm

        client_mention = (intent.get("client_mention") or "").strip()

        # fallback client extraction if LLM didn't provide it
        if not client_mention:
            client_mention = self._extract_client_guess(query)

        # apply alias mapping correction (client can be anywhere in query)
        client_mention = self._apply_client_alias_map(client_mention, query)

        # session memory fallback if still blank
        if not client_mention and session_id and session_id in self.sessions:
            client_mention = self.sessions[session_id].get("client_name") or ""

        if not client_mention:
            return "Client name not detected. Please include a client name like 'amgen' or 'google'."

        # resolve master CSV row (country-aware preference)
        row = self._resolve_client_row(client_mention, country=intent.get("country"))
        if row is None:
            return f"'{client_mention}' was not found in master CSV database."

        real_client_name = str(row[self.client_name_col]).strip()
        cid = str(row[self.cid_col]).strip()

        resolved_country = intent.get("country") or row.get("__country")
        resolved_country = normalize_country(resolved_country)

        resolved_category = intent.get("category") or "finance"
        if resolved_category not in {"finance", "legal"}:
            resolved_category = "finance"

        # update session memory
        if session_id:
            self.sessions[session_id] = {
                "client_name": client_mention,
                "country": resolved_country,
                "category": resolved_category,
            }

        # ---- CAPID direct ----
        if self._is_capid_query(query):
            if resolved_country:
                return f"CAPID / Client ID for {real_client_name} ({resolved_country}) is {cid}."
            return f"CAPID / Client ID for {real_client_name} is {cid}."

        # ---- manifest selection ----
        manifests = self._candidate_manifests(
            client_name=real_client_name,
            category=resolved_category,
            country=resolved_country,
        )

        # If none found, retry without country constraint
        if not manifests:
            manifests = self._candidate_manifests(
                client_name=real_client_name,
                category=resolved_category,
                country=None,
            )

        if not manifests:
            return (
                f"Found client '{real_client_name}' (CID={cid}) but no manifest.json was found "
                f"under {self.faiss_index_dir} for that client."
            )

        # read all docs from manifests
        all_docs: List[ManifestDoc] = []
        for mp in manifests[:12]:
            all_docs.extend(self._read_manifest_docs(mp))

        if not all_docs:
            return f"Found manifest.json for {real_client_name}, but it has no docs/metadata inside."

        # detect list-files + attributes
        wants_file_list = bool(intent.get("wants_file_list")) or self._is_list_files_query(query)
        attributes = intent.get("attributes") or []
        if not attributes:
            attributes = self._extract_attributes_fallback(query)

        # collect keys for matching
        all_keys = sorted({k for d in all_docs for k in (d.metadata or {}).keys()})

        # If user wants file list only
        if wants_file_list and not attributes:
            names = sorted({d.policy_id for d in all_docs if d.policy_id})
            if not names:
                return "No policy/file names found in manifest metadata."
            return "Files found:\n- " + "\n- ".join(names[:300])

        # Build structured context for LLM OR deterministic output
        context = self._build_context(
            client_name=real_client_name,
            cid=cid,
            category=resolved_category,
            country=resolved_country,
            docs=all_docs,
            wants_file_list=wants_file_list,
            want_attribute=(attributes[0] if attributes else None),
            all_keys=all_keys,
        )

        # If LLM enabled → final text answer
        if use_llm:
            return self._final_answer_llm(query=query, context=context)

        # Deterministic fallback (no LLM)
        header = ""
        if wants_file_list:
            names = sorted({d.policy_id for d in all_docs if d.policy_id})
            header = "Files found:\n- " + "\n- ".join(names[:50])

        if attributes:
            want = attributes[0]
            key = self._best_key_match(want, all_keys)
            if not key:
                return (header + "\n\n" + f"Attribute '{want}' was not found in manifest metadata.").strip()

            lines = []
            for d in all_docs:
                if key in d.metadata:
                    lines.append(f"{d.policy_id}: {key} = {d.metadata.get(key)}")

            if not lines:
                return (header + "\n\n" + f"No values found for '{key}' in manifest metadata.").strip()

            return (header + "\n\n" + "\n".join(lines[:200])).strip()

        return "I could not determine which metadata attribute you are asking for."

    # -------------------- context + final answer --------------------
    def _build_context(
        self,
        client_name: str,
        cid: str,
        category: str,
        country: Optional[str],
        docs: List[ManifestDoc],
        wants_file_list: bool,
        want_attribute: Optional[str],
        all_keys: List[str],
        max_docs: int = 40,
    ) -> str:
        lines: List[str] = []
        lines.append(f"CLIENT: {client_name}")
        lines.append(f"CID: {cid}")
        lines.append(f"CATEGORY: {category}")
        if country:
            lines.append(f"COUNTRY: {country}")
        lines.append("")

        if wants_file_list:
            names = sorted({d.policy_id for d in docs if d.policy_id})
            lines.append("FILES:")
            for n in names[:200]:
                lines.append(f"- {n}")
            lines.append("")

        if want_attribute:
            lines.append(f"REQUESTED_ATTRIBUTE: {want_attribute}")
            best_key = self._best_key_match(want_attribute, all_keys)
            lines.append(f"BEST_MATCH_METADATA_KEY: {best_key if best_key else '(not found)'}")
            lines.append("")

        lines.append("PER-FILE METADATA:")
        best_key = self._best_key_match(want_attribute, all_keys) if want_attribute else None

        for d in docs[:max_docs]:
            lines.append(f"- policy_id: {d.policy_id}")
            if best_key and best_key in d.metadata:
                lines.append(f"  {best_key}: {d.metadata.get(best_key)}")
            else:
                shown = 0
                for k, v in list(d.metadata.items())[:10]:
                    lines.append(f"  {k}: {v}")
                    shown += 1
                if shown == 0:
                    lines.append("  (no metadata fields)")
            lines.append("")

        return "\n".join(lines).strip()

    def _final_answer_llm(self, query: str, context: str) -> str:
        chain = _ANSWER_PROMPT | self._llm() | StrOutputParser()
        return (chain.invoke({"query": query, "context": context}) or "").strip()
