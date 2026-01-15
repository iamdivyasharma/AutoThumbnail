
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import openpyxl
except Exception:
    raise SystemExit("openpyxl is required. Install: pip install openpyxl")


def _safe_to_json(v: Any) -> Any:
    """Convert values to JSON-safe types."""
    if pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            pass
    try:
        import numpy as np
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (np.bool_)):
            return bool(v)
    except Exception:
        pass
    return v


def _infer_column_profile(series: pd.Series, max_unique: int) -> Dict[str, Any]:
    """Build a profile for a single column using a small sample."""
    non_null = series.dropna()
    dtype = str(series.dtype)

    profile: Dict[str, Any] = {
        "dtype": dtype,
        "nullable": bool(series.isna().any()),
        "null_fraction_sample": float(series.isna().mean()),
        "sample_values": [],
    }

    if non_null.empty:
        return profile

    # Sample values (unique)
    try:
        uniq = list(pd.unique(non_null))
        uniq = [_safe_to_json(x) for x in uniq[:max_unique]]
        profile["sample_values"] = uniq
    except Exception:
        profile["sample_values"] = [_safe_to_json(x) for x in non_null.head(min(5, len(non_null))).tolist()]

    # Numeric ranges if possible
    if pd.api.types.is_numeric_dtype(non_null):
        try:
            profile["min_sample"] = _safe_to_json(non_null.min())
            profile["max_sample"] = _safe_to_json(non_null.max())
        except Exception:
            pass

    # Datetime ranges if possible
    if pd.api.types.is_datetime64_any_dtype(non_null):
        try:
            profile["min_sample"] = _safe_to_json(non_null.min())
            profile["max_sample"] = _safe_to_json(non_null.max())
        except Exception:
            pass

    return profile


def build_schema(excel_path: Path, sample_rows: int = 200, max_unique: int = 20) -> Dict[str, Any]:
    """Build workbook schema as a JSON-serializable dict."""
    excel_path = excel_path.resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    sheet_names = wb.sheetnames

    result: Dict[str, Any] = {
        "workbook": str(excel_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rows_per_sheet": sample_rows,
        "max_unique_values_per_column": max_unique,
        "sheets": [],
    }

    for sh in sheet_names:
        ws = wb[sh]
        n_rows = int(ws.max_row or 0)
        n_cols = int(ws.max_column or 0)

        # Read only a small sample for profiling
        try:
            df = pd.read_excel(excel_path, sheet_name=sh, nrows=sample_rows, engine="openpyxl")
        except Exception:
            df = pd.read_excel(excel_path, sheet_name=sh, nrows=sample_rows)

        df.columns = [str(c).strip() for c in df.columns]

        columns_profile: Dict[str, Any] = {}
        for col in df.columns:
            columns_profile[col] = _infer_column_profile(df[col], max_unique=max_unique)

        result["sheets"].append({
            "name": sh,
            "n_rows_approx": n_rows,
            "n_cols": n_cols,
            "columns": columns_profile,
        })

    try:
        wb.close()
    except Exception:
        pass

    return result


def schema_to_markdown(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Excel Schema")
    lines.append("")
    lines.append(f"**Workbook:** `{schema.get('workbook')}`")
    lines.append(f"**Generated at:** `{schema.get('generated_at')}`")
    lines.append(f"**Sample rows per sheet:** `{schema.get('sample_rows_per_sheet')}`")
    lines.append("")

    for s in schema.get("sheets", []):
        lines.append(f"## Sheet: {s.get('name')}")
        lines.append(f"- Rows (approx): {s.get('n_rows_approx')}")
        lines.append(f"- Cols: {s.get('n_cols')}")
        lines.append("")
        lines.append("| Column | dtype | nullable | sample values |")
        lines.append("|---|---|---:|---|")

        cols = s.get("columns", {}) or {}
        for col, prof in cols.items():
            samples = prof.get("sample_values") or []
            sample_str = ", ".join([str(x) for x in samples[:8]])
            lines.append(f"| `{col}` | `{prof.get('dtype')}` | `{prof.get('nullable')}` | {sample_str} |")

        lines.append("")

    lines.append("---")
    lines.append("## LLM Prompt Snippet")
    lines.append("Use this schema to map user questions to exact fields/filters.")
    lines.append("")
    lines.append("> You MUST use only the column names listed above. If the user asks for a field that doesn't exist, reply with `unknown_field`.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel-path", required=True, help="Path to input Excel workbook (e.g., data/metadata.xlsx)")
    ap.add_argument("--out-json", default="schema.json", help="Output JSON path")
    ap.add_argument("--out-md", default="schema.md", help="Output Markdown path")
    ap.add_argument("--sample-rows", type=int, default=200, help="Rows to sample per sheet")
    ap.add_argument("--max-unique", type=int, default=20, help="Max unique sample values per column")
    args = ap.parse_args()

    schema = build_schema(
        excel_path=Path(args.excel_path),
        sample_rows=args.sample_rows,
        max_unique=args.max_unique,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(schema_to_markdown(schema), encoding="utf-8")

    print(f"[OK] Wrote schema JSON -> {out_json}")
    print(f"[OK] Wrote schema Markdown -> {out_md}")


if __name__ == "__main__":
    main()
