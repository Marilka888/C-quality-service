"""Parse a package folder (TZ/PZ/PMI files) into packages/<package_id>.json.

Usage:
python scripts/parse_package_from_files.py --input ./raw/pkg_001 --output ./data/packages/pkg_001.json
"""

import argparse, json, re
from pathlib import Path
import docx
import fitz  # PyMuPDF

def chunks_from_docx(path: Path):
    d = docx.Document(str(path))
    chunks, idx = [], 1
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        chunks.append({"chunk_id": f"{path.stem}_{idx:05d}", "section_path": None, "text": t, "order": idx, "page": None})
        idx += 1
    return chunks

def chunks_from_pdf(path: Path):
    doc = fitz.open(str(path))
    chunks, idx = [], 1
    for page_i in range(len(doc)):
        text = doc[page_i].get_text("text") or ""
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p in parts:
            chunks.append({"chunk_id": f"{path.stem}_{idx:05d}", "section_path": None, "text": p, "order": idx, "page": page_i + 1})
            idx += 1
    return chunks

def detect_doc_type(filename: str):
    name = filename.lower()
    if "тз" in name or "tz" in name: return "TZ"
    if "пз" in name or "pz" in name: return "PZ"
    if "пми" in name or "pmi" in name: return "PMI"
    return "OTHER"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--package_id", default=None)
    args = ap.parse_args()

    in_dir = Path(args.input)
    pkg_id = args.package_id or in_dir.name

    docs = []
    for fp in sorted(in_dir.iterdir()):
        if not fp.is_file():
            continue
        dt = detect_doc_type(fp.name)
        if fp.suffix.lower() == ".docx":
            ch = chunks_from_docx(fp)
        elif fp.suffix.lower() == ".pdf":
            ch = chunks_from_pdf(fp)
        else:
            continue
        docs.append({"doc_id": fp.stem, "doc_type": dt, "chunks": ch})

    out = {"package_id": pkg_id, "documents": docs}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
