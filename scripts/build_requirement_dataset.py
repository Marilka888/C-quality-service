"""
Build a sentence-level dataset for the binary "is requirement" classifier.

Inputs:
  - data/c_pairs_variant_a.csv        — source of confirmed positives (tz_req_text)
  - data/packages/*.json              — output of batch_parse_packages.py
                                        (contains TZ documents with chunks[])

Outputs (to --out, default data/req_dataset/):
  - train.jsonl
  - val.jsonl
  - test.jsonl
  - stats.json

Each JSONL line:
  {"text": str, "label": 0|1, "package_id": str,
   "source": "csv" | "pkg_json", "chunk_id": str | null}

Positives    — unique normalised tz_req_text values from the CSV, plus any
               package-JSON sentences that fuzzy-match one of them.
Negatives    — sentences from package-JSON TZ chunks that do NOT fuzzy-match
               any positive (i.e. descriptive text, headings, definitions, …).

Splits are stratified by package_id: a given TZ appears in exactly one split.
This prevents train/test leakage where paraphrased variants of the same
requirement would be split across partitions.

Usage:
  python scripts/build_requirement_dataset.py \
      --csv ./data/c_pairs_variant_a.csv \
      --packages ./data/packages \
      --out ./data/req_dataset
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# rapidfuzz is the preferred fuzzy matcher, but we fall back to a lightweight
# token-Jaccard score if it is not installed — the dataset builder must run
# in plain Python environments (e.g. Colab without extra installs).
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _rf_fuzz

    def _similarity(a: str, b: str) -> float:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0

    _MATCHER_NAME = "rapidfuzz.token_set_ratio"
except Exception:
    def _tok_set(s: str) -> set:
        return set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s.lower()))

    def _similarity(a: str, b: str) -> float:
        ta, tb = _tok_set(a), _tok_set(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    _MATCHER_NAME = "jaccard_fallback"


# ---------------------------------------------------------------------------
# Reuse the Russian sentence splitter from the main pipeline — keeping a
# single implementation avoids train/serve skew (the classifier must see
# sentences tokenised exactly the same way at inference time).
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.application.use_cases.build_requirements import (  # noqa: E402
    _classify_section,
    _is_requirement_section_by_title,
    _split_sentences_ru,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalise_key(text: str) -> str:
    """Lowercase, collapsed whitespace, stripped trailing punctuation."""
    t = _normalise(text).lower()
    t = re.sub(r"[\s\.,;:!?\-–—«»\"']+$", "", t)
    return t


def _is_reasonable_sentence(text: str, min_words: int = 5, max_words: int = 80) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


_TZ_FILENAME_HINTS = ("tz", "тз", "техническ", "требовани")

# Section kind in the training dataset:
#   "req"     — requirement section (4.x by GOST or title matches req keywords)
#   "non_req" — descriptive section (Введение / Термины / Содержание / …)
#   "unknown" — structure unclear; SKIP from training to avoid label noise
SectionKind = str  # "req" | "non_req" | "unknown"


def _filename_looks_like_tz(filename: str) -> bool:
    low = filename.lower()
    return any(h in low for h in _TZ_FILENAME_HINTS)


def _classify_kind(section_id: Optional[str], title: Optional[str]) -> SectionKind:
    verdict = _classify_section(section_id, title)
    if verdict is True:
        return "req"
    if verdict is False:
        return "non_req"
    return "unknown"


# ---------------------------------------------------------------------------
# Heading detection for `chunks[]` format
# ---------------------------------------------------------------------------
#
# `batch_parse_packages.py` emits flat chunks with `section_path: None`, so
# we have to reconstruct section context by scanning chunks sequentially.
# A "heading" chunk is a short line that introduces a new section; subsequent
# chunks are assumed to live in that section until the next heading.
#
# Heuristic: chunk text, stripped, that is <= 15 words and either
#   (a) starts with a section number like "1.", "4.1", "4.2.3"
#   (b) is entirely a short recognisable non-req title ("ВВЕДЕНИЕ", …)
# is treated as a heading. We then call _classify_kind on (num, title-text).

_HEADING_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.\s]+(.+?)\s*$")
_ALL_CAPS_TITLE_RE = re.compile(r"^[А-ЯЁA-Z][А-ЯЁA-Z\s\d]{2,}$")

# Trigger words that indicate a body sentence rather than a heading. When one
# of these appears after a leading section number (e.g. "4.3.1 Система
# должна …"), we treat the whole line as requirement prose, not as a
# structural heading. Missing this check in v3 caused real requirements with
# glued-in section numbers to be labelled as negatives.
_BODY_TRIGGER_RE = re.compile(
    r"\b(должен|должна|должны|должно|не\s+должен|не\s+должна|не\s+должно|"
    r"необходимо|обязан|обязана|обязаны|обеспечивать|реализовывать|"
    r"поддерживать|хранить|предусматривать|выполнять|предоставлять|"
    r"осуществлять|контролировать|следует|требуется|разрешено|запрещено|"
    r"допускается|не\s+допускается)\b",
    re.I,
)

# A heading title has at most this many words after the leading section
# number. Real GOST-style headings are short (2–5 words typically).
_MAX_HEADING_TITLE_WORDS = 8


def _try_heading(text: str) -> Optional[Tuple[str, str]]:
    """Return (section_id, title) if `text` looks like a heading chunk, else None."""
    stripped = text.strip().strip(".").strip()
    if not stripped:
        return None
    # Any modality/requirement trigger anywhere in the line rules out the
    # "heading" verdict — this catches "4.3.1 Система должна …" style chunks
    # where a section number is glued onto a body sentence by the PDF parser.
    if _BODY_TRIGGER_RE.search(stripped):
        return None
    m = _HEADING_NUM_RE.match(stripped)
    if m:
        title_part = m.group(2).strip()
        if 0 < len(title_part.split()) <= _MAX_HEADING_TITLE_WORDS:
            return m.group(1), title_part
        return None
    # Pure title without numbering (e.g. "ВВЕДЕНИЕ", "Содержание")
    if len(stripped.split()) <= _MAX_HEADING_TITLE_WORDS and (
        _ALL_CAPS_TITLE_RE.match(stripped) or len(stripped.split()) <= 4
    ):
        # Only treat as a heading if the title itself is a req/non-req hint —
        # avoids misclassifying short body sentences as structural markers.
        if _is_requirement_section_by_title(stripped) is not None:
            return "", stripped
    return None


def _iter_sections_format(fp: Path) -> Iterable[Tuple[str, str, str, SectionKind]]:
    """
    `actual/pdf|docx/*.json` format:
      {"ok": bool, "format": "pdf"|"docx", "filename": "...",
       "sections": [{"sectionId": str, "num": str, "title": str, "text": str}]}
    """
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Skipping {fp.name}: {exc}", file=sys.stderr)
        return
    filename = data.get("filename") or fp.name
    if not _filename_looks_like_tz(filename):
        return
    package_id = fp.stem
    for sec in data.get("sections", []):
        title = (sec.get("title") or "").strip()
        section_id = sec.get("sectionId") or sec.get("num") or title
        kind = _classify_kind(section_id, title)
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        for sent in _split_sentences_ru(text):
            sent = _normalise(sent)
            if _is_reasonable_sentence(sent):
                yield package_id, str(section_id), sent, kind


def _iter_chunks_format(fp: Path) -> Iterable[Tuple[str, str, str, SectionKind]]:
    """
    `data/packages/*.json` format produced by scripts/batch_parse_packages.py.

    Reconstructs section context via heading detection (section_path is None
    in that format, so we walk chunks sequentially and flip the current
    section kind whenever a heading-shaped chunk appears).
    """
    try:
        pkg = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Skipping {fp.name}: {exc}", file=sys.stderr)
        return
    package_id = pkg.get("package_id") or fp.stem
    for doc in pkg.get("documents", []):
        if (doc.get("doc_type") or "").upper() != "TZ":
            continue
        # Before the first recognised heading we are on the title/cover page
        # and the "Introduction" of a GOST-style TZ — conventionally non-req.
        # Defaulting to "non_req" here means boilerplate like
        # "Настоящий документ разработан в соответствии с ГОСТ …" becomes
        # training signal for the negative class instead of being skipped.
        current_kind: SectionKind = "non_req"
        current_section_id = "preamble"
        for ch in doc.get("chunks", []):
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            chunk_id = ch.get("chunk_id") or "unknown"

            heading = _try_heading(text)
            if heading is not None:
                section_id, title = heading
                new_kind = _classify_kind(section_id, title)
                if new_kind != "unknown":
                    current_kind = new_kind
                    current_section_id = section_id or title

                # The heading text itself is a structural marker, not a
                # requirement — emit it as a non-req training row so the
                # classifier learns "short title-shaped lines are not reqs".
                # We skip headings that are themselves very short
                # single-token fragments (page numbers etc.) which would
                # be filtered by _is_reasonable_sentence anyway.
                heading_text = _normalise(text)
                if _is_reasonable_sentence(heading_text, min_words=1):
                    yield package_id, chunk_id, heading_text, "non_req"
                continue

            for sent in _split_sentences_ru(text):
                sent = _normalise(sent)
                if _is_reasonable_sentence(sent):
                    yield package_id, chunk_id, sent, current_kind


def _detect_format(fp: Path) -> str:
    """Peek at the JSON and return 'sections' | 'chunks' | 'unknown'."""
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    if isinstance(data, dict):
        if "sections" in data and "documents" not in data:
            return "sections"
        if "documents" in data:
            return "chunks"
    return "unknown"


def _iter_tz_sentences(packages_dirs: List[Path]) -> Iterable[Tuple[str, str, str, bool]]:
    """
    Yield (package_id, chunk_or_section_id, sentence, non_req_title_hint)
    across one or more input directories, auto-detecting JSON format.
    """
    for pkg_dir in packages_dirs:
        if not pkg_dir.exists():
            print(f"[WARN] Packages dir does not exist: {pkg_dir}", file=sys.stderr)
            continue
        for fp in sorted(pkg_dir.glob("*.json")):
            fmt = _detect_format(fp)
            if fmt == "sections":
                yield from _iter_sections_format(fp)
            elif fmt == "chunks":
                yield from _iter_chunks_format(fp)
            else:
                print(f"[WARN] Unknown JSON format: {fp}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def load_positives_from_csv(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path, usecols=["tz_req_text"])
    texts = [_normalise(t) for t in df["tz_req_text"].dropna().astype(str)]
    texts = [t for t in texts if _is_reasonable_sentence(t, min_words=3)]
    # Deduplicate on normalised key
    seen: Dict[str, str] = {}
    for t in texts:
        k = _normalise_key(t)
        if k and k not in seen:
            seen[k] = t
    return list(seen.values())


def classify_rows(
    sentences: List[Tuple[str, str, str, SectionKind]],
    positives: List[str],
    threshold: float,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], Dict[str, int]]:
    """
    Section-aware labelling pipeline.

    Rules:
      - sentences from `non_req` sections → negatives (no further check)
      - sentences from `req` sections     → positives ONLY if they fuzzy-match
                                            a CSV positive; otherwise SKIPPED
                                            (grey-zone: we don't know)
      - sentences from `unknown` sections → SKIPPED (avoid label noise)
      - CSV-matched sentence in non_req section → stays NEGATIVE (section
        verdict wins; CSV has false positives — e.g. document headers were
        shown to be picked up as CSV positives during v1 inspection)

    Returns (positives_rows, negatives_rows, stats).
    """
    # Index positives by word-count bucket for faster matching
    by_wc: Dict[int, List[str]] = defaultdict(list)
    for p in positives:
        by_wc[len(p.split())].append(p)
    buckets = sorted(by_wc.keys())

    def _matches_any_positive(sent: str) -> bool:
        wc = len(sent.split())
        lo, hi = max(1, wc // 2), wc * 2
        for b in buckets:
            if b < lo or b > hi:
                continue
            for p in by_wc[b]:
                if _similarity(sent, p) >= threshold:
                    return True
        return False

    positives_out: List[Tuple[str, str, str]] = []
    negatives_out: List[Tuple[str, str, str]] = []
    stats: Dict[str, int] = {
        "non_req_sentences": 0,
        "req_sentences_total": 0,
        "req_sentences_matched": 0,
        "req_sentences_grey_skipped": 0,
        "unknown_sentences_skipped": 0,
    }

    stats["modality_gated_negatives_skipped"] = 0

    for pkg, sid, sent, kind in sentences:
        has_modality = bool(_BODY_TRIGGER_RE.search(sent))

        if kind == "non_req":
            # A sentence with explicit modality ("должен" / "необходимо" /
            # "обеспечивать" …) must never end up in the negative class,
            # regardless of which section our heuristics placed it in. This
            # was the core bug in v3/v4: packages whose section headings the
            # regex failed to recognise defaulted to "non_req", and every
            # real requirement inside them ended up as a false negative,
            # teaching the model the inverse of what we wanted.
            if has_modality:
                stats["modality_gated_negatives_skipped"] += 1
                continue
            negatives_out.append((pkg, sid, sent))
            stats["non_req_sentences"] += 1
        elif kind == "req":
            stats["req_sentences_total"] += 1
            if _matches_any_positive(sent):
                positives_out.append((pkg, sid, sent))
                stats["req_sentences_matched"] += 1
            else:
                stats["req_sentences_grey_skipped"] += 1
        else:  # unknown
            stats["unknown_sentences_skipped"] += 1

    return positives_out, negatives_out, stats


def stratified_split_by_package(
    package_ids: List[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, str]:
    """
    Return {package_id: "train" | "val" | "test"}.
    Entire TZ documents are kept together in one split.
    """
    rng = random.Random(seed)
    ids = sorted(set(package_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # Remaining goes to test; adjust so all splits are non-empty when possible
    n_test = max(n - n_train - n_val, 0)
    if n_test == 0 and n >= 3:
        n_train -= 1
        n_test = 1
    assignment: Dict[str, str] = {}
    for i, pid in enumerate(ids):
        if i < n_train:
            assignment[pid] = "train"
        elif i < n_train + n_val:
            assignment[pid] = "val"
        else:
            assignment[pid] = "test"
    return assignment


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_rows(
    csv_positives: List[str],
    matched_sents: List[Tuple[str, str, str]],
    negatives: List[Tuple[str, str, str]],
    neg_ratio: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)

    rows: List[dict] = []

    for t in csv_positives:
        rows.append({
            "text": t,
            "label": 1,
            "package_id": "_csv_positives",
            "source": "csv",
            "chunk_id": None,
        })

    for pkg, chunk, sent in matched_sents:
        rows.append({
            "text": sent,
            "label": 1,
            "package_id": pkg,
            "source": "pkg_json",
            "chunk_id": chunk,
        })

    # Balance negatives to neg_ratio × positives
    pos_count = sum(1 for r in rows if r["label"] == 1)
    target_neg = pos_count * neg_ratio
    rng.shuffle(negatives)
    picked = negatives[:target_neg] if len(negatives) > target_neg else negatives

    for pkg, chunk, sent in picked:
        rows.append({
            "text": sent,
            "label": 0,
            "package_id": pkg,
            "source": "pkg_json",
            "chunk_id": chunk,
        })

    rng.shuffle(rows)
    return rows


def split_rows(rows: List[dict], assignment: Dict[str, str]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    # CSV-positive bucket goes to train only — these are the seed supervision
    # signal; evaluating on them would be circular.
    for r in rows:
        pid = r["package_id"]
        if pid == "_csv_positives":
            out["train"].append(r)
        else:
            split = assignment.get(pid, "train")
            out[split].append(r)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./data/c_pairs_variant_a.csv")
    ap.add_argument(
        "--packages",
        action="append",
        default=None,
        help="Directory with TZ JSONs. Pass multiple times to combine sources. "
             "Supports both formats: data/packages/*.json (chunks[]) and "
             "actual/**/*.json (sections[]). Default: ./data/packages",
    )
    ap.add_argument("--out", default="./data/req_dataset")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.85,
                    help="Similarity threshold (0..1) for positive match.")
    ap.add_argument("--neg-ratio", type=int, default=3,
                    help="Negatives per positive in the final dataset.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    pkg_dirs = [Path(p) for p in (args.packages or ["./data/packages"])]
    out_dir = Path(args.out)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    existing_dirs = [d for d in pkg_dirs if d.exists()]
    if not existing_dirs:
        raise SystemExit(
            f"None of the --packages dirs exist: {pkg_dirs}. "
            f"Run scripts/batch_parse_packages.py first, "
            f"or point --packages at e.g. C:/Users/Marilka/PycharmProjects/actual/pdf"
        )

    print(f"[info] matcher: {_MATCHER_NAME}")
    print(f"[info] loading positives from {csv_path}")
    csv_positives = load_positives_from_csv(csv_path)
    print(f"[info] unique positives from CSV: {len(csv_positives)}")

    print(f"[info] streaming TZ sentences from {[str(d) for d in existing_dirs]}")
    all_sents = list(_iter_tz_sentences(existing_dirs))
    print(f"[info] total sentences from TZ chunks: {len(all_sents)}")
    if not all_sents:
        raise SystemExit(
            "No TZ sentences collected. Check that package JSONs contain "
            "documents with doc_type='TZ'."
        )

    # Distribution of sentences across section kinds — this tells us whether
    # the chunks-format heading detection is actually catching anything.
    kind_counts = Counter(k for _, _, _, k in all_sents)
    print(f"[info] section-kind distribution: {dict(kind_counts)}")

    print(f"[info] classifying sentences (threshold={args.fuzzy_threshold})")
    matched, negatives, cls_stats = classify_rows(
        all_sents, csv_positives, args.fuzzy_threshold
    )
    print(f"[info] classification stats: {cls_stats}")
    print(f"[info] positives from req-sections matched to CSV: {len(matched)}")
    print(f"[info] negatives from non-req sections: {len(negatives)}")

    positives_recall = (
        len({_normalise_key(s) for _, _, s in matched}) / max(1, len(csv_positives))
    )
    print(f"[info] CSV-positive recall in req sections: {positives_recall:.1%}")

    rows = build_rows(
        csv_positives=csv_positives,
        matched_sents=matched,
        negatives=negatives,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    package_ids = [r["package_id"] for r in rows if r["package_id"] != "_csv_positives"]
    assignment = stratified_split_by_package(package_ids, seed=args.seed)
    splits = split_rows(rows, assignment)

    for name, items in splits.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, items)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _split_stats(items: List[dict]) -> dict:
        n = len(items)
        pos = sum(r["label"] for r in items)
        srcs = Counter(r["source"] for r in items)
        pkgs = Counter(r["package_id"] for r in items)
        word_counts = [len(r["text"].split()) for r in items]
        return {
            "count": n,
            "positives": pos,
            "negatives": n - pos,
            "pos_fraction": round(pos / n, 3) if n else None,
            "sources": dict(srcs),
            "unique_packages": len(pkgs),
            "word_count_p50": sorted(word_counts)[len(word_counts) // 2] if word_counts else None,
            "word_count_p95": sorted(word_counts)[int(len(word_counts) * 0.95)] if word_counts else None,
        }

    stats = {
        "matcher": _MATCHER_NAME,
        "csv_path": str(csv_path),
        "packages_dirs": [str(d) for d in existing_dirs],
        "fuzzy_threshold": args.fuzzy_threshold,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "csv_positives_unique": len(csv_positives),
        "sentences_from_packages": len(all_sents),
        "section_kind_distribution": dict(kind_counts),
        "classification_stats": cls_stats,
        "positives_in_req_sections": len(matched),
        "negatives_in_non_req_sections": len(negatives),
        "csv_positive_recall_in_req_sections": round(positives_recall, 4),
        "splits": {name: _split_stats(items) for name, items in splits.items()},
        "package_split_assignment": assignment,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Dataset stats ===")
    print(json.dumps(
        {k: v for k, v in stats.items() if k not in ("package_split_assignment",)},
        ensure_ascii=False, indent=2,
    ))
    print(f"\n[info] wrote {out_dir}/{{train,val,test}}.jsonl  +  stats.json")


if __name__ == "__main__":
    main()
