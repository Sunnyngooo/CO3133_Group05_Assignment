"""Audit local N24News metadata without downloading images or training models."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(path):
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of N24News records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("All records must be JSON objects")

    fields = [
        "section", "headline", "abstract", "caption", "image_url",
        "article_url", "image_id",
    ]

    def value(record, key):
        return str(record.get(key) or "").strip()

    def duplicate_counts(key):
        counts = Counter(value(record, key) for record in records if value(record, key))
        return {
            "repeated_values": sum(count > 1 for count in counts.values()),
            "extra_records": sum(count - 1 for count in counts.values()),
        }

    lengths = sorted(len(value(record, "headline").split()) for record in records)
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_bytes": len(raw),
        "records": len(records),
        "fields": fields,
        "class_counts": dict(sorted(Counter(value(r, "section") for r in records).items())),
        "missing_fields": {
            key: sum(not value(record, key) for record in records) for key in fields
        },
        "duplicates": {
            key: duplicate_counts(key)
            for key in ["image_id", "image_url", "article_url", "headline"]
        },
        "headline_words": {
            "min": lengths[0],
            "median": (lengths[(len(lengths) - 1) // 2] + lengths[len(lengths) // 2]) / 2,
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "max": lengths[-1],
            "method": "whitespace word counts; p95 uses a sorted-index estimate",
        },
        "image_access_and_pair_quality": "not_verified_by_this_metadata_audit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/assignment3/raw/nytimes_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/assignment3/metadata_audit.json"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--retrieved-date", required=True)
    args = parser.parse_args()
    result = audit(args.metadata)
    result.update({
        "source_revision": args.source_revision,
        "source_url": f"https://raw.githubusercontent.com/billywzh717/N24News/{args.source_revision}/nytimes_dataset.json",
        "retrieved_date": args.retrieved_date,
        "license_status": "not_explicitly_stated_in_author_repository",
        "instructor_approval": "not_recorded",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {result['records']} records across {len(result['class_counts'])} classes")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
