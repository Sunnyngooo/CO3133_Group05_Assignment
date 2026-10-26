"""Deterministic grouping and selection for the N24News data preparation stage."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def canonical_url(url):
    parsed = urlsplit(str(url).strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))


def load_records(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Expected a JSON list of metadata objects")
    ids = [r.get("image_id") for r in records]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("N24News image_id values must be present and unique")
    return records


def metadata_groups(records):
    """Group all metadata before selecting a subset, including transitive links."""
    union = UnionFind(len(records))
    seen = {}
    for index, record in enumerate(records):
        for field in ("article_url", "image_url"):
            url = canonical_url(record.get(field, ""))
            if not url:
                continue
            key = (field, url)
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union.find(index)].append(record)
    groups = []
    for rows in grouped.values():
        group_id = "meta_" + min(row["image_id"] for row in rows)
        groups.append((group_id, rows))
    return sorted(groups, key=lambda item: item[0])


def select_records(records, per_class, seed):
    groups = metadata_groups(records)
    rng = random.Random(seed)
    rng.shuffle(groups)
    counts = Counter()
    selected = []
    for group_id, rows in groups:
        additions = Counter(row["section"] for row in rows)
        if per_class and any(counts[label] + number > per_class for label, number in additions.items()):
            continue
        for row in rows:
            selected.append({**row, "metadata_group": group_id})
        counts.update(additions)
    selected.sort(key=lambda row: row["image_id"])
    return selected


def selection_digest(records):
    return hashlib.sha256("\n".join(sorted(r["image_id"] for r in records)).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
