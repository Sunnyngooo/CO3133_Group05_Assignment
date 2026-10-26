"""Build audited, conservatively grouped N24News manifests and EDA artifacts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from data_utils import ROOT, UnionFind, canonical_url, selection_digest, write_json


def perceptual_features(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (gray[:, 1:] > gray[:, :-1]).flatten()
        code = sum(int(bit) << index for index, bit in enumerate(bits))
        thumb = np.asarray(image.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32)
        return code, thumb, float(gray.std()), image.width / image.height


def perceptual_feature_stream(rows, workers):
    # map yields in input order, so worker completion order cannot change groups.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(perceptual_features, (ROOT / row["image_path"] for row in rows))


def build_groups(rows, workers=4):
    union = UnionFind(len(rows))
    seen = {}
    for index, row in enumerate(rows):
        keys = [
            ("metadata_group", row["metadata_group"]),
            ("pixels", row["pixel_sha256"]),
            ("file", row["file_sha256"]),
            ("headline", row["normalized_headline"]),
            ("image_url", canonical_url(row["image_url"])),
            ("article_url", canonical_url(row["article_url"])),
        ]
        for key in keys:
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index

    # Conservative similarity groups, not claims that the news event is identical.
    by_hash = defaultdict(list)
    features = []
    near_pairs = []
    for index, (row, feature) in enumerate(zip(rows, perceptual_feature_stream(rows, workers))):
        code, thumbnail, variation, aspect = feature
        if variation >= 10:
            for candidate in [code] + [code ^ (1 << bit) for bit in range(64)]:
                for previous in by_hash[candidate]:
                    _, previous_thumb, previous_variation, previous_aspect = features[previous]
                    if previous_variation < 10 or abs(aspect - previous_aspect) > 0.05:
                        continue
                    rmse = float(np.sqrt(np.mean((thumbnail - previous_thumb) ** 2)))
                    if rmse <= 5:
                        union.union(index, previous)
                        near_pairs.append({
                            "left_id": rows[previous]["image_id"], "right_id": row["image_id"],
                            "dhash_distance": (code ^ features[previous][0]).bit_count(),
                            "thumbnail_rgb_rmse": round(rmse, 4),
                            "same_label": rows[previous]["label"] == row["label"],
                        })
            by_hash[code].append(index)
        features.append(feature)
        if (index + 1) % 1000 == 0:
            print(f"Image similarity audit {index + 1}/{len(rows)}", flush=True)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[union.find(index)].append(row)
    groups = []
    for members in grouped.values():
        group_id = "group_" + min(row["image_id"] for row in members)
        for row in members:
            row["group_id"] = group_id
        groups.append(members)
    return sorted(groups, key=lambda members: members[0]["group_id"]), near_pairs


def assign_splits(groups, seed, ratios=(0.8, 0.1, 0.1)):
    """Greedily minimize per-class deviation while keeping each whole group intact."""
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    totals = Counter(row["label"] for group in groups for row in group)
    names = ("train", "validation", "test")
    assigned = {name: Counter() for name in names}
    output = {name: [] for name in names}
    for members in groups:
        addition = Counter(row["label"] for row in members)
        costs = []
        for name, ratio in zip(names, ratios):
            cost = sum(
                ((assigned[name][label] + number - totals[label] * ratio) ** 2
                 - (assigned[name][label] - totals[label] * ratio) ** 2)
                / max(totals[label] * ratio, 1)
                for label, number in addition.items()
            )
            costs.append(cost)
        target = names[min(range(3), key=lambda index: (costs[index], index))]
        assigned[target].update(addition)
        for row in members:
            row["split"] = target
            output[target].append(row)
    for rows in output.values():
        rows.sort(key=lambda row: row["image_id"])
    validate_splits(output, set(totals))
    return output


def validate_splits(splits, labels):
    seen_ids = set()
    seen_groups = set()
    seen_keys = {field: set() for field in ("pixel_sha256", "file_sha256", "canonical_image_url", "canonical_article_url", "normalized_headline")}
    for name, rows in splits.items():
        ids = [row["image_id"] for row in rows]
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError(f"Sample overlap in {name}")
        groups = {row["group_id"] for row in rows}
        if seen_groups.intersection(groups):
            raise ValueError(f"Group overlap in {name}")
        if {row["label"] for row in rows} != labels:
            raise ValueError(f"Some classes are absent from {name}")
        for field, previous in seen_keys.items():
            values = {row[field] for row in rows}
            if previous.intersection(values):
                raise ValueError(f"{field} overlap in {name}")
            previous.update(values)
        seen_ids.update(ids)
        seen_groups.update(groups)


def write_manifests(directory, splits):
    hashes = {}
    for name, rows in splits.items():
        path = directory / (name + ".jsonl")
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        encoded = content.encode("utf-8")
        path.write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    return hashes


def plot_eda(output, rows, metadata_counts, splits):
    labels = sorted(metadata_counts)
    selected_counts = Counter(row["label"] for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), constrained_layout=True)
    positions = np.arange(len(labels))
    for axis, values, title in [
        (axes[0], metadata_counts, "Full metadata (before image verification)"),
        (axes[1], selected_counts, "Prepared pairs (verified image files)"),
    ]:
        axis.barh(positions, [values[label] for label in labels], color="#3175a8")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Records")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.savefig(output / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    lengths = [len(row["headline"].split()) for row in rows]
    axes[0].hist(lengths, bins=range(min(lengths), max(lengths) + 2), color="#3175a8")
    axes[0].set(xlabel="Headline whitespace words", ylabel="Records", title="Headline length")
    axes[1].scatter([r["width"] for r in rows], [r["height"] for r in rows], s=3, alpha=0.15)
    axes[1].set(xlabel="Width (pixels)", ylabel="Height (pixels)", title="Image dimensions")
    names = list(splits)
    axes[2].bar(names, [len(splits[name]) for name in names], color=["#3175a8", "#e7a23b", "#4b9d72"])
    axes[2].set(ylabel="Records", title="Group-aware split sizes")
    fig.savefig(output / "input_and_split_stats.png", dpi=150)
    plt.close(fig)


def make_review_gallery(output, rows, per_class, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    chosen = []
    for label in sorted(by_class):
        candidates = sorted(by_class[label], key=lambda row: row["image_id"])
        chosen.extend(rng.sample(candidates, min(per_class, len(candidates))))
    review_path = output / "pair_review.csv"
    previous_review = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8", newline="") as stream:
            previous_review = {row["image_id"]: row for row in csv.DictReader(stream)}
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "label", "headline", "image_path", "relation", "reviewer", "notes"])
        writer.writeheader()
        for row in chosen:
            value = {key: row[key] for key in ("image_id", "label", "headline", "image_path")}
            value.update({key: previous_review.get(row["image_id"], {}).get(key, "")
                          for key in ("relation", "reviewer", "notes")})
            writer.writerow(value)
    gallery = output / "review_gallery"
    gallery.mkdir(exist_ok=True)
    for start in range(0, len(chosen), 8):
        fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
        for offset, axis in enumerate(axes.flat):
            index = start + offset
            axis.axis("off")
            if index >= len(chosen):
                continue
            row = chosen[index]
            with Image.open(ROOT / row["image_path"]) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            heading = f"{index + 1}. {row['label']} | {row['image_id'][:8]}"
            axis.set_title(heading + "\n" + textwrap.fill(row["headline"], 65), fontsize=10)
        fig.savefig(gallery / f"pairs_{start // 8 + 1:02d}.jpg", dpi=110)
        plt.close(fig)
    return chosen


def write_eda_report(output, summary, download_summary, rows):
    counts = Counter(row["label"] for row in rows)
    labels = sorted(counts)
    class_table = "\n".join(
        "| " + label + " | " + str(counts[label]) + " | "
        + " | ".join(str(summary["split_class_counts"][name].get(label, 0))
                     for name in ("train", "validation", "test")) + " |"
        for label in labels
    )
    exclusions = summary["excluded"] or {"none": 0}
    if summary["selection_scope"] == "full_metadata":
        scope_note = "All metadata records were selected without a per-class cap. Only failed images or empty headlines are excluded. Class counts retain the observed corpus imbalance; the group-aware split is custom and is not the authors' original split."
    else:
        scope_note = "A capped subset was selected. It can change the natural class prior; results describe this custom protocol, not the full benchmark or original paper split."
    report = f"""# N24News: data preparation and EDA

This report is generated from the downloaded/audited data. It contains no model results.

## Scope and provenance

- Full metadata: {summary['metadata_records']:,} records; selected records: {summary['selected_records']:,}.
- Verified, complete image/headline pairs: **{summary['prepared_pairs']:,}**, across **{summary['classes']} classes**.
- Image source variant: `{summary['image_source']}`; query-only rendering parameters removed from the same public asset path.
- Selection seed: {summary['seed']}; selected-ID SHA-256: `{summary['selection_sha256']}`.
- Download statuses: `{download_summary['status_counts']}`; valid file bytes: {download_summary['total_valid_file_bytes']:,}.
- Wall time of this downloader invocation: {download_summary['elapsed_seconds']:.1f} seconds. If resumed, this is not the total wall time across all invocations.
- Cleaning exclusions: `{dict(exclusions)}`.
- {scope_note}

## Distribution and input sizes

![Class distribution](class_distribution.png)

![Headline/image/split statistics](input_and_split_stats.png)

Headline length is measured in whitespace words, not DistilBERT tokens. Token-length and truncation checks remain part of the post-approval encoder setup.

| Class | Prepared pairs | Train | Validation | Test |
|---|---:|---:|---:|---:|
{class_table}

## Duplicate grouping and split checks

- Final groups: {summary['groups']:,}; groups with multiple records: {summary['multi_record_groups']}; largest group: {summary['largest_group']}.
- Strict perceptual-similarity links: {summary['near_duplicate_links']}.
- Groups containing multiple editorial labels: {summary['mixed_label_groups']} ({summary['mixed_label_group_records']} records). Different headlines can legitimately share a photo; these groups are retained together, not automatically called annotation errors.
- Exact repeated headline extra records in the prepared data: {summary['headline_exact_duplicate_extra_records']}. Matching normalized headlines are grouped for text-overlap control, not interpreted as event identity.
- **Leakage verification:** {summary['leakage_checks']}.
- Rules: {summary['group_rules']}.
- Limitation: {summary['limitation']}

Split hashes:

```json
{json.dumps(summary['split_sha256'], indent=2)}
```

The test manifest is fixed here but has not been evaluated by any model. Use train/validation to select models; test evaluation happens after the experimental design is locked.

## Pair quality and remaining review

A separate AI visual review examined 120 pilot pairs (5 per class) before this run. The review used metadata-rendered images. It recorded 81 pairs with clear topic cues, 37 weak/contextual pairs, and 2 uncertain pairs; this is not a population quality estimate or an independent human audit. The records and limitations are in [pilot review](../pilot_seed44/pair_review.csv) and [pilot summary](../pilot_seed44/pair_review_summary.json). This run's gallery samples are selected independently and must be reviewed separately.

The pilot showed generic portraits, overlapping editorial topics, reused photos and many Published-year overlays. Original-asset routing was checked and recorded to avoid query-rendered date banners. The current-variant gallery and blank human review form are provided in `review_gallery/` and `pair_review.csv`; rerunning preparation preserves existing review notes for matching IDs.

Dataset usage conditions and instructor approval are not yet recorded. Group Name is G-05. Review [the proposal](../../../assignment3/PROPOSAL.md) and fill in the responsible member and approval record before main experiments.
"""
    (output / "EDA.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_seed44_original")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--review-per-class", type=int, default=5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name) or args.review_per_class < 1:
        parser.error("Invalid run-name or review-per-class")
    source = ROOT / "data/assignment3/processed" / args.run_name
    output = ROOT / "outputs/assignment3" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    download_summary = json.loads((source / "download_summary.json").read_text(encoding="utf-8"))
    with (source / "image_audit.csv").open(encoding="utf-8", newline="") as stream:
        image_audit = {row["image_id"]: row for row in csv.DictReader(stream)}
    if set(image_audit) != {row["image_id"] for row in selected} or download_summary["selection_sha256"] != selection_digest(selected):
        raise ValueError("Download audit is incomplete or does not match the selection")
    labels = sorted({row["section"] for row in selected})
    label_to_id = {label: index for index, label in enumerate(labels)}
    prepared = []
    exclusions = Counter()
    for row in selected:
        audit = image_audit[row["image_id"]]
        if audit["status"] == "failed":
            exclusions["download_failed"] += 1
            continue
        if not row.get("headline", "").strip():
            exclusions["empty_headline"] += 1
            continue
        if not (ROOT / audit["path"]).is_file():
            raise ValueError(f"Audited image no longer exists: {row['image_id']}")
        prepared.append({
            "sample_id": row["image_id"], "image_id": row["image_id"],
            "image_path": Path(audit["path"]).as_posix(),
            "image_source": audit.get("image_source", "metadata"),
            "headline": row["headline"].strip(), "abstract": row.get("abstract", ""),
            "normalized_headline": " ".join(row["headline"].split()).casefold(),
            "caption": row.get("caption", ""), "label": row["section"], "label_id": label_to_id[row["section"]],
            "image_url": row["image_url"], "article_url": row["article_url"],
            "canonical_image_url": canonical_url(row["image_url"]),
            "canonical_article_url": canonical_url(row["article_url"]),
            "metadata_group": row["metadata_group"],
            "pixel_sha256": audit["pixel_sha256"], "file_sha256": audit["file_sha256"],
            "width": int(audit["width"]), "height": int(audit["height"]), "file_bytes": int(audit["bytes"]),
        })
    if len(prepared) < 5000 or len(labels) < 5:
        raise ValueError("Prepared scope does not meet the default classification threshold")
    missing_classes = set(labels) - {row["label"] for row in prepared}
    if missing_classes:
        raise ValueError(f"No valid pairs remain for selected classes: {sorted(missing_classes)}")
    groups, near_pairs = build_groups(prepared)
    splits = assign_splits(groups, args.seed)
    hashes = write_manifests(source, splits)
    write_json(source / "label_mapping.json", label_to_id)
    write_json(output / "near_duplicate_pairs.json", near_pairs)
    metadata = json.loads((ROOT / "data/assignment3/raw/nytimes_dataset.json").read_text(encoding="utf-8"))
    full_selection = {row["image_id"] for row in selected} == {row["image_id"] for row in metadata}
    metadata_counts = dict(Counter(row["section"] for row in metadata))
    plot_eda(output, prepared, metadata_counts, splits)
    review = make_review_gallery(output, prepared, args.review_per_class, args.seed)
    conflicts = [members for members in groups if len({row["label"] for row in members}) > 1]
    summary = {
        "run_name": args.run_name, "metadata_records": len(metadata), "selected_records": len(selected),
        "selection_scope": "full_metadata" if full_selection else "capped_subset",
        "per_class_cap": download_summary["per_class_cap"],
        "image_source": download_summary.get("image_source", "metadata"),
        "prepared_pairs": len(prepared), "classes": len(labels), "excluded": dict(exclusions),
        "seed": args.seed, "selection_sha256": selection_digest(selected), "split_sha256": hashes,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_class_counts": {name: dict(sorted(Counter(row["label"] for row in rows).items())) for name, rows in splits.items()},
        "groups": len(groups), "largest_group": max(map(len, groups)),
        "multi_record_groups": sum(len(group) > 1 for group in groups),
        "mixed_label_groups": len(conflicts), "mixed_label_group_records": sum(map(len, conflicts)),
        "near_duplicate_links": len(near_pairs),
        "headline_exact_duplicate_extra_records": sum(count - 1 for count in Counter(row["headline"] for row in prepared).values()),
        "review_sample_size": len(review), "review_status": "pending_visual_review",
        "leakage_checks": "Passed: sample IDs, group IDs, pixel/file hashes, canonical image/article URLs and normalized exact headlines disjoint across splits",
        "group_rules": "Full-metadata article/image canonical URL components; selected pixel/file hashes; normalized exact headlines grouped to prevent text memorization; strict perceptual similarity (dHash<=1, RGB thumbnail RMSE<=5, aspect delta<=0.05; flat images excluded)",
        "limitation": "No complete event-level IDs; broader semantic event overlap remains unverified. Identical headlines are grouped conservatively for text-overlap control, not claimed to be the same event.",
        "instructor_approval": "not_recorded",
        "pilot_semantic_review": "outputs/assignment3/pilot_seed44/pair_review_summary.json (AI visual review of 120 selected IDs, metadata rendering variant; human review pending)",
    }
    write_json(output / "eda_summary.json", summary)
    write_eda_report(output, summary, download_summary, prepared)
    print(json.dumps({key: summary[key] for key in ["prepared_pairs", "classes", "split_counts", "groups", "near_duplicate_links", "mixed_label_groups", "leakage_checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
