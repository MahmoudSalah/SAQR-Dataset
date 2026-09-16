#!/usr/bin/env python3
"""
Curate the Arabic Handwriting dataset from clean_crops/ into a standard flat format.

Input structure:
    clean_crops/
    ├── 1. boys/         (writer folders with paired GT/HW images)
    ├── 2. girls/
    ├── 3. undetrimined/
    └── collection/      (BOYS/, girls/, undtrimined/ — merged into main categories)

Output structure:
    clean_crops_curated/
    ├── gt/              (ground truth printed images)
    ├── hw/              (handwritten images, same ID = same pair)
    └── manifest.csv     (metadata for every pair)

Pairing logic:
    Within each leaf folder, sort images by numeric suffix.
    First half = GT (printed), second half = HW (handwritten), paired in order.
"""

import csv
import os
import re
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SRC_DIR = Path("/home/salah/Downloads/clean_crops")
DST_DIR = Path("/home/salah/Downloads/clean_crops_curated")

# Map raw category folder names → canonical category name
CATEGORY_MAP = {
    "1. boys": "boys",
    "2. girls": "girls",
    "3. undetrimined": "undetermined",
    # collection sub-categories merge into the same canonical names
    "BOYS": "boys",
    "girls": "girls",
    "undtrimined": "undetermined",
}


def numeric_suffix(filename: str) -> int:
    """Extract the last numeric part from a filename like 'Abdo_061-1-14.png' → 14."""
    m = re.search(r"-(\d+)\.png$", filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract numeric suffix from: {filename}")


def extract_writer(folder_name: str) -> str:
    """Extract writer name from folder like 'Abdo_061-1' → 'Abdo'."""
    return folder_name.split("_")[0]


def collect_leaf_folders(src: Path) -> list[tuple[Path, str]]:
    """
    Walk the source directory and return a sorted list of
    (leaf_folder_path, canonical_category) tuples.
    """
    results = []

    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue

        if entry.name == "collection":
            # collection has one extra nesting level: collection/BOYS/, collection/girls/, ...
            for sub_cat in sorted(entry.iterdir()):
                if not sub_cat.is_dir():
                    continue
                cat = CATEGORY_MAP.get(sub_cat.name, sub_cat.name)
                for leaf in sorted(sub_cat.iterdir()):
                    if leaf.is_dir():
                        results.append((leaf, cat))
        else:
            cat = CATEGORY_MAP.get(entry.name, entry.name)
            for leaf in sorted(entry.iterdir()):
                if leaf.is_dir():
                    results.append((leaf, cat))

    return results


def pair_images(folder: Path) -> list[tuple[str, str]]:
    """
    Given a leaf folder, return a list of (gt_filename, hw_filename) pairs.
    Images are sorted by numeric suffix; first half = GT, second half = HW.
    """
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(".png")],
        key=numeric_suffix,
    )

    n = len(files)
    if n == 0:
        return []
    if n % 2 != 0:
        print(f"  ⚠ WARNING: Odd number of images ({n}) in {folder}, skipping last image")
        n = n - 1
        files = files[:n]

    half = n // 2
    gt_files = files[:half]
    hw_files = files[half:]
    return list(zip(gt_files, hw_files))


def main():
    print("=" * 60)
    print("Arabic Handwriting Dataset Curation")
    print("=" * 60)
    print(f"Source : {SRC_DIR}")
    print(f"Output : {DST_DIR}")
    print()

    # Create output directories
    gt_dir = DST_DIR / "gt"
    hw_dir = DST_DIR / "hw"
    gt_dir.mkdir(parents=True, exist_ok=True)
    hw_dir.mkdir(parents=True, exist_ok=True)

    # Collect all leaf folders
    leaf_folders = collect_leaf_folders(SRC_DIR)
    print(f"Found {len(leaf_folders)} leaf folders to process")
    print()

    # Process and build manifest rows
    manifest_rows = []
    pair_id = 0
    stats = {"boys": 0, "girls": 0, "undetermined": 0}
    writer_stats: dict[str, int] = {}

    for folder_path, category in leaf_folders:
        source_folder = folder_path.name
        writer = extract_writer(source_folder)
        pairs = pair_images(folder_path)

        for gt_file, hw_file in pairs:
            pair_id += 1
            pid = f"{pair_id:06d}"

            gt_dst = f"gt/{pid}.png"
            hw_dst = f"hw/{pid}.png"

            # Copy files
            shutil.copy2(folder_path / gt_file, gt_dir / f"{pid}.png")
            shutil.copy2(folder_path / hw_file, hw_dir / f"{pid}.png")

            manifest_rows.append({
                "pair_id": pid,
                "gt_path": gt_dst,
                "hw_path": hw_dst,
                "writer": writer,
                "category": category,
                "source_folder": source_folder,
                "original_gt": gt_file,
                "original_hw": hw_file,
            })

            stats[category] = stats.get(category, 0) + 1
            writer_stats[writer] = writer_stats.get(writer, 0) + 1

    # Write manifest CSV
    manifest_path = DST_DIR / "manifest.csv"
    fieldnames = [
        "pair_id", "gt_path", "hw_path", "writer",
        "category", "source_folder", "original_gt", "original_hw",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    # Print summary
    print()
    print("=" * 60)
    print("✅ CURATION COMPLETE")
    print("=" * 60)
    print(f"Total pairs written : {pair_id}")
    print(f"GT images           : {len(list(gt_dir.glob('*.png')))}")
    print(f"HW images           : {len(list(hw_dir.glob('*.png')))}")
    print(f"Manifest rows       : {len(manifest_rows)}")
    print()
    print("By category:")
    for cat, count in sorted(stats.items()):
        print(f"  {cat:15s} : {count}")
    print()
    print("By writer:")
    for writer, count in sorted(writer_stats.items()):
        print(f"  {writer:15s} : {count}")
    print()
    print(f"Output directory    : {DST_DIR}")
    print(f"Manifest            : {manifest_path}")


if __name__ == "__main__":
    main()
