#!/usr/bin/env python3
"""
Generate Writer-Independent Train/Val/Test Splits for the Curated Dataset.

This script reads the main manifest (manifest.csv), identifies all unique
writers (`student_id`), and allocates them into Train (70%), Validation (15%),
and Test (15%) splits.

The resulting split assignment is added as a new `split` column to the manifest,
and the updated manifest is saved back to disk. Writer independence ensures
that handwriting styles seen in the test set were completely unseen during training.
"""

import csv
import random
from pathlib import Path

# Paths
CURATED_DIR = Path("/home/salah/Downloads/clean_crops_curated")
MANIFEST_PATH = CURATED_DIR / "manifest.csv"

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO implicitly the remaining 0.15

# Set seed for reproducibility
random.seed(42)

def main():
    print(f"Reading manifest from {MANIFEST_PATH}")
    
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "split" not in fieldnames:
        fieldnames.append("split")

    # Group pairs by student_id to ensure writer independence
    students = {} # student_id -> list of row indices
    
    for idx, row in enumerate(rows):
        sid = row["student_id"]
        # Skip pairs where text extraction failed or student wasn't identified properly
        if not sid or sid == "UNKNOWN" or row["text"] == "":
            row["split"] = "exclude"
            continue
            
        if sid not in students:
            students[sid] = []
        students[sid].append(idx)

    # Sort student IDs to ensure deterministic splitting before shuffle
    unique_students = sorted(list(students.keys()))
    print(f"Found {len(unique_students)} valid distinct student writers.")
    
    # Shuffle students deterministically due to seed
    random.shuffle(unique_students)

    # Calculate split sizes (by number of students)
    num_students = len(unique_students)
    train_end = int(num_students * TRAIN_RATIO)
    val_end = train_end + int(num_students * VAL_RATIO)

    train_students = set(unique_students[:train_end])
    val_students = set(unique_students[train_end:val_end])
    test_students = set(unique_students[val_end:])

    # Assign splits back to the rows
    train_count = 0
    val_count = 0
    test_count = 0
    excluded_count = 0

    for idx, row in enumerate(rows):
        if row.get("split") == "exclude":
            excluded_count += 1
            continue
            
        sid = row["student_id"]
        if sid in train_students:
            row["split"] = "train"
            train_count += 1
        elif sid in val_students:
            row["split"] = "val"
            val_count += 1
        else:
            row["split"] = "test"
            test_count += 1

    total_valid = train_count + val_count + test_count
    
    # Save the updated manifest
    print("\nSaving updated manifest with splits...")
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print("\n" + "="*40)
    print("DATASET SPLIT SUMMARY (Line-level pairs)")
    print("="*40)
    print(f"Total Writers: {len(unique_students)}")
    print(f"  Train writers: {len(train_students)}")
    print(f"  Val writers:   {len(val_students)}")
    print(f"  Test writers:  {len(test_students)}")
    print("-" * 40)
    print(f"Total Paired Images: {len(rows)}")
    print(f"  Train pairs:   {train_count} ({train_count/total_valid*100:.1f}%)")
    print(f"  Val pairs:     {val_count} ({val_count/total_valid*100:.1f}%)")
    print(f"  Test pairs:    {test_count} ({test_count/total_valid*100:.1f}%)")
    if excluded_count > 0:
        print(f"  Excluded:      {excluded_count} (missing text or student ID)")
    print("="*40)
    print("Done!")

if __name__ == "__main__":
    main()
