"""
================================================================================
SCRIPT: split_dataset.py
PROJECT: SpartanSat Low-Light Enhancement Model (IllumiCurveNet)
================================================================================
DESCRIPTION:
This script performs a stratified, disjoint dataset split on the Crescent-Moon
dataset (waning_crescent and waxing_crescent).

SPLIT SPECIFICATION (80% Train / 10% Val / 10% Test):
- Train Set:      80% total (40% waning_crescent + 40% waxing_crescent)
- Validation Set: 10% total ( 5% waning_crescent +  5% waxing_crescent)
- Test Set:       10% total ( 5% waning_crescent +  5% waxing_crescent)

DATA INTEGRITY & OVERLAP PREVENTION:
1. Disjoint Partitioning: Each unique image file is allocated to exactly ONE split.
   Once assigned to train, it is removed from the selection pool and cannot be
   assigned to val or test.
2. Verification Step: Automated post-split assertion guarantees 0 overlap across
   Train, Validation, and Test sets.
3. Reproducibility: Uses a fixed random seed (default: 42) for deterministic splits.
4. Audit Manifest: Outputs a complete CSV log ('dataset_split_manifest.csv')
   mapping each file to its assigned split and destination path.
================================================================================
"""

import os
import shutil
import random
import csv
from pathlib import Path

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA_DIR = BASE_DIR / "crescent_moons"
OUTPUT_SPLITS_DIR = BASE_DIR / "data_splits"
MANIFEST_PATH = BASE_DIR / "dataset_split_manifest.csv"

# Classes / Subfolders
CLASSES = ["waning_crescent", "waxing_crescent"]

# Split ratios within each class (80% train, 10% val, 10% test)
# Since both classes have equal counts, this results in:
# - Train: 40% waning + 40% waxing = 80% of total
# - Val:    5% waning +  5% waxing = 10% of total
# - Test:   5% waning +  5% waxing = 10% of total
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Fixed seed for complete reproducibility
RANDOM_SEED = 42

# Set to True to copy files, or False to simulate without copying
EXECUTE_COPY = True

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_image_files(class_dir: Path) -> list:
    """Scan and collect all valid image file paths in the class directory."""
    if not class_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {class_dir}")
    
    files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    # Sort files initially for deterministic ordering prior to shuffling
    files.sort(key=lambda x: x.name)
    return files


def partition_dataset(files: list, train_ratio: float, val_ratio: float) -> tuple:
    """
    Partitions a list of files into disjoint (Train, Val, Test) subsets.
    
    Guarantees:
    - Sets are mutually exclusive (zero intersection).
    - Union of sets equals the original file list.
    """
    total_count = len(files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    # The remainder goes to test to ensure every single image is accounted for
    test_count = total_count - train_count - val_count

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    return train_files, val_files, test_files


# ==============================================================================
# 3. MAIN SPLIT PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print("STARTING STRATIFIED DATASET SPLIT PIPELINE")
    print("=" * 80)
    print(f"Source Directory:      {SOURCE_DATA_DIR}")
    print(f"Output Directory:      {OUTPUT_SPLITS_DIR}")
    print(f"Random Seed:           {RANDOM_SEED}")
    print(f"Ratios (Train/Val/Test): {TRAIN_RATIO * 100:.0f}% / {VAL_RATIO * 100:.0f}% / {TEST_RATIO * 100:.0f}%")
    print("-" * 80)

    # Set random seed for deterministic shuffle
    random.seed(RANDOM_SEED)

    # Track all assignments for manifest and overlap verification
    manifest_records = []
    split_sets = {"train": set(), "val": set(), "test": set()}
    summary_counts = {
        "train": {"waning_crescent": 0, "waxing_crescent": 0},
        "val":   {"waning_crescent": 0, "waxing_crescent": 0},
        "test":  {"waning_crescent": 0, "waxing_crescent": 0},
    }

    # Process each class individually (Stratified Sampling)
    for class_name in CLASSES:
        class_src_dir = SOURCE_DATA_DIR / class_name
        image_files = get_image_files(class_src_dir)
        print(f"\n[CLASS: {class_name}] Found {len(image_files)} images.")

        # Shuffle deterministically with fixed seed
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)

        # Disjoint partitioning into Train, Val, Test
        train_list, val_list, test_list = partition_dataset(
            shuffled_files, TRAIN_RATIO, VAL_RATIO
        )

        splits_map = {
            "train": train_list,
            "val": val_list,
            "test": test_list
        }

        # Process each split
        for split_name, file_list in splits_map.items():
            dest_dir = OUTPUT_SPLITS_DIR / split_name / class_name
            if EXECUTE_COPY:
                dest_dir.mkdir(parents=True, exist_ok=True)

            summary_counts[split_name][class_name] = len(file_list)

            for src_file in file_list:
                dest_file = dest_dir / src_file.name

                # Add unique relative path to set for strict uniqueness check
                unique_key = f"{class_name}/{src_file.name}"
                split_sets[split_name].add(unique_key)

                # Record manifest entry
                manifest_records.append({
                    "filename": src_file.name,
                    "class_name": class_name,
                    "split": split_name,
                    "source_path": str(src_file),
                    "destination_path": str(dest_file)
                })

                # Copy file to structured destination
                if EXECUTE_COPY:
                    shutil.copy2(src_file, dest_file)

    # ==========================================================================
    # 4. VERIFICATION & OVERLAP VALIDATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("VERIFYING SPLIT INTEGRITY & ZERO OVERLAP")
    print("=" * 80)

    train_set = split_sets["train"]
    val_set = split_sets["val"]
    test_set = split_sets["test"]

    # Compute set intersections to confirm no duplicate files across splits
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    print(f"Overlap between Train and Val:   {len(overlap_train_val)} (Expected: 0)")
    print(f"Overlap between Train and Test:  {len(overlap_train_test)} (Expected: 0)")
    print(f"Overlap between Val and Test:    {len(overlap_val_test)} (Expected: 0)")

    assert len(overlap_train_val) == 0, f"Error: Overlap found between Train and Val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Error: Overlap found between Train and Test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Error: Overlap found between Val and Test: {overlap_val_test}"

    total_assigned = len(train_set) + len(val_set) + len(test_set)
    print(f"Total Unique Images Processed:   {total_assigned}")
    print("[SUCCESS] All splits are strictly disjoint. Zero data leakage detected.")

    # ==========================================================================
    # 5. WRITE MANIFEST FILE
    # ==========================================================================
    with open(MANIFEST_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "class_name", "split", "source_path", "destination_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"\n[MANIFEST] Saved audit manifest to: {MANIFEST_PATH}")

    # ==========================================================================
    # 6. SUMMARY REPORT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL DATASET SPLIT BREAKDOWN")
    print("=" * 80)
    print(f"{'Split':<12} | {'Waning Crescent':<16} | {'Waxing Crescent':<16} | {'Total':<10} | {'Percentage'}")
    print("-" * 80)

    for split_name in ["train", "val", "test"]:
        waning_n = summary_counts[split_name]["waning_crescent"]
        waxing_n = summary_counts[split_name]["waxing_crescent"]
        subtotal = waning_n + waxing_n
        pct = (subtotal / total_assigned) * 100
        print(f"{split_name.upper():<12} | {waning_n:<16} | {waxing_n:<16} | {subtotal:<10} | {pct:.1f}%")

    print("-" * 80)
    total_waning = sum(summary_counts[s]["waning_crescent"] for s in summary_counts)
    total_waxing = sum(summary_counts[s]["waxing_crescent"] for s in summary_counts)
    print(f"{'TOTAL':<12} | {total_waning:<16} | {total_waxing:<16} | {total_assigned:<10} | 100.0%")
    print("=" * 80)
    print("[DONE] Dataset successfully split and verified.")


if __name__ == "__main__":
    main()
