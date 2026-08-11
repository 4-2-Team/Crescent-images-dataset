#!/usr/bin/env python3

import csv
import io
import time
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def download_image(url, timeout=30, retries=3):
    """
    Download an image and return its bytes.
    Retries automatically if the request fails.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)

            with urlopen(request, timeout=timeout) as response:
                return response.read()

        except (HTTPError, URLError, TimeoutError) as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")

            if attempt < retries:
                time.sleep(2)

    return None


def main():

    parser = argparse.ArgumentParser(
        description="Download crescent Moon images from NASA URLs stored in CSV."
    )

    parser.add_argument(
        "--csv",
        default="crescent_moon_training_links.csv",
        help="CSV file containing Moon image URLs",
    )

    parser.add_argument(
        "--out",
        default="crescent_moons",
        help="Output directory",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to download. 0 = all images",
    )

    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Apply suggested_rotation_deg from CSV",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Delay between downloads in seconds",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip this many rows before downloading",
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Check CSV
    # -------------------------------------------------------

    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"\nERROR: CSV file not found:")
        print(csv_path.resolve())
        print(
            "\nPlace crescent_moon_training_links.csv "
            "in the same folder as this script,"
        )
        print("or specify it with:")
        print(
            "python3 download_crescent_moons.py "
            "--csv /path/to/file.csv"
        )
        return

    # -------------------------------------------------------
    # Import Pillow only if rotation is requested
    # -------------------------------------------------------

    Image = None

    if args.rotate:
        try:
            from PIL import Image as PILImage
            Image = PILImage
        except ImportError:
            print("\nPillow is required when using --rotate.")
            print("Install it with:")
            print("\npip3 install pillow\n")
            return

    # -------------------------------------------------------
    # Read CSV
    # -------------------------------------------------------

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV contains no image rows.")
        return

    # Check required columns

    required_columns = {
        "frame",
        "crescent_type",
        "suggested_rotation_deg",
        "image_url",
    }

    available_columns = set(rows[0].keys())

    missing_columns = required_columns - available_columns

    if missing_columns:
        print(
            "CSV is missing required columns:",
            ", ".join(sorted(missing_columns)),
        )
        return

    # -------------------------------------------------------
    # Apply start / limit
    # -------------------------------------------------------

    if args.start > 0:
        rows = rows[args.start:]

    if args.limit > 0:
        rows = rows[:args.limit]

    total = len(rows)

    # -------------------------------------------------------
    # Create output folder
    # -------------------------------------------------------

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("Crescent Moon Dataset Downloader")
    print("=" * 60)
    print(f"CSV:       {csv_path}")
    print(f"Output:    {output_dir.resolve()}")
    print(f"Images:    {total}")
    print(f"Rotation:  {'enabled' if args.rotate else 'disabled'}")
    print("=" * 60)
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    # -------------------------------------------------------
    # Download
    # -------------------------------------------------------

    for index, row in enumerate(rows, start=1):

        try:
            frame = int(row["frame"])

        except (ValueError, TypeError):
            print(
                f"[{index}/{total}] Invalid frame value: "
                f"{row.get('frame')}"
            )
            failed += 1
            continue

        crescent_type = (
            row["crescent_type"]
            .strip()
            .lower()
            .replace(" ", "_")
        )

        try:
            angle = int(float(row["suggested_rotation_deg"]))

        except (ValueError, TypeError):
            angle = 0

        image_url = row["image_url"].strip()

        # ---------------------------------------------------
        # Folder structure
        #
        # crescent_moons/
        #     waxing_crescent/
        #     waning_crescent/
        # ---------------------------------------------------

        folder = output_dir / crescent_type
        folder.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------
        # Correct filename construction
        # ---------------------------------------------------

        if args.rotate:
            filename = (
                f"moon_{frame:04d}_rot{angle:03d}.jpg"
            )
        else:
            filename = (
                f"moon_{frame:04d}.jpg"
            )

        dest = folder / filename

        # ---------------------------------------------------
        # Skip existing files
        # ---------------------------------------------------

        if dest.exists():

            print(
                f"[{index}/{total}] SKIP "
                f"{dest}"
            )

            skipped += 1
            continue

        print(
            f"[{index}/{total}] Downloading "
            f"{crescent_type} frame {frame}"
        )

        # ---------------------------------------------------
        # Download image
        # ---------------------------------------------------

        data = download_image(image_url)

        if data is None:

            print(
                f"  FAILED: {image_url}"
            )

            failed += 1
            continue

        # ---------------------------------------------------
        # Save image
        # ---------------------------------------------------

        try:

            if args.rotate:

                image = Image.open(
                    io.BytesIO(data)
                ).convert("RGB")

                rotated = image.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(0, 0, 0),
                )

                rotated.save(
                    dest,
                    format="JPEG",
                    quality=95,
                )

            else:

                dest.write_bytes(data)

            downloaded += 1

            print(
                f"  Saved: {dest}"
            )

        except Exception as e:

            print(
                f"  ERROR saving image: {e}"
            )

            failed += 1

        # Avoid hammering the server

        if args.sleep > 0:
            time.sleep(args.sleep)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    print(f"Total:      {total}")
    print()
    print(f"Images saved in:")
    print(output_dir.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()