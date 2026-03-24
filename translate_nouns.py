#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required packages not installed. Install with:")
    print("  pip install requests python-dotenv")
    sys.exit(1)

load_dotenv()

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def get_api_key() -> str:
    """Read and validate the Google API key from the environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "Add it to your .env file:\n"
            "  GOOGLE_API_KEY=your_api_key_here\n\n"
            "Get a key at: https://console.cloud.google.com/apis/credentials"
        )
    return api_key


def get_max_id(path: Path) -> int:
    """Return the highest existing ID in a CSV, or 0 if the file doesn't exist / is empty."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            return 0
        ids = [int(row["id"]) for row in reader if row["id"].isdigit()]
    return max(ids, default=0)


def translate_batch(nouns: list[str], target: str, source: str, api_key: str) -> list[str]:
    """Send a batch of nouns to the Google Translate REST API and return translations."""
    response = requests.post(
        TRANSLATE_URL,
        params={"key": api_key},
        json={
            "q": nouns,
            "source": source,
            "target": target,
            "format": "text",
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Google Translate API error ({response.status_code}): {response.text}"
        )

    translations = response.json()["data"]["translations"]
    return [t["translatedText"] for t in translations]


def translate_nouns(
    input_path: Path,
    output_path: Path,
    target_language: str = "km",
    source_language: str = "en",
    batch_size: int = 50,
    delay: float = 0.1,
) -> int:
    """
    Read nouns from a CSV, translate them, and write to a new CSV.

    Args:
        input_path: Path to input CSV (must have 'id' and 'noun' columns)
        output_path: Path to output CSV
        target_language: Target language code (default: 'km' for Khmer)
        source_language: Source language code (default: 'en' for English)
        batch_size: Number of nouns to translate per API call (max 128)
        delay: Seconds to wait between API calls to avoid rate limits

    Returns:
        Number of nouns translated
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read all nouns from input CSV
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "noun" not in (reader.fieldnames or []):
            raise ValueError(f"Input CSV must have a 'noun' column. Found: {reader.fieldnames}")
        for row in reader:
            rows.append(row)

    if not rows:
        print("No nouns found in input file.")
        return 0

    # Check which nouns still need translating (support re-runs)
    existing_ids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id", "").isdigit():
                    existing_ids.add(row["id"])
        print(f"Found {len(existing_ids)} already-translated nouns in {output_path}, skipping those.")

    rows_to_translate = [r for r in rows if r["id"] not in existing_ids]

    if not rows_to_translate:
        print("All nouns already translated. Nothing to do.")
        return 0

    print(f"Translating {len(rows_to_translate)} nouns to '{target_language}'...")

    api_key = get_api_key()

    # Write header if output file is new
    write_header = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    translated_count = 0
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "noun", "translation", "language"])

        # Process in batches
        for i in range(0, len(rows_to_translate), batch_size):
            batch = rows_to_translate[i : i + batch_size]
            nouns = [row["noun"] for row in batch]

            translations = translate_batch(nouns, target_language, source_language, api_key)

            for row, translation in zip(batch, translations):
                writer.writerow([
                    row["id"],
                    row["noun"],
                    translation,
                    target_language,
                ])
                translated_count += 1

            progress = min(i + batch_size, len(rows_to_translate))
            print(f"  Translated {progress}/{len(rows_to_translate)}...")

            if i + batch_size < len(rows_to_translate):
                time.sleep(delay)

    return translated_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate nouns from a CSV file using Google Translate API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate generated_nouns.csv to Khmer (default)
  python translate_nouns.py

  # Specify input/output files
  python translate_nouns.py -i my_nouns.csv -o khmer_nouns.csv

  # Translate to a different language (e.g. French)
  python translate_nouns.py --target fr

Setup:
  1. Enable Cloud Translation API in your Google Cloud project
  2. Create an API key at: https://console.cloud.google.com/apis/credentials
  3. Add to .env file:
       GOOGLE_API_KEY=your_key_here
        """,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("generated_nouns.csv"),
        help="Input CSV file with nouns (default: generated_nouns.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("translated_nouns.csv"),
        help="Output CSV file (default: translated_nouns.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="km",
        help="Target language code (default: km for Khmer). "
             "See https://cloud.google.com/translate/docs/languages",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="en",
        help="Source language code (default: en for English)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Nouns per API call, max 128 (default: 50)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds between API calls to avoid rate limits (default: 0.1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_size < 1 or args.batch_size > 128:
        print("Error: --batch-size must be between 1 and 128", file=sys.stderr)
        return 2

    try:
        count = translate_nouns(
            input_path=args.input,
            output_path=args.output,
            target_language=args.target,
            source_language=args.source,
            batch_size=args.batch_size,
            delay=args.delay,
        )

        if count > 0:
            print(f"\n✓ Translated {count} nouns")
            print(f"✓ Saved to {args.output}")

            # Show a sample
            print("\nSample translations:")
            with args.output.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 5:
                        break
                    print(f"  {row['noun']:20s} → {row['translation']}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())