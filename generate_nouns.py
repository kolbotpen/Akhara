#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: Required packages not installed. Install with: pip install requests")
    sys.exit(1)

OLLAMA_BASE_URL = "http://localhost:11434"


def generate_nouns(
    count: int = 100,
    categories: list[str] | None = None,
    model: str = "llama3",
    temperature: float = 0.8,
) -> list[str]:
    """
    Generate nouns using a local Ollama model.

    Args:
        count: Number of nouns to generate
        categories: List of noun categories to focus on (e.g., "receipt items", "school supplies")
        model: Ollama model to use (default: llama3)
        temperature: Temperature for generation (0.0-1.0, higher = more creative)
    """
    # Check Ollama is running
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is installed and running:\n"
            "  1. Install: https://ollama.com/download\n"
            "  2. Start:   ollama serve\n"
            f"  3. Pull model: ollama pull {model}"
        )

    if categories is None:
        categories = [
            "receipt items (groceries, household products, tools)",
            "school supplies (textbooks, pens, erasers, notebooks)",
            "everyday objects (phone, computer, water bottle, keys)",
            "clothing items (shirt, shoes, jacket, socks)",
            "furniture (desk, chair, table, lamp)",
        ]

    category_text = "\n".join(f"- {c}" for c in categories)

    prompt = f"""Generate exactly {count} common English nouns.
Focus on everyday items, especially items you'd see on a receipt or in daily life.

Include items from these categories:
{category_text}

Requirements:
- One noun per line, no numbering or extra text
- Nouns should be common, concrete objects (not adjectives or verbs)
- Include a mix of categories
- Avoid proper nouns and brand names
- Keep it to simple, everyday words

Generate {count} nouns:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2000,
            },
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama request failed (HTTP {response.status_code}): {response.text}\n"
            f"Make sure the model is pulled: ollama pull {model}"
        )

    response_text = response.json().get("response", "")
    nouns = [
        line.strip().lower()
        for line in response_text.strip().split("\n")
        if line.strip() and not line.strip()[0].isdigit()
    ]

    return nouns[:count]


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


def save_to_csv(nouns: list[str], output_path: Path) -> None:
    """Save nouns to CSV file, appending if it already exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()
    start_id = get_max_id(output_path)

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "noun"])
        for i, noun in enumerate(nouns, start_id + 1):
            writer.writerow([i, noun])


def save_to_json(nouns: list[str], output_path: Path) -> None:
    """Save nouns to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"nouns": nouns, "count": len(nouns)}, f, indent=2)


def save_to_txt(nouns: list[str], output_path: Path) -> None:
    """Save nouns to plain text file (one per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(nouns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate everyday nouns using a local Ollama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 nouns with default settings
  python generate_nouns.py

  # Generate 500 nouns and save to JSON
  python generate_nouns.py -n 500 -o nouns.json

  # Use a different local model
  python generate_nouns.py -m mistral

  # Generate with custom categories
  python generate_nouns.py --categories "food items" "tools" "animals"

Setup:
  1. Install Ollama:      https://ollama.com/download
  2. Start the server:   ollama serve
  3. Pull a model:       ollama pull llama3
        """,
    )
    parser.add_argument(
        "-n", "--count", type=int, default=100,
        help="Number of nouns to generate (default: 100)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("generated_nouns.csv"),
        help="Output file path (default: generated_nouns.csv). "
             "Format inferred from extension (.csv, .json, .txt)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="llama3",
        help="Ollama model to use (default: llama3). Must be pulled first with: ollama pull <model>",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.8,
        help="Temperature for generation, 0.0-1.0 (default: 0.8)",
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Custom noun categories to focus on "
             "(default: receipt items, school supplies, everyday objects, etc.)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("Error: --count must be > 0", file=sys.stderr)
        return 2

    if args.temperature < 0.0 or args.temperature > 1.0:
        print("Error: --temperature must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    try:
        print(f"Generating {args.count} nouns using local model '{args.model}'...")
        nouns = generate_nouns(
            count=args.count,
            categories=args.categories,
            model=args.model,
            temperature=args.temperature,
        )

        output_path = args.output
        suffix = output_path.suffix.lower()
        is_csv = suffix not in (".json", ".txt")
        appending = is_csv and output_path.exists()

        if suffix == ".json":
            save_to_json(nouns, output_path)
        elif suffix == ".txt":
            save_to_txt(nouns, output_path)
        else:
            save_to_csv(nouns, output_path)

        print(f"✓ Generated {len(nouns)} nouns")
        print(f"✓ {'Appended to' if appending else 'Saved to'} {output_path}")
        print()
        print("Sample nouns:")
        for noun in nouns[:10]:
            print(f"  - {noun}")
        if len(nouns) > 10:
            print(f"  ... and {len(nouns) - 10} more")

        return 0

    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())