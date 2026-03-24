#!/usr/bin/env python3
"""Build a Khmer name corpus from the Perchance name generator source.

This script pulls the generator payload from:
https://perchance.org/api/downloadGenerator?generatorName=namegen-as-kh

It extracts masculine names (nameM), feminine names (nameF), and surnames,
keeps Khmer Unicode only, randomizes names, and writes a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SOURCE_URL = "https://perchance.org/api/downloadGenerator?generatorName=namegen-as-kh"
KHMER_RE = re.compile(r"[\u1780-\u17FF]+")
ENTRY_RE = re.compile(r"\(([^)]*)\)(?:\^(\d+(?:\.\d+)?))?$")
DISALLOWED_END_CHAR = "្"


@dataclass(frozen=True)
class WeightedName:
    text: str
    weight: float


def is_valid_name_part(name_part: str) -> bool:
    return bool(name_part) and not name_part.endswith(DISALLOWED_END_CHAR)


def fetch_generator_source(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # The generator payload is URL-encoded inside HTML and often keeps line
    # breaks as escaped text ("\\n"). Decode and normalize so list sections are
    # easy to parse line-by-line.
    decoded = urllib.parse.unquote(raw)
    return decoded.replace("\\r\\n", "\n").replace("\\n", "\n")


def extract_section(decoded_html: str, section_name: str) -> str:
    marker = f"\n{section_name}\n"
    start = decoded_html.find(marker)
    if start == -1:
        raise ValueError(f"Section '{section_name}' not found in payload")

    start += len(marker)
    # Sections in this generator are contiguous and terminated by the next
    # list label or explicit END marker.
    candidates = []
    for next_marker in ("\nnameM\n", "\nnameF\n", "\nsurname\n", "\n// END //"):
        idx = decoded_html.find(next_marker, start)
        if idx != -1:
            candidates.append(idx)

    if not candidates:
        raise ValueError(f"Could not detect end of section '{section_name}'")

    end = min(candidates)
    return decoded_html[start:end]


def parse_weighted_names(section_text: str) -> list[WeightedName]:
    names: list[WeightedName] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        match = ENTRY_RE.search(line)
        if not match:
            continue

        khmer_in_parens = match.group(1)
        weight = float(match.group(2)) if match.group(2) is not None else 1.0
        khmer_only = "".join(KHMER_RE.findall(khmer_in_parens))
        if is_valid_name_part(khmer_only):
            names.append(WeightedName(text=khmer_only, weight=weight))

    if not names:
        raise ValueError("No Khmer names were parsed from section")

    return names


def choose_weighted(rng: random.Random, pool: list[WeightedName]) -> str:
    choice = rng.choices(pool, weights=[item.weight for item in pool], k=1)[0]
    return choice.text


def generate_rows(
    count: int,
    surnames: list[WeightedName],
    male_names: list[WeightedName],
    female_names: list[WeightedName],
    seed: int | None,
    unique: bool,
) -> list[tuple[int, str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()

    max_attempts = count * 50 if unique else count
    attempts = 0
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        gender = rng.choice(["male", "female"])
        given_pool = male_names if gender == "male" else female_names

        surname = choose_weighted(rng, surnames)
        given_name = choose_weighted(rng, given_pool)
        if not is_valid_name_part(surname) or not is_valid_name_part(given_name):
            continue

        full_name = f"{surname} {given_name}"
        full_name = " ".join(full_name.split())
        key = (gender, full_name)

        if unique and key in seen:
            continue

        seen.add(key)
        rows.append((len(rows) + 1, gender, full_name))

    return rows


def write_csv(rows: list[tuple[int, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gender", "khmer_name"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Khmer names from the Perchance generator source, randomize "
            "male/female full names, and export CSV."
        )
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1000,
        help="Number of randomized names to generate (default: 1000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("khmer_name_corpus.csv"),
        help="Output CSV path (default: khmer_name_corpus.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible output",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate randomized names in output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("--count must be > 0", file=sys.stderr)
        return 2

    try:
        decoded = fetch_generator_source(SOURCE_URL)
        male_section = extract_section(decoded, "nameM")
        female_section = extract_section(decoded, "nameF")
        surname_section = extract_section(decoded, "surname")

        male_names = parse_weighted_names(male_section)
        female_names = parse_weighted_names(female_section)
        surnames = parse_weighted_names(surname_section)

        rows = generate_rows(
            count=args.count,
            surnames=surnames,
            male_names=male_names,
            female_names=female_names,
            seed=args.seed,
            unique=not args.allow_duplicates,
        )
        if len(rows) < args.count:
            print(
                (
                    f"Requested {args.count} names, generated {len(rows)} unique rows. "
                    "Use --allow-duplicates if you need more rows."
                ),
                file=sys.stderr,
            )

        write_csv(rows, args.output)
        print(f"Wrote {len(rows)} rows to {args.output}")
        return 0
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Network error while fetching generator source: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Parsing error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
