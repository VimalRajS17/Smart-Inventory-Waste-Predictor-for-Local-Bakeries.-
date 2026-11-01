"""Convert transaction-level bakery CSV datasets into daily rows per item (date, item, sold).

Supports Kaggle datasets such as 'French Bakery Daily Sales' and 'The Bread Basket'.
Reads real headers dynamically without assuming column names.
Aggregates sales by day, isolates the top N items by volume (default 10), groups the rest as 'Other',
and inserts zero-sale rows for open days.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import glob
import os
import sys


def parse_date_str(val: str) -> str | None:
    """Extract standard ISO YYYY-MM-DD date from various timestamp/date string formats."""
    val = val.strip()
    if not val:
        return None

    # Handle standard YYYY-MM-DD or strings starting with YYYY-MM-DD
    if len(val) >= 10 and val[4] == "-" and val[7] == "-":
        try:
            datetime.strptime(val[:10], "%Y-%m-%d")
            return val[:10]
        except ValueError:
            pass

    # Try common alternate formats
    first_part = val.split()[0].replace(".", "-")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(first_part, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def detect_columns(headers: list[str]) -> tuple[str, str, str | None]:
    """Inspect headers to detect column names for date, item/article, and optional quantity."""
    norm = [h.strip() for h in headers]
    norm_lower = {h.lower(): h for h in norm}

    # 1. Detect Date Column
    date_col = None
    for candidate in ("date", "datetime", "timestamp", "transaction_date"):
        if candidate in norm_lower:
            date_col = norm_lower[candidate]
            break
    if not date_col:
        for candidate in norm_lower:
            if "date" in candidate:
                date_col = norm_lower[candidate]
                break

    # 2. Detect Item Column
    item_col = None
    for candidate in ("article", "item", "product", "product_name", "item_name", "articles"):
        if candidate in norm_lower:
            item_col = norm_lower[candidate]
            break
    if not item_col:
        for candidate in norm_lower:
            if "item" in candidate or "prod" in candidate or "article" in candidate:
                item_col = norm_lower[candidate]
                break

    # 3. Detect Quantity Column (optional, e.g. Bread Basket has no quantity column)
    qty_col = None
    for candidate in ("quantity", "qty", "sold", "units", "count"):
        if candidate in norm_lower:
            qty_col = norm_lower[candidate]
            break
    if not qty_col:
        for candidate in norm_lower:
            if "qty" in candidate or "quantity" in candidate:
                qty_col = norm_lower[candidate]
                break

    if not date_col or not item_col:
        raise ValueError(
            f"Could not automatically detect date and item columns from headers: {norm}."
        )

    return date_col, item_col, qty_col


def process_kaggle_file(input_path: str, output_dir: str = "data/processed", top_n: int = 10) -> str:
    """Read a raw transaction CSV, process it into daily rows per item, and write to output_dir."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read the file's real header and print it
    with open(input_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError(f"Input file '{input_path}' is empty.")

    print(f"Real headers detected in '{os.path.basename(input_path)}': {raw_headers}")

    date_col, item_col, qty_col = detect_columns(raw_headers)
    print(f"Mapped columns -> Date: '{date_col}', Item: '{item_col}', Quantity: '{qty_col or '[None - counting each row as 1]'}'")

    # Read all transactions to aggregate total volume per item
    raw_rows = []
    item_volume = Counter()

    with open(input_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            date_raw = row.get(date_col, "")
            item_raw = row.get(item_col, "")

            if not date_raw or not item_raw:
                continue

            clean_date = parse_date_str(date_raw)
            if not clean_date:
                continue

            clean_item = item_raw.strip()
            # Filter placeholder items (e.g. Bread Basket contains 'NONE')
            if clean_item.upper() in ("NONE", "ADJUSTMENT", ""):
                continue

            qty = 1
            if qty_col and row.get(qty_col) is not None:
                val = row[qty_col].strip().replace(",", ".")
                try:
                    parsed_qty = float(val)
                    if parsed_qty <= 0:
                        continue
                    qty = int(round(parsed_qty))
                except (ValueError, TypeError):
                    continue

            item_volume[clean_item] += qty
            raw_rows.append((clean_date, clean_item, qty))

    if not raw_rows:
        raise ValueError(f"No valid transaction rows found in '{input_path}'.")

    # Keep top N items by volume, group the rest as 'Other'
    top_items_list = [item for item, _ in item_volume.most_common(top_n)]
    top_items_set = set(top_items_list)
    has_other = len(item_volume) > top_n

    print(f"\nTotal unique items found: {len(item_volume)}")
    print(f"Top {min(top_n, len(top_items_list))} items by total volume:")
    for rank, (it, count) in enumerate(item_volume.most_common(top_n), start=1):
        print(f"  {rank}. {it}: {count:,} units")
    if has_other:
        other_total = sum(cnt for it, cnt in item_volume.items() if it not in top_items_set)
        print(f"  Remaining {len(item_volume) - top_n} items grouped as 'Other': {other_total:,} units")

    # Aggregate by (date, item)
    daily_sales = Counter()
    open_dates = set()

    for d, it, qty in raw_rows:
        mapped_item = it if it in top_items_set else "Other"
        daily_sales[(d, mapped_item)] += qty
        open_dates.add(d)

    sorted_open_dates = sorted(open_dates)
    tracked_items = sorted(top_items_list + (["Other"] if has_other else []))

    # Insert zero-sale rows for open days
    for d in sorted_open_dates:
        for it in tracked_items:
            # Ensure key exists in counter (defaults to 0 if not present)
            if (d, it) not in daily_sales:
                daily_sales[(d, it)] = 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_daily.csv")

    # Write processed daily rows
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "item", "sold"])
        for d in sorted_open_dates:
            for it in tracked_items:
                writer.writerow([d, it, daily_sales[(d, it)]])

    total_daily_rows = len(sorted_open_dates) * len(tracked_items)
    print(f"\nProcessing complete:")
    print(f"  Raw transactions read: {len(raw_rows):,}")
    print(f"  Open days: {len(sorted_open_dates)}")
    print(f"  Tracked items per day: {len(tracked_items)}")
    print(f"  Total daily rows written: {total_daily_rows:,}")
    print(f"  Output saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert transaction-level bakery CSV to daily sales per item (date, item, sold)."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to raw transaction CSV file. If omitted, scans data/raw/*.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top items to keep by volume (default: 10). Remainder grouped into 'Other'.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save processed CSVs (default: data/processed).",
    )

    args = parser.parse_args()

    files_to_process = []
    if args.file:
        files_to_process.append(args.file)
    else:
        raw_candidates = glob.glob(os.path.join("data", "raw", "*.csv"))
        if raw_candidates:
            files_to_process.extend(raw_candidates)
        else:
            print("No input file provided and no CSV files found in 'data/raw/'.")
            print("Usage: python scripts/load_kaggle.py <path-to-raw-csv> [--top-n 10]")
            sys.exit(1)

    for fp in files_to_process:
        print(f"\n========================================================")
        print(f"Processing '{fp}'...")
        print(f"========================================================")
        try:
            process_kaggle_file(fp, output_dir=args.output_dir, top_n=args.top_n)
        except Exception as e:
            print(f"Error processing '{fp}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

