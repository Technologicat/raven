import csv
import pathlib
from typing import Any, Dict, List, Optional, Union

from unpythonic import islice

def parse_csv(file_path: Union[pathlib.Path, str],
              has_header: Optional[bool] = None,
              delimiter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse CSV file into structured data.

    `file_path`: Path to CSV file
    `has_header`: If True, first line is header; if False, no header; if None, autodetect (default)
    `delimiter`: If specified, use this delimiter; if None, autodetect tab/semicolon (default)

    Returns a list of dict (one dict per CSV row), where the keys are the field names.

    Raises `ValueError` if the file cannot be read or parsed.
    """
    # Autodetect delimiter if not provided
    if delimiter is None:
        try:
            with open(file_path, "r", newline="") as f:
                # Read first 10 lines to scan for delimiter patterns
                lines = list(islice(f)[:10])
                if not lines:  # Empty file
                    return []

                # Check for tab delimiter
                if any("\t" in row for row in lines):
                    delimiter = "\t"
                # Check for semicolon delimiter
                elif any(";" in row for row in lines):
                    delimiter = ";"
                else:
                    raise ValueError("Could not determine delimiter (tab or semicolon)")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    # Parse CSV
    try:
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Handle header detection
            if has_header is None:
                # Autodetect: if the first row has any alphabetic character, assume it's a header.
                # Heuristic, not perfect — users can override with an explicit `has_header`.
                try:
                    first_row = next(reader)
                    has_header = any(
                        any(c.isalpha() for c in field)
                        for field in first_row
                    )
                except StopIteration:
                    has_header = False

            # Rewind for the main parse loop. We consumed up to one row above in the
            # autodetect path; the explicit `has_header=False` path also needs a fresh
            # reader. The explicit `has_header=True` path hasn't touched the reader yet,
            # but rewinding is harmless.
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)

            # Parse entries
            entries = []
            header = None
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                if header is None:
                    if has_header:  # Use first line as header
                        header = row
                    else:  # No header: use generic field names
                        header = [f"column {i + 1}" for i in range(len(row))]
                        entries.append(dict(zip(header, row)))
                else:
                    entries.append(dict(zip(header, row)))
            return entries
    except Exception as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
