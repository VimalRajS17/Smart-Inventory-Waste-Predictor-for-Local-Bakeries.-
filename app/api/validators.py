"""Input validation helpers for API endpoints."""

from datetime import date


def parse_non_negative_int(val, field_name: str, allow_none: bool = False):
    """Validate and convert value to non-negative integer. Returns (int_val, error_msg)."""
    if val is None:
        if allow_none:
            return None, None
        return None, f"Field '{field_name}' is required."

    try:
        # Check if float string like '3.5'
        if isinstance(val, str) and "." in val:
            return None, f"Field '{field_name}' must be an integer, got '{val}'."
        int_val = int(val)
    except (ValueError, TypeError):
        return None, f"Field '{field_name}' must be an integer."

    if int_val < 0:
        return None, f"Field '{field_name}' must be a non-negative integer (got {int_val})."

    return int_val, None


def parse_iso_date(date_str, field_name: str = "date", allow_none: bool = False):
    """Validate that date_str parses as YYYY-MM-DD. Returns (date_str, error_msg)."""
    if date_str is None or (isinstance(date_str, str) and not date_str.strip()):
        if allow_none:
            return None, None
        return None, f"Field '{field_name}' is required."

    if not isinstance(date_str, str):
        return None, f"Field '{field_name}' must be a date string in YYYY-MM-DD format."

    clean_str = date_str.strip()
    try:
        parsed = date.fromisoformat(clean_str)
        # Ensure it has exactly YYYY-MM-DD format
        if len(clean_str) != 10 or clean_str[4] != "-" or clean_str[7] != "-":
            return None, f"Field '{field_name}' must be in YYYY-MM-DD format (got '{date_str}')."
        return clean_str, None
    except ValueError:
        return None, f"Field '{field_name}' must be a valid date in YYYY-MM-DD format (got '{date_str}')."

