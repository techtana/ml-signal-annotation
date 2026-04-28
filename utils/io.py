import json
from datetime import datetime as dt


def update_json(path: str, ndx: int, item: str, new_value):
    """Update a single field in a list-of-dicts JSON config file."""
    with open(path, "r+") as f:
        data = json.load(f)
        data[ndx][item] = new_value
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def print_section(phrase: str, level: str = "", n: int = 1, width: int = 60):
    """Print a timestamped bordered section header to stdout.

    Parameters
    ----------
    phrase : str
        Text to display inside the border.
    level : str
        "header" for double-line border, "debug" for hash border,
        or empty string for plain border.
    n : int
        Number of border lines to print above and below.
    width : int
        Total width of the border.
    """
    level = level.lower()
    if level == "debug":
        top, side, corner = "#", "#", "#"
    elif level == "header":
        top, side, corner = "=", "|", "+"
    else:
        top, side, corner = "-", "|", "+"

    border = corner + top * width + corner
    maxwidth = width - 4

    chunks = [phrase[i:i + maxwidth] for i in range(0, len(phrase), maxwidth)] if len(phrase) > maxwidth else [phrase]
    chunks += [" ", f"TIMESTAMP >> {dt.today().strftime('%Y-%m-%d %H:%M:%S')}"]

    for _ in range(n):
        print(border)
    for line in chunks:
        pad = (width - len(line)) - 2
        print(f"{side} {line}{' ' * pad} {side}")
    for _ in range(n):
        print(border)
