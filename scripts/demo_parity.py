"""Check that the four language demos print the same report.

Usage:  python scripts/demo_parity.py py.txt cpp.txt rust.txt java.txt

Each file is a captured demo stdout.  Before comparing we normalise line
endings (CRLF to LF), trailing whitespace, and the one non-ASCII character
the demos print (the em dash in the title, which Java prints as a hyphen).
Every number must then match the Python reference character for character.
Exits non-zero and prints a unified diff on the first mismatch.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path


def normalise(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("—", "-")
    return [line.rstrip() for line in text.strip("\n").split("\n")]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    paths = [Path(p) for p in argv[1:]]
    ref = normalise(paths[0].read_text(encoding="utf-8", errors="replace"))
    if len(ref) < 10:
        print(f"{paths[0]}: reference output has only {len(ref)} lines")
        return 1
    status = 0
    for p in paths[1:]:
        got = normalise(p.read_text(encoding="utf-8", errors="replace"))
        if got == ref:
            print(f"{p}: identical to {paths[0]} ({len(ref)} lines)")
            continue
        status = 1
        print(f"{p}: differs from {paths[0]}")
        sys.stdout.writelines(
            line + "\n"
            for line in difflib.unified_diff(ref, got, str(paths[0]), str(p), lineterm="")
        )
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
