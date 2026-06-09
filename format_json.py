"""
Format ULTRA MEGA SMESH/project.json from a single minified line
into pretty-printed JSON so it can be searched and navigated by line number.

Output: ULTRA MEGA SMESH/project_formatted.json
"""
import json
import os
import sys

SRC = os.path.join("ULTRA MEGA SMESH", "project.json")
DST = os.path.join("ULTRA MEGA SMESH", "project_formatted.json")


def main() -> None:
    print(f"Reading {SRC} …")
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Writing {DST} …")
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    src_size = os.path.getsize(SRC)
    dst_size = os.path.getsize(DST)
    with open(DST, "r", encoding="utf-8") as f:
        lines = sum(1 for _ in f)

    print(f"Done.")
    print(f"  {SRC}: {src_size / 1_000_000:.1f} MB  (1 line)")
    print(f"  {DST}: {dst_size / 1_000_000:.1f} MB  ({lines:,} lines)")


if __name__ == "__main__":
    main()
