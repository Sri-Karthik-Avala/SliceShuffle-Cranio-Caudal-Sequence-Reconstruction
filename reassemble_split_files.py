"""Rebuild files that were split into <100 MB parts for GitHub.

Run from anywhere:  python reassemble_split_files.py [--delete-parts]
"""
import hashlib
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(os.path.join(here, "split_files.json"), encoding="utf-8"))
for item in manifest["files"]:
    target = os.path.join(here, item["path"])
    tmp = target + ".partial"
    digest = hashlib.sha256()
    with open(tmp, "wb") as out:
        for part in item["parts"]:
            with open(os.path.join(here, part), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
                    out.write(block)
    if digest.hexdigest() != item["sha256"] or os.path.getsize(tmp) != item["size"]:
        os.remove(tmp)
        sys.exit("checksum mismatch: " + item["path"])
    os.replace(tmp, target)
    if "--delete-parts" in sys.argv:
        for part in item["parts"]:
            os.remove(os.path.join(here, part))
    print("rebuilt", item["path"])
