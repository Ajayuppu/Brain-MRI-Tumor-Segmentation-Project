"""File hashing and time helpers shared by training, freezing and test evaluation."""

import hashlib
import time


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
