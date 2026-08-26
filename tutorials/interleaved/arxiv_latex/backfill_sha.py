#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Record the SHA-256 of each sampled submission's raw bytes.

This is the **reuse key**: a document may be skipped by a later iteration only if
the pool already holds a conversion of the same input bytes under the same
converter config.  Keying on ``arxiv_id`` instead would be wrong -- arXiv lets
authors replace a submission, so a later snapshot can carry different source for
the same id, and a name-keyed check would silently serve the stale conversion.

The sample was staged before this was captured, so this backfills it by
re-reading each member from its source tar.  New iterations record it during
sampling and never need this script.

Writes ``source_sha256.json`` -- ``{manifest_dir: sha256}``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path


def member_name(shard: str, member: str) -> str:
    """Reproduce the sampler's directory naming, so the hash joins to the manifest."""
    return f"{shard.replace('.tar', '')}__{Path(member).name.removesuffix('.gz')}"


def backfill(src: Path, run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    out_path = run_dir / "source_sha256.json"
    hashes: dict[str, str] = json.loads(out_path.read_text()) if out_path.exists() else {}

    wanted = collections.defaultdict(set)
    for entry in manifest:
        if entry["dir"] not in hashes:
            wanted[entry["shard"]].add(entry["dir"])
    print(f"{len(manifest)} documents, {len(hashes)} already hashed, {sum(map(len, wanted.values()))} to do")

    done = 0
    for shard, names in sorted(wanted.items()):
        path = src / shard
        if not path.exists():
            print(f"  ! missing shard {shard}")
            continue
        try:
            with tarfile.open(path) as tf:
                for info in tf:
                    if not info.isfile():
                        continue
                    name = member_name(shard, info.name)
                    if name not in names:
                        continue
                    handle = tf.extractfile(info)
                    if handle is None:
                        continue
                    digest = hashlib.sha256()
                    # Streamed: some submissions are hundreds of MB, and this
                    # runs over every sampled document.
                    for chunk in iter(lambda h=handle: h.read(1 << 20), b""):
                        digest.update(chunk)
                    hashes[name] = digest.hexdigest()
                    done += 1
        except tarfile.TarError as exc:
            print(f"  ! {shard}: {exc}")
        out_path.write_text(json.dumps(hashes, indent=1, sort_keys=True))

    print(f"hashed {done} new; {len(hashes)}/{len(manifest)} total -> {out_path}")
    missing = [e["dir"] for e in manifest if e["dir"] not in hashes]
    if missing:
        print(f"still missing {len(missing)}, e.g. {missing[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    backfill(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
