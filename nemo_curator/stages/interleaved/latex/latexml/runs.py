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

r"""Record what each conversion iteration tried, and why.

A conversion run is an experiment.  Six months later the only questions that
matter are *what was different about this one* and *can I reproduce it* -- and
neither is answerable from the output HTML alone.

The fields here exist because each has already been needed:

``rationale`` / ``changes``
    Two runs of this pipeline differ only in flag **order** and produce corpora
    that are not comparable -- one has math, one silently does not.  A diff of
    the configs shows *what* changed; only prose says *why*.
``argv``
    Recorded as an ordered list, never a set or a summary, for the same reason.
``tools``
    LaTeXML version *and* commit, ar5iv bindings commit, and the sha256 of the
    pinned container.  An image tag is a mutable pointer; a hash is not.
``results``
    Filled in after the fact so a run carries its own outcome, making two
    iterations directly comparable without re-deriving anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUN_FILENAME = "run.json"


@dataclass
class RunRecord:
    """Provenance and intent for one conversion iteration."""

    run_id: str
    label: str
    rationale: str
    """Why this iteration was run, in prose.  The field a config diff cannot replace."""

    changes: list[str] = field(default_factory=list)
    """What differs from the previous iteration, one concrete change per entry."""

    argv: list[str] = field(default_factory=list)
    """Full ordered converter argv, verbatim.  Order is semantically load-bearing."""

    tools: dict[str, str] = field(default_factory=dict)
    sample: dict[str, object] = field(default_factory=dict)
    results: dict[str, object] = field(default_factory=dict)
    created_utc: str = ""
    superseded_by: str | None = None

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / RUN_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def read(cls, directory: Path) -> RunRecord | None:
        path = directory / RUN_FILENAME
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))


def diff_argv(before: list[str], after: list[str]) -> dict[str, list[str]]:
    """Compare two argvs, keeping order changes visible.

    A plain set difference would report *no change* for the reordering that
    silently deletes all math from a corpus, so a reordering of otherwise
    identical flags is reported explicitly.
    """
    before_set, after_set = set(before), set(after)
    added = [a for a in after if a not in before_set]
    removed = [b for b in before if b not in after_set]
    common_before = [b for b in before if b in after_set]
    common_after = [a for a in after if a in before_set]
    return {
        "added": added,
        "removed": removed,
        "reordered": common_before if common_before != common_after else [],
    }
