# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Patch the installed vLLM so Nemotron-Parse weights load correctly.

Stock vLLM 0.18.x ships a bug in ``nemotron_parse.py``: the cross-attention
``encoder_attn.kv_proj`` is a *merged* K+V projection (no Q), so its weight
loader shard ids must be integer indices ``0``/``1``. The stock file instead
passes the string ids ``"k"``/``"v"`` (only valid for a fused QKV layer). The
loader then falls through to ``validate_shard_id`` and raises::

    ValueError: This line should not be reached

This script rewrites those two shard ids to ``0``/``1`` in the vLLM copy
installed in the *current* Python environment.

It is idempotent (safe to re-run) and self-locating (finds vLLM via import), so
you can re-run it after every ``uv sync`` / venv rebuild, which restores the
stock (buggy) file::

    uv run python scripts/patch_vllm_nemotron_parse.py

Exit codes: 0 = patched now or already patched; 1 = target not found / error.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

# The stock (buggy) fragments and their fixed replacements. Keyed so the check
# for "already patched" is unambiguous and the edit stays surgical.
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ('".encoder_attn.k_proj", "k")', '".encoder_attn.k_proj", 0)'),
    ('".encoder_attn.v_proj", "v")', '".encoder_attn.v_proj", 1)'),
)


def locate_nemotron_parse() -> Path | None:
    """Return the path to vLLM's ``nemotron_parse.py`` in the active env."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    vllm_dir = Path(next(iter(spec.submodule_search_locations)))
    # Return the expected path whether or not it currently exists, so callers can
    # restore it from the .bak backup if a filesystem purge removed the file.
    return vllm_dir / "model_executor" / "models" / "nemotron_parse.py"


def apply_patch(path: Path, *, make_backup: bool = True) -> str:
    """Apply the shard-id fix to *path*.

    Returns one of ``"patched"``, ``"already"``, or ``"unexpected"``.
    """
    text = path.read_text(encoding="utf-8")

    if all(old not in text for old, _ in _REPLACEMENTS):
        # None of the buggy fragments remain. Confirm the fix is present so we
        # don't silently pass on a file that simply changed shape upstream.
        if all(new in text for _, new in _REPLACEMENTS):
            return "already"
        return "unexpected"

    if make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")

    # Drop any stale bytecode so the patched source is recompiled on next import.
    for pyc in (path.parent / "__pycache__").glob("nemotron_parse.*.pyc"):
        pyc.unlink()

    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path",
        default=None,
        help="Explicit path to vllm/model_executor/models/nemotron_parse.py "
        "(default: auto-locate in the active Python environment)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak backup before patching")
    args = parser.parse_args()

    path = Path(args.path) if args.path else locate_nemotron_parse()
    if path is None:
        print(
            "ERROR: could not locate the vLLM package. "
            "Is vLLM installed in this environment? (activate the venv or use `uv run`).",
            file=sys.stderr,
        )
        return 1
    if not path.is_file():
        # A filesystem purge can delete the model file while its .bak backup
        # survives. Restore from the backup so a re-run self-heals instead of
        # failing with "failed to be inspected" on every rank.
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.is_file():
            print(f"{path} missing; restoring from {backup.name} before patching.", file=sys.stderr)
            shutil.copy2(backup, path)
        else:
            print(
                f"ERROR: {path} is missing and no {backup.name} exists to restore from. "
                "Rebuild the environment: uv sync --extra interleaved_cuda12",
                file=sys.stderr,
            )
            return 1

    result = apply_patch(path, make_backup=not args.no_backup)
    if result == "patched":
        print(f"Patched {path} (encoder_attn.kv_proj shard ids -> 0/1).")
        return 0
    if result == "already":
        print(f"Already patched: {path} (nothing to do).")
        return 0

    print(
        f"ERROR: {path} does not contain the expected stock or patched shard-id lines. "
        "The vLLM version may have changed; inspect the file manually.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
