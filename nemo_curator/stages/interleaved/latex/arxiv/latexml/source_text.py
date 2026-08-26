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

"""Reading raw LaTeX source well enough to decide what to hand to LaTeXML.

Deliberately duplicated from the sibling ``regex`` path rather than imported
from it.  The two conversion paths are kept independent so that either can
change its parsing without silently altering the other; these two functions
are the whole of the overlap, and a shared module for them would couple the
packages for forty lines.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Environments whose bodies are literal text, so a ``%`` inside one is
#: content rather than the start of a comment.
VERBATIM_ENVIRONMENTS: tuple[str, ...] = (
    "verbatim",
    "verbatim*",
    "Verbatim",
    "lstlisting",
    "alltt",
    "minted",
    "comment",
    "semiverbatim",
)


def _alternation(names: Iterable[str]) -> str:
    """Build a regex alternation with longer names first so ``verbatim*`` wins over ``verbatim``."""
    return "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))


# ``(?<!\\)((?:\\\\)*)`` keeps ``\%`` (escaped percent) but allows ``\\%``
# (escaped backslash followed by a real comment).
_COMMENT_RE = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")
_VERBATIM_BLOCK_RE = re.compile(
    r"\\begin\s*\{(" + _alternation(VERBATIM_ENVIRONMENTS) + r")\}.*?\\end\s*\{\1\}",
    re.DOTALL,
)


def decode_text(payload: bytes) -> str:
    """Decode LaTeX source, falling back to latin-1 for the ~2% that are not UTF-8."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="replace")


def strip_comments(tex: str) -> str:
    r"""Remove ``%`` comments, preserving ``\%`` and the contents of verbatim environments."""
    if "%" not in tex:
        return tex
    out: list[str] = []
    cursor = 0
    for block in _VERBATIM_BLOCK_RE.finditer(tex):
        out.append(_COMMENT_RE.sub(r"\1", tex[cursor : block.start()]))
        out.append(block.group(0))
        cursor = block.end()
    out.append(_COMMENT_RE.sub(r"\1", tex[cursor:]))
    return "".join(out)
