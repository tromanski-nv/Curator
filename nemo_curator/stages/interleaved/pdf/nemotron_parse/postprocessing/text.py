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

"""Text-level helpers.

Kept separate and pure so they can be tested on strings alone --
:func:`is_to_be_continued` in particular decides where paragraphs join, and
getting it wrong silently corrupts documents.
"""

from __future__ import annotations

import re
import unicodedata

_RE_WORD = re.compile(r"\w{2,}([-.]\w+)*", re.UNICODE)
_RE_GREEK_MATH = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu"
    r"|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)",
    re.IGNORECASE,
)
_RE_TABULAR = re.compile(r"\\begin\{(tabular|table)\}.*?\\end\{\1\}", re.DOTALL)

#: Terminal punctuation: the thought is complete.
_TERMINATORS = ".?!"

#: Skipped when scanning backwards for the last meaningful character.  A
#: paragraph frequently finishes with a trailing reference marker such as
#: "... as shown in [12], 34" -- the sentence ended before those, so they must
#: not be mistaken for content.
_TRAILING_NOISE = "0123456789, \n\t"


def count_words(text: str) -> int:
    return sum(1 for _ in _RE_WORD.finditer(text))


def count_greek_math(text: str) -> int:
    return sum(1 for _ in _RE_GREEK_MATH.finditer(text))


def contains_tabular(text: str) -> bool:
    """A Table element with no tabular environment did not survive extraction."""
    return bool(_RE_TABULAR.search(text))


def count_char_types(text: str) -> tuple[int, int]:
    """``(latin, non-latin)`` letter counts, for charset skew."""
    latin = nonlatin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            if "LATIN" in unicodedata.name(ch):
                latin += 1
            else:
                nonlatin += 1
        except ValueError:
            nonlatin += 1
    return latin, nonlatin


def is_to_be_continued(text: str) -> bool:
    """Does this text run on into the next box?

    Scans BACKWARDS for the last meaningful character.  Terminal ``.?!`` means
    the thought is complete.  Digits, commas and whitespace are skipped rather
    than treated as an ending -- see :data:`_TRAILING_NOISE`.

    Anything else means the text stops mid-thought and the next box continues
    it.
    """
    for ch in reversed(text):
        if ch in _TERMINATORS:
            return False
        if ch not in _TRAILING_NOISE:
            return True
    return False


def _fix_dots(match: re.Match[str]) -> str:
    """Collapse a run of dots to at most five -- table-of-contents leaders."""
    s = match.group(0)
    return (" " if s.startswith(" ") else "") + min(5, s.count(".")) * "." + (" " if s.endswith(" ") else "")


def strip_markdown(text: str) -> str:
    """Reduce markdown to plain text.

    Order matters: bold (``**``) must go before italic (``*``), or the italic
    rule eats one asterisk of each bold delimiter and leaves the other behind.
    """
    text = re.sub(r"^(#+)\s*(.*)", r"\2", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"^\s*([-*+]|[0-9]+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"^\s*>\s*(.*)", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"(?:\s*\.\s*){3,}", _fix_dots, text, flags=re.DOTALL)


def is_degenerate(text: str, *, min_words: int, max_ratio: float) -> bool:
    """A long block dominated by a single repeated word is extraction failure."""
    words = text.split()
    if len(words) <= min_words:
        return False
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words) > max_ratio
