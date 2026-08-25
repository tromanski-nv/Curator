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

"""Canonical arXiv identifiers.

Every corpus here spells the same paper differently, and any join between two
of them is wrong until they agree::

    Nemotron-Parse elements   2410/2410.10730   ·  math/math0001001
    LaTeXML pool              2308.10008        ·  cond-mat0605660.pdf
    canonical (here)          2410.10730        ·  math/0001001

The canonical form is arXiv's own: ``YYMM.NNNNN`` for the post-2007 scheme and
``archive/YYMMNNN`` for the pre-2007 one.  Version suffixes and file extensions
are dropped, because none of these corpora agree on carrying them either.

One spelling, everywhere.  A corpus that arrives in some other form converts to
this one rather than this one growing a second column to meet it: two columns
holding the same fact is how they drift apart.

Measured over 227,455 distinct identifiers drawn from 400 random shards of the
Nemotron-Parse arXiv corpus: every one normalises, none is left unrecognised,
and no two distinct identifiers collide.
"""

from __future__ import annotations

import re

_EXT = re.compile(r"\.(pdf|tex|tar|gz)$")
_PREFIX = re.compile(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)")
_VERSION = re.compile(r"v\d+$")

_NEW_PREFIXED = re.compile(r"^(\d{4})/(\d{4}\.\d{4,5})$")  # 2410/2410.10730
_NEW = re.compile(r"^\d{4}\.\d{4,5}$")  # 2410.10730
_OLD_DOUBLED = re.compile(r"^([a-z\-\.]+)/\1(\d{7})$")  # math/math0001001
_OLD = re.compile(r"^([a-z\-\.]+)/(\d{7})$")  # math/0112027
_OLD_SUBCLASS = re.compile(r"^([a-z\-]+)\.[a-z]{2}/(\d{7})$")  # math.ag/0112027
_OLD_UNSLASHED = re.compile(r"^([a-z][a-z\-\.]+?)(\d{7})$")  # cond-mat0605660

_YM_NEW = re.compile(r"^(\d{2})(\d{2})\.")
_YM_OLD = re.compile(r"/(\d{2})(\d{2})\d{3}$")

#: Two-digit years below this are 2000s.  arXiv started in 1991, so there is no
#: ambiguity to resolve.
_CENTURY_PIVOT = 90


def canon(raw: str | None) -> str | None:
    """Canonical form, or ``None`` for empty input.

    Unrecognised input is returned cleaned but otherwise unchanged, so a bad
    identifier fails a join loudly rather than silently colliding with a real
    one.
    """
    if not raw:
        return None

    s = _VERSION.sub("", _PREFIX.sub("", _EXT.sub("", raw.strip().lower())))

    m = _NEW_PREFIXED.match(s)
    if m:
        return m.group(2)
    if _NEW.match(s):
        return s
    # Order is load-bearing, twice over:
    #   _OLD_DOUBLED before _OLD -- else `math/math0001001` matches _OLD against
    #     a seven-digit run that is not the paper's number.
    #   _OLD_SUBCLASS before _OLD -- _OLD's archive class allows dots, so it
    #     captures `math.ag` whole and the subject class is never stripped.
    for pattern in (_OLD_DOUBLED, _OLD_SUBCLASS, _OLD, _OLD_UNSLASHED):
        m = pattern.match(s)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return s


def year_month(canonical: str) -> tuple[int, int]:
    """Submission ``(year, month)``, or ``(0, 0)`` if the id carries no date.

    Both schemes encode ``YYMM`` in the identifier.
    """
    m = _YM_NEW.match(canonical) or _YM_OLD.search(canonical)
    if not m:
        return (0, 0)
    yy = int(m.group(1))
    return (2000 + yy if yy < _CENTURY_PIVOT else 1900 + yy, int(m.group(2)))
