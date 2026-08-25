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

"""Tests for canonical arXiv identifiers.

Every case here is a spelling that actually appears in one of the corpora.
A join is only as good as this function, and a silent normalisation bug shows
up as a coverage gap rather than as an error.
"""

from __future__ import annotations

import pytest

from nemo_curator.stages.interleaved.identifiers.arxiv import canon, year_month


class TestCanon:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Post-2007 scheme, as each corpus spells it.
            ("2308.10008", "2308.10008"),
            ("2410/2410.10730", "2410.10730"),  # Nemotron-Parse elements
            ("0709.0053.pdf", "0709.0053"),  # LaTeXML pool
            ("2308.10008v3", "2308.10008"),
            ("arXiv:2308.10008", "2308.10008"),
            ("https://arxiv.org/abs/2308.10008", "2308.10008"),
            ("  2308.10008  ", "2308.10008"),
            # Pre-2007 scheme.
            ("math/0112027", "math/0112027"),
            ("math/math0001001", "math/0001001"),  # Nemotron-Parse doubles the archive
            ("cond-mat/cond-mat0410443", "cond-mat/0410443"),
            ("hep-th/0112149", "hep-th/0112149"),
            ("cond-mat0605660.pdf", "cond-mat/0605660"),  # LaTeXML pool drops the slash
            ("math.AG/0112027", "math/0112027"),  # subject class stripped
            # Nothing to normalise.
            (None, None),
            ("", None),
        ],
    )
    def test_every_spelling_the_corpora_use(self, raw: str | None, expected: str | None) -> None:
        assert canon(raw) == expected

    def test_a_doubled_archive_is_not_read_as_a_plain_old_id(self) -> None:
        """``math/math0001001`` also matches the plain old-scheme pattern,
        against a seven-digit run that is not the paper's number.  The order of
        the patterns in canon() is what keeps this right, so pin it."""
        assert canon("math/math0001001") == "math/0001001"
        assert canon("math/math0001001") != "math/math0001"

    def test_canon_is_idempotent(self) -> None:
        """A corpus that has already been canonicalised must survive a second
        pass unchanged, or re-running a pipeline would corrupt it."""
        for raw in ("2410/2410.10730", "cond-mat0605660.pdf", "math/math0001001", "2308.10008v3"):
            once = canon(raw)
            assert canon(once) == once, raw

    def test_an_unrecognised_id_is_passed_through_rather_than_guessed_at(self) -> None:
        """It fails a join loudly instead of colliding with a real paper."""
        assert canon("not-an-arxiv-id") == "not-an-arxiv-id"


class TestYearMonth:
    @pytest.mark.parametrize(
        ("canonical", "expected"),
        [
            ("2308.10008", (2023, 8)),
            ("2512.00001", (2025, 12)),
            ("2601.00001", (2026, 1)),
            ("math/0112027", (2001, 12)),
            ("hep-th/9901001", (1999, 1)),  # a two-digit year >= 90 is the 1900s
            ("not-an-id", (0, 0)),
        ],
    )
    def test_both_schemes_carry_their_date(self, canonical: str, expected: tuple[int, int]) -> None:
        assert year_month(canonical) == expected
