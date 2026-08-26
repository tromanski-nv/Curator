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

r"""Strip converter-generated chrome that is not part of the paper.

LaTeXML closes every document with a ``ltx_page_logo`` div -- "Generated on
<date> by LaTeXML" plus a base64 mascot image. It is byte-identical across the
corpus apart from the timestamp, so left in place it would appear once in every
extracted document and become one of the most frequent n-grams in the dataset.

Removing it at the HTML layer rather than in a text post-filter matters: the
timestamp makes the string non-constant, so a naive exact-match dedup or
boilerplate filter downstream would not catch it.
"""

from __future__ import annotations

import re

#: The generator footer. Matched on the class rather than the text so it is
#: robust to LaTeXML changing the wording or the date format.
_PAGE_LOGO_RE = re.compile(
    r"<div\b[^>]*\bclass=\"[^\"]*\bltx_page_logo\b[^\"]*\"[^>]*>.*?</div>",
    re.DOTALL | re.IGNORECASE,
)


def strip_boilerplate(html: str) -> str:
    """Remove the LaTeXML generator footer and mascot from a converted document."""
    return _PAGE_LOGO_RE.sub("", html)
