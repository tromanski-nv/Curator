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

"""Ingestion of arXiv LaTeX source.

Two paths convert arXiv bulk source tarballs, in separate subpackages.  This
package re-exports neither, so an import names the path it means.

``latexml``
    Runs ``latexmlc``, a full LaTeX processor, over each submission and emits
    HTML5 with presentation MathML.  Math is rendered as MathML with the
    original TeX kept in ``alttext``; macros are expanded by the TeX engine.
    Every submission produces a row, including those that produced no HTML, so
    the denominator stays "of all submissions".  Requires the ``latexmlc``
    binary, which is not a Python package and is not installed by pip; see the
    package README for the container image.

``regex``
    Pattern-matches the LaTeX source directly, without invoking a LaTeX
    processor, and emits an :class:`~nemo_curator.tasks.InterleavedBatch` whose
    rows follow the document's reading order -- body text, figure, caption,
    body text.  Math and unrecognised macros become placeholders.  Requires
    nothing beyond Curator.
"""
