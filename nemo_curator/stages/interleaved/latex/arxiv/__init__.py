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

Two independent paths convert arXiv bulk source tarballs.  They differ in
method, dependencies and output shape; neither subsumes the other, and both
are maintained here side by side.  Import from the subpackage you want --
this package deliberately re-exports neither, so the choice is explicit at
every call site.

``latexml``
    Runs ``latexmlc``, a full LaTeX processor, over each submission and emits
    HTML5 with presentation MathML.  Math is rendered as MathML with the
    original TeX preserved in ``alttext``; macros are expanded by the TeX
    engine.  Requires the ``latexmlc`` binary, which is not a Python package
    and is not installed by pip -- see the tutorial for the container image.
    Costs roughly 44 core-seconds per document.

``regex``
    Pattern-matches the LaTeX source directly and emits text segments with
    figure references.  Math and unrecognised macros become placeholders.
    Requires nothing beyond Curator, and runs in milliseconds per document.

Choose on what the downstream task needs: ``latexml`` when math and document
structure must survive, ``regex`` when volume matters more than fidelity or
when no external binary is available.
"""
