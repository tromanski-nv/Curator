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

"""Fixtures for the arXiv LaTeXML tests.

The converter is stubbed rather than installed: ``latexmlc`` is a 2.5 GB Perl
stack, and none of the behaviour under test is LaTeXML's.  What is under test is
what this package does with whatever LaTeXML returns -- which argv it builds,
how it grades the output, what it writes when nothing comes back.
"""

import gzip
import io
import os
import stat
import tarfile
from pathlib import Path

import pytest

#: A minimal but realistic converted document: enough text to clear the
#: MIN_TEXT_CHARS gate, one <math> with alttext, one figure, two headings.
GOOD_HTML = (
    "<html><body><h1>Title</h1><p>" + ("word " * 200) + "</p>"
    "<math alttext='E=mc^2'><mi>E</mi></math><img src='f.png'><h2>Section</h2></body></html>"
)

LATEX_SOURCE = rb"""\documentclass{article}
\begin{document}
Hello $E=mc^2$ world.  % a comment
\begin{figure}\includegraphics{f.png}\caption{c}\end{figure}
\end{document}
"""


def write_latexmlc(directory: Path, *, body: str = GOOD_HTML, stderr: str = "", exit_code: int = 0) -> Path:
    """Write a fake ``latexmlc`` that emits *body* to its --destination."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "latexmlc"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, pathlib\n"
        "dest = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--destination='))\n"
        "p = pathlib.Path(dest); p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_text({body!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def latexmlc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Put a fake ``latexmlc`` on PATH; return a callable to reconfigure it."""

    bindir = tmp_path / "bin"

    def configure(**kwargs) -> Path:
        path = write_latexmlc(bindir, **kwargs)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        return path

    configure()
    return configure


def make_shard(path: Path, members: dict[str, bytes]) -> Path:
    """Build an arXiv-style source tar from {member_name: payload}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    """One shard holding a LaTeX submission and a PDF-only submission."""
    return make_shard(
        tmp_path / "src" / "arXiv_src_0301_001.tar",
        {
            "0301/astro-ph0301029.gz": gzip.compress(LATEX_SOURCE),
            "0301/1203.5560.pdf": b"%PDF-1.4 not latex",
        },
    )
