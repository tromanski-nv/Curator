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

"""LatexmlConfig and the LaTeXML log parsing."""

import dataclasses

import pytest

from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import (
    AR5IV_CONFIG,
    EXCLUDED_ARGS,
    LATEXML_TIMEOUT_S,
    WALL_TIMEOUT_MARGIN_S,
    LatexmlConfig,
    count_severities,
    error_kinds,
)

#: The invocation the published arXiv corpus was converted with.  Pinned in full
#: because the argv is recorded verbatim as dataset provenance: a change here
#: silently redefines what every future row means, so it should have to be made
#: on purpose.
AR5IV_ARGV = (
    "latexmlc",
    "--preload=ar5iv.sty",
    "--path=/opt/ar5iv-bindings/bindings",
    "--path=/opt/ar5iv-bindings/supported_originals",
    "--format=html5",
    "--pmml",
    "--mathtex",
    "--timeout=600",
    "--noinvisibletimes",
    "--source=main.tex",
    "--destination=/out/index.html",
)


def test_default_config_reproduces_the_corpus_invocation():
    assert AR5IV_CONFIG.argv("main.tex", "/out/index.html") == AR5IV_ARGV


def test_pmml_precedes_mathtex():
    """--mathtex attaches alttext to what --pmml produced; reversed, alttext is lost."""
    argv = AR5IV_CONFIG.argv("a", "b")
    assert argv.index("--pmml") < argv.index("--mathtex")


def test_source_and_destination_come_last():
    argv = AR5IV_CONFIG.argv("a.tex", "b.html")
    assert argv[-2:] == ("--source=a.tex", "--destination=b.html")


def test_config_is_frozen():
    """Config is recorded as provenance, so it must not drift after the fact."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        AR5IV_CONFIG.latexml_timeout_s = 1


def test_wall_timeout_sits_above_the_latexml_timeout():
    """LaTeXML must get to report its own timeout rather than be killed mid-write."""
    config = LatexmlConfig(latexml_timeout_s=2700)
    assert config.wall_timeout_s == 2700 + WALL_TIMEOUT_MARGIN_S
    assert AR5IV_CONFIG.wall_timeout_s == LATEXML_TIMEOUT_S + WALL_TIMEOUT_MARGIN_S


def test_bindings_can_be_omitted_entirely():
    argv = LatexmlConfig(bindings_root=None, preload=()).argv("a", "b")
    assert not [a for a in argv if a.startswith("--path=") or a.startswith("--preload=")]


def test_extra_args_land_before_source_and_destination():
    argv = LatexmlConfig(extra_args=("--verbose",)).argv("a", "b")
    assert argv[argv.index("--verbose") + 1].startswith("--source=")


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(("--includestyles",), id="includestyles"),
        pytest.param(("--css",), id="css_bare"),
        pytest.param(("--css=custom.css",), id="css_with_value"),
        pytest.param(("--verbose", "--includestyles"), id="among_others"),
    ],
)
def test_excluded_args_are_rejected(extra):
    """--includestyles fights the ar5iv bindings; --css embeds a CDN dependency."""
    with pytest.raises(ValueError, match="must not be passed"):
        LatexmlConfig(extra_args=extra)


def test_excluded_args_are_the_documented_pair():
    assert set(EXCLUDED_ARGS) == {"--includestyles", "--css"}


# --- log parsing ---


def test_count_severities():
    log = "Warning:a:b x\nError:c:d y\nWarning:e:f z\nFatal:g:h w\n"
    assert count_severities(log) == (2, 1, 1)


def test_count_severities_ignores_mid_line_matches():
    assert count_severities("something Error:x:y not at line start\n") == (0, 0, 0)


def test_error_kinds_are_distinct_and_ordered_by_first_appearance():
    log = "Error:undefined:a\nError:misdefined:b\nError:undefined:c\n"
    assert error_kinds(log) == ("undefined", "misdefined")


def test_error_kinds_empty_for_a_clean_log():
    assert error_kinds("Warning:foo:bar harmless\n") == ()
