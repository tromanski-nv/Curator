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

"""What differs between Nemotron-Parse releases, in one place.

Everything the pipeline needs to know about a model release lives on a
:class:`NemotronParseProfile`: which weights to load, what prompt to send, and
whether the model emits floats in reading order.  The stages read the profile
rather than sniffing the model path, so moving to a new release is one
argument, not a hunt through four modules for hard-coded version strings.

Swapping v1.2 for a later release that keeps the same output contract::

    NemotronParsePDFReader(manifest_path=..., parse_version="v2.0",
                           model_path="nvidia/NVIDIA-Nemotron-Parse-v2.0")

An unregistered version is accepted on the caller's word that the contract is
unchanged; :func:`resolve` says so in the log rather than failing, and the
model path must be given explicitly because guessing a HuggingFace id is worse
than asking for one.  When a release *does* change the contract, register what
it actually does instead::

    register_profile(
        NemotronParseProfile(
            name="v2.0",
            model_path="nvidia/NVIDIA-Nemotron-Parse-v2.0",
            markers=("v2.0",),
            prompt_base=...,
            text_in_pic_tokens=...,
            floats_in_reading_order=True,
        )
    )
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from loguru import logger

#: The prompt prefix shared by the releases recorded here.
PROMPT_BASE = "</s><s><predict_bbox><predict_classes><output_markdown>"

#: ``(predict, do not predict)`` text inside pictures.
TEXT_IN_PIC_TOKENS = ("<predict_text_in_pic>", "<predict_no_text_in_pic>")


@dataclass(frozen=True)
class NemotronParseProfile:
    """One Nemotron-Parse release, and what the pipeline must do differently for it.

    Parameters
    ----------
    name
        Short version label, e.g. ``"v1.2"``.  This is what ``parse_version``
        takes and what appears in task metadata.
    model_path
        Default HuggingFace id or local path for these weights.  ``None`` means
        the release's canonical id is not recorded here and the caller must
        supply one -- a wrong-but-plausible model id is worse than a required
        argument.
    markers
        Substrings that identify this release inside a model path, used to
        recognise a version the caller did not name.
    prompt_base
        The task prompt, before the text-in-pic token.
    text_in_pic_tokens
        ``(predict, do not predict)``, or ``None`` for a release that takes no
        such control -- then ``prompt_base`` is sent alone.
    floats_in_reading_order
        Whether the model emits pictures and captions where they belong.  v1.1
        emits them at the end of the page, so they have to be put back; v1.2
        and later do not.
    """

    name: str
    model_path: str | None
    markers: tuple[str, ...] = ()
    prompt_base: str = PROMPT_BASE
    text_in_pic_tokens: tuple[str, str] | None = TEXT_IN_PIC_TOKENS
    floats_in_reading_order: bool = True

    def task_prompt(self, *, text_in_pic: bool = False) -> str:
        """The full prompt string for this release."""
        if self.text_in_pic_tokens is None:
            return self.prompt_base
        predict, do_not = self.text_in_pic_tokens
        return f"{self.prompt_base}{predict if text_in_pic else do_not}"

    def resolved_model_path(self, override: str | None = None) -> str:
        """The weights to load, preferring an explicit override."""
        path = override or self.model_path
        if path is None:
            msg = f"Nemotron-Parse {self.name} has no model path recorded; pass model_path= explicitly."
            raise ValueError(msg)
        return path


V1_1 = NemotronParseProfile(
    name="v1.1",
    # Not recorded: this pipeline has only ever been pointed at v1.1 weights by
    # explicit path, so there is no canonical id here to copy.
    model_path=None,
    markers=("v1.1",),
    # The text-in-pic control is documented as v1.2+.  The token is still sent,
    # because that is what this pipeline has always done and dropping it would
    # be a behaviour change made on a guess.
    floats_in_reading_order=False,
)

V1_2 = NemotronParseProfile(
    name="v1.2",
    model_path="nvidia/NVIDIA-Nemotron-Parse-v1.2",
    markers=("v1.2",),
)

#: The release assumed when the caller names none.
DEFAULT_PROFILE = V1_2

_PROFILES: dict[str, NemotronParseProfile] = {p.name: p for p in (V1_1, V1_2)}


def register_profile(profile: NemotronParseProfile) -> None:
    """Add or replace a release profile."""
    _PROFILES[profile.name] = profile


def known_versions() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def _default() -> NemotronParseProfile:
    """The default release, honouring a profile registered over it."""
    return _PROFILES.get(DEFAULT_PROFILE.name, DEFAULT_PROFILE)


def _by_model_path(model_path: str) -> NemotronParseProfile | None:
    """The release a path names, or ``None`` when it names none.

    A path that matches more than one release names none of them: choosing by
    registry order would make the answer depend on which profile happened to be
    registered first.
    """
    matched = [p for p in _PROFILES.values() if any(marker in model_path for marker in p.markers)]
    if len(matched) == 1:
        return matched[0]
    if matched:
        msg = (
            f"Model path {model_path!r} matches more than one Nemotron-Parse release "
            f"({', '.join(sorted(p.name for p in matched))}). Pass parse_version= to say which."
        )
        raise ValueError(msg)
    return None


def resolve(version: str | None = None, model_path: str | None = None) -> NemotronParseProfile:
    """The profile to run with, from a version name, a model path, or neither.

    Precedence: a named version wins; failing that the model path is matched
    against each release's markers; failing that the default release is used.
    An unrecognised version or path is taken at face value -- the caller is
    saying the output contract is unchanged -- and logged, so a release that
    quietly changed the contract shows up as a line in the job log rather than
    as silently mangled output.
    """
    if version is not None:
        known = _PROFILES.get(version)
        if known is not None:
            return replace(known, model_path=model_path or known.model_path)
        if model_path is None:
            msg = (
                f"Unknown Nemotron-Parse version {version!r} and no model_path given. "
                f"Known versions: {', '.join(known_versions())}. Pass model_path= to use "
                f"{_default().name}'s output contract with other weights, or call "
                f"register_profile() to describe what {version!r} actually does."
            )
            raise ValueError(msg)
        default = _default()
        logger.info(
            f"Nemotron-Parse version {version!r} is not registered; assuming the "
            f"{default.name} output contract for {model_path}. "
            f"Call register_profile() if it differs."
        )
        return replace(default, name=version, model_path=model_path, markers=())

    if model_path is not None:
        matched = _by_model_path(model_path)
        if matched is not None:
            return replace(matched, model_path=model_path)
        default = _default()
        logger.info(
            f"No Nemotron-Parse version recognised in {model_path!r}; assuming the {default.name} output contract."
        )
        return replace(default, model_path=model_path, markers=())

    return _default()
