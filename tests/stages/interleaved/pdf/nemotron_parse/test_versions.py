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

"""Tests for the Nemotron-Parse release profiles.

The point of this module is that moving to a new release is one argument.
These tests pin the two things that makes true: that the behaviour differences
between releases live here and nowhere else, and that an unregistered release
is taken at the caller's word rather than guessed at.
"""

from __future__ import annotations

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse import versions

# ---- what each recorded release does ----------------------------------------


class TestRecordedReleases:
    def test_v1_2_is_the_default(self) -> None:
        assert versions.DEFAULT_PROFILE.name == "v1.2"
        assert versions.resolve().model_path == "nvidia/NVIDIA-Nemotron-Parse-v1.2"

    def test_v1_1_emits_floats_at_the_end_of_the_page_and_v1_2_does_not(self) -> None:
        """The one behavioural difference the pipeline has to compensate for."""
        assert versions.resolve("v1.1", model_path="/weights/np-v1.1").floats_in_reading_order is False
        assert versions.resolve("v1.2").floats_in_reading_order is True

    def test_the_prompt_carries_the_text_in_pic_token(self) -> None:
        profile = versions.resolve("v1.2")

        assert profile.task_prompt(text_in_pic=False).endswith("<predict_no_text_in_pic>")
        assert profile.task_prompt(text_in_pic=True).endswith("<predict_text_in_pic>")

    def test_a_release_with_no_such_control_gets_the_base_prompt_alone(self) -> None:
        profile = versions.NemotronParseProfile(name="x", model_path="x", text_in_pic_tokens=None)

        assert profile.task_prompt(text_in_pic=True) == versions.PROMPT_BASE


# ---- resolution -------------------------------------------------------------


class TestResolve:
    def test_a_named_version_wins_over_the_path(self) -> None:
        profile = versions.resolve("v1.1", model_path="/weights/whatever")

        assert profile.name == "v1.1"
        assert profile.model_path == "/weights/whatever"

    def test_an_unnamed_version_is_recognised_from_the_path(self) -> None:
        assert versions.resolve(model_path="nvidia/NVIDIA-Nemotron-Parse-v1.1").name == "v1.1"

    def test_an_unrecognisable_path_falls_back_to_the_default_contract(self) -> None:
        """A local checkout says nothing about which release it holds, and the
        pipeline has always assumed the current one."""
        profile = versions.resolve(model_path="/scratch/my-finetune")

        assert profile.name == versions.DEFAULT_PROFILE.name
        assert profile.model_path == "/scratch/my-finetune"
        assert profile.floats_in_reading_order is True

    def test_nothing_at_all_is_the_default_release(self) -> None:
        assert versions.resolve().name == versions.DEFAULT_PROFILE.name

    def test_a_path_naming_two_releases_names_neither(self) -> None:
        """Picking one by registry order would make the answer depend on which
        profile happened to be registered first."""
        with pytest.raises(ValueError, match="matches more than one"):
            versions.resolve(model_path="/ckpt/compare-v1.1-against-v1.2/step-4000")

    def test_a_profile_registered_over_the_default_is_the_one_used(self) -> None:
        original = versions._PROFILES[versions.DEFAULT_PROFILE.name]
        versions.register_profile(
            versions.NemotronParseProfile(
                name=versions.DEFAULT_PROFILE.name,
                model_path="org/patched-default",
                markers=original.markers,
            )
        )
        try:
            assert versions.resolve().model_path == "org/patched-default"
        finally:
            versions.register_profile(original)


# ---- moving to a new release ------------------------------------------------


class TestNewReleases:
    def test_an_unregistered_version_with_weights_is_taken_at_the_caller_s_word(self) -> None:
        """ "Same output contract" is a claim only the caller can make; the
        module records it rather than refusing."""
        profile = versions.resolve("v2.0", model_path="nvidia/NVIDIA-Nemotron-Parse-v2.0")

        assert profile.name == "v2.0"
        assert profile.model_path == "nvidia/NVIDIA-Nemotron-Parse-v2.0"
        assert profile.floats_in_reading_order is versions.DEFAULT_PROFILE.floats_in_reading_order

    def test_an_unregistered_version_without_weights_is_refused(self) -> None:
        """Guessing a HuggingFace id is worse than asking for one."""
        with pytest.raises(ValueError, match="Unknown Nemotron-Parse version"):
            versions.resolve("v2.0")

    def test_a_release_that_changed_the_contract_can_be_described(self) -> None:
        profile = versions.NemotronParseProfile(
            name="test-release",
            model_path="org/test-release",
            markers=("test-release",),
            floats_in_reading_order=False,
        )
        versions.register_profile(profile)
        try:
            assert versions.resolve("test-release").floats_in_reading_order is False
            assert versions.resolve(model_path="org/test-release").name == "test-release"
            assert "test-release" in versions.known_versions()
        finally:
            versions._PROFILES.pop("test-release")

    def test_a_profile_that_names_no_weights_insists_on_being_given_some(self) -> None:
        with pytest.raises(ValueError, match="no model path recorded"):
            versions.V1_1.resolved_model_path()


# ---- the profile has to reach the stages that act on it ---------------------


class TestTheStagesHonourTheProfile:
    """Naming a release and loading a different release's weights is the exact
    failure this module exists to prevent, so it is pinned here."""

    @staticmethod
    def _reader(**kwargs: object):  # noqa: ANN205
        from nemo_curator.stages.interleaved.pdf.nemotron_parse import NemotronParsePDFReader

        # The manifest is never opened: decompose() only builds the stages.
        return NemotronParsePDFReader(manifest_path="manifest.jsonl", **kwargs).decompose()

    def test_the_default_release_is_unchanged(self) -> None:
        inference = self._reader()[2]

        assert inference.model_path == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
        assert inference.profile.name == "v1.2"

    def test_naming_a_release_brings_its_weights_with_it(self) -> None:
        profile = versions.NemotronParseProfile(
            name="test-release",
            model_path="org/test-release",
            markers=("test-release",),
            floats_in_reading_order=False,
        )
        versions.register_profile(profile)
        try:
            stages = self._reader(parse_version="test-release")
        finally:
            versions._PROFILES.pop("test-release")

        assert stages[2].model_path == "org/test-release"
        # ...and the postprocess stage knows it has to put the floats back.
        assert stages[-1].profile.floats_in_reading_order is False

    def test_the_postprocess_stage_gets_the_profile_from_the_driver(self) -> None:
        """It is resolved once, where ``register_profile`` was called, rather
        than looked up again in a worker process that never saw the registry."""
        stages = self._reader(model_path="nvidia/NVIDIA-Nemotron-Parse-v1.1")

        assert stages[-1].profile is not None
        assert stages[-1].profile.name == "v1.1"

    def test_a_local_checkout_keeps_the_path_it_was_given(self) -> None:
        assert self._reader(model_path="/scratch/my-finetune")[2].model_path == "/scratch/my-finetune"
