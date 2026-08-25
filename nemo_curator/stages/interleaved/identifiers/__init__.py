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

"""Canonical document identifiers for interleaved corpora.

``sample_id`` is whatever the producing corpus called a document.  These are
the stages that turn it into the identifier the outside world uses, so that a
join is a column comparison rather than a transformation every consumer has to
remember.
"""

from nemo_curator.stages.interleaved.identifiers.arxiv import canon, year_month
from nemo_curator.stages.interleaved.identifiers.stages import AddArxivIdStage

__all__ = ["AddArxivIdStage", "canon", "year_month"]
