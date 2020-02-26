# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for strawberryfields.api.job
"""
import pytest

from strawberryfields.api import InvalidJobOperationError, Job, JobStatus

# pylint: disable=bad-continuation,no-self-use

pytestmark = pytest.mark.api


class TestJob:
    """Tests for the ``Job`` class."""

    def test_incomplete_job_raises_on_result_access(self, connection):
        """Tests that `job.result` raises an error for an incomplete job."""
        job = Job("abc", status=JobStatus.QUEUED, connection=connection)

        with pytest.raises(
            AttributeError,
            match="The result is undefined for jobs that are not completed",
        ):
            _ = job.result

    def test_final_job_raises_on_cancel(self, connection):
        """Tests that `job.cancel()` raises an error for a complete, failed, or
        aleady cancelled job."""
        job = Job("abc", status=JobStatus.FAILED, connection=connection)

        with pytest.raises(
            InvalidJobOperationError, match="A failed job cannot be cancelled"
        ):
            job.cancel()
