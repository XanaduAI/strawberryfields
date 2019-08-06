# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Starship
========

**Module name:** :mod:`strawberryfields.starship`

.. currentmodule:: strawberryfields.starship

This module provides classes to interact with the Starship API, and for submitting remote jobs via
APIClient.
"""

from .backends.base import BaseBackend
from strawberryfields.api_client import APIClient, Job, JobExecutionError
from strawberryfields.configuration import DEFAULT_CONFIG
from strawberryfields.io import to_blackbird
from time import sleep

from multiprocessing import Process, Queue


class Starship:
    """
    Starship quantum program executor engine.

    Executes :class:`.Program` instances on the chosen remote backend, and makes
    the results available via :class:`.Result`.

    Args:
        polling_delay_seconds (float): The number of seconds to wait when polling the server
    """

    SUPPORTED_BACKENDS = ("chip0",)

    def __init__(self, polling_delay_seconds=1, **kwargs):
        class Chip0Backend(BaseBackend):
            circuit_spec = "chip0"

        self.backend = Chip0Backend()

        api_client_params = {k: v for k, v in kwargs.items() if k in DEFAULT_CONFIG["api"].keys()}
        self.client = APIClient(**api_client_params)
        self.polling_delay_seconds = polling_delay_seconds
        self.jobs = []
        self.processes = []
        self.complete_jobs_queue = Queue()
        self.failed_jobs_queue = Queue()

        self.complete_jobs = []
        self.failed_jobs = []

    def __str__(self):
        return self.__class__.__name__ + "({})".format(self.backend_name)

    def merge_jobs(self):
        """
        Process jobs that have been put into a queue.
        """
        for _ in range(self.complete_jobs_queue.qsize()):
            self.complete_jobs.append(self.complete_jobs_queue.get())

        for _ in range(self.failed_jobs_queue.qsize()):
            self.failed_jobs.append(self.failed_jobs_queue.get())

    def _create_job(self, job_content):
        """
        Create a Job instance based on job_content, and send the job to the API. Append to list
        of jobs.

        Args:
            job_content (str): the Blackbird code to execute

        Returns:
            (strawberryfields.api_client.Job): a Job instance referencing the queued job
        """
        job = Job(client=self.client)
        job.manager.create(circuit=job_content)
        return job

    def _poll_for_job_results(self, job):
        """
        Regularly fetch updated job statuses from server.
        """
        while job.is_processing:
            job.reload()
            sleep(self.polling_delay_seconds)

        if job.is_complete:
            self.complete_jobs_queue.put(job)
        else:
            self.failed_jobs_queue.put(job)
            raise JobExecutionError("Job execution failed. Please try again.")

    def _compile_program(self, program, *, compile_options=None):
        """
        Compiles the program using compile_options and the given backend.

        Args:
            program (Program): the program to compile
            compile_options (dict): options to use when compiling program (passed to compile)

        Returns:
            A compiled, locked Program instance.
        """
        compile_options = compile_options or {}
        # TODO: do we need to provide default shots??
        target = self.backend.circuit_spec
        program = program.compile(target, **compile_options)
        program.lock()
        return program

    def _send_program_as_job(self, program):
        """
        Converts a program into Blackbird code as a string and creates a job using that code.
        Appends the new Job instance to the list of jobs.

        Args:
            program (Program): a compiled Program instance to send to the Starship

        Returns:
            a Job instance
        """
        job_content = to_blackbird(program, version="1.0").serialize()
        job = self._create_job(job_content)
        self.jobs.append(job)
        return job

    def _process_job(self, job, asynchronous=False):
        """
        Given a particular Job instance, creates a polling process and adds it to the queue.
        """

        # TODO: when batching support is added, this will no longer be necessary
        process = Process(target=self._poll_for_job_results, args=(job,))
        process.start()
        self.processes.append(process)
        if not asynchronous:
            process.join()
        return process

    def run(self, program, *, compile_options=None, **kwargs):
        """
        Run a single program synchronously.
        """
        program = self._compile_program(program, compile_options=compile_options)
        job = self._send_program_as_job(program)
        self._process_job(job)
        # This is hacky right now but needed in order to refresh the Job instance
        # even though this extra reload is not necessary
        job.reload()
        job.result.manager.get()
        return job

    def run_many(self, programs, *, compile_options=None, **kwargs):
        """
        Run many programs asynchronously.
        """
        for program in programs:
            program = self._compile_program(program, compile_options=compile_options)
            job = self._send_program_as_job(program)
            self._process_job(job, asynchronous=True)

    def run_async(self, program, *, compile_options=None, **kwargs):
        """
        Run a single program asynchronously.
        """
        self.run_many([program], compile_options=compile_options, **kwargs)
