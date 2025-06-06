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
from setuptools import setup, find_packages


with open("strawberryfields/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "networkx>=2.0",
    "numba",
    "numpy>=1.17.4",
    "python-dateutil>=2.8.0",
    "quantum-blackbird>=0.4.0",
    "requests>=2.22.0",
    "scipy>=1.8.0",
    "sympy>=1.5",
    "thewalrus>=0.18.0",
    "toml",
    "urllib3>=1.25.3",
    "quantum-xir>=0.1.1",
    "xanadu-cloud-client>=0.2.1",
]

info = {
    "name": "StrawberryFields",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/StrawberryFields",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {"strawberryfields": ["backends/data/*", "apps/data/feature_data/*",
                                          "apps/data/sample_data/*"]},
    "include_package_data": True,
    "description": "Open source library for continuous-variable quantum computation",
    "long_description": open("README.md", encoding="utf-8").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["strawberryfields"],
    "install_requires": requirements,
    "command_options": {
        "build_sphinx": {"version": ("setup.py", version), "release": ("setup.py", version)}
    },
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
