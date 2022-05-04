# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

version = '0.2.1.dev0'
sha = 'Unknown'
src_folder = 'holocron'
package_index = 'pylocron'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    if sha != 'Unknown':
        version += '+' + sha[:7]
print(f"Building wheel {package_index}-{version}")

with open(cwd.joinpath(src_folder, 'version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md') as f:
    readme = f.read()


_deps = [
    "torch>=1.9.1",
    "torchvision>=0.10.1",
    "tqdm>=4.1.0",
    "numpy>=1.17.2",
    "fastprogress>=1.0.0",
    "matplotlib>=3.0.0",
    "Pillow>=8.4.0",  # cf. https://github.com/pytorch/vision/issues/4934
    "huggingface-hub>=0.4.0",
    # Testing
    "pytest>=5.3.2",
    "coverage>=4.5.4",
    # Quality
    "flake8>=3.9.0",
    "isort>=5.7.0",
    "mypy>=0.812",
    "pydocstyle>=6.0.0",
    # Docs
    "sphinx<=3.4.3",
    "sphinx-rtd-theme==0.4.3",
    "sphinxemoji>=0.1.8",
    "sphinx-copybutton>=0.3.1",
    "docutils<0.18",
    "recommonmark>=0.7.1",
    "sphinx-markdown-tables>=0.0.15",
    "Jinja2<3.1",  # cf. https://github.com/readthedocs/readthedocs.org/issues/9038
]

# Borrowed from https://github.com/huggingface/transformers/blob/master/setup.py
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


install_requires = [
    deps["torch"],
    deps["torchvision"],
    deps["tqdm"],
    deps["numpy"],
    deps["fastprogress"],
    deps["matplotlib"],
    deps["Pillow"],
    deps["huggingface-hub"],
]

extras = {}

extras["testing"] = deps_list(
    "pytest",
    "coverage",
)

extras["quality"] = deps_list(
    "flake8",
    "isort",
    "mypy",
    "pydocstyle",
)

extras["docs"] = deps_list(
    "sphinx",
    "sphinx-rtd-theme",
    "sphinxemoji",
    "sphinx-copybutton",
    "docutils",
    "recommonmark",
    "sphinx-markdown-tables",
    "Jinja2",
)

extras["dev"] = (
    extras["testing"]
    + extras["quality"]
    + extras["docs"]
)


setup(
    name=package_index,
    version=version,
    author='François-Guillaume Fernandez',
    author_email='fg-feedback@protonmail.com',
    description='Modules, operations and models for computer vision in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/Holocron',
    download_url='https://github.com/frgfm/Holocron/tags',
    license='Apache',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['pytorch', 'deep learning', 'vision', 'models'],
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
    package_data={'': ['LICENSE']},
)
