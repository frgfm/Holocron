#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2019 Foodvisor
# Written by François-Guillaume Fernandez
# --------------------------------------------------------

"""
Package installation setup
"""

import os
import sys

from setuptools import setup


if sys.argv[-1] == 'publish':
    os.system('python3 setup.py sdist upload')
    sys.exit()

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='holocron',
    version='0.1.0',
    author='François-Guillaume Fernandez',
    description='Modules, operations and models for computer vision in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/Holocron',
    packages=['holocron'],
    package_data={'': ['LICENSE']},
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    license='MIT',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='pytorch deep learning vision models'
)