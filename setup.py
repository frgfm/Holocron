#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Package installation setup
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

version = '0.1.0a0'
sha = 'Unknown'
package_name = 'holocron'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))

def write_version_file():
    version_path = os.path.join(cwd, 'holocron', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


if sys.argv[-1] == 'publish':
    os.system('python3 setup.py sdist upload')
    sys.exit()

write_version_file()

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name=package_name,
    version=version,
    author='François-Guillaume Fernandez',
    description='Modules, operations and models for computer vision in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/Holocron',
    packages=find_packages(exclude=('test',)),
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
    keywords=['pytorch', 'deep learning', 'vision', 'models']
)