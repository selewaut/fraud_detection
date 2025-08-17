#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    install_requires=requirements,
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords="fraud_detector",
    name="fraud_detector",
    packages=find_packages(include=["fraud_detector", "fraud_detector,.*"]),
    test_suite="tests",
    version="0.0.1",
    zip_safe=False,
)
