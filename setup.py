#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

from version import __version__ as package_version

project_root = Path(__file__).parent
install_requires = (project_root / "requirements.txt").read_text().splitlines()

setup(
    name="homecook-nlp",
    version=package_version,
    description="A repository for with homecook NLP part.",
    author=["Adam Woch"],
    author_email=["adam.woch4@gmail.com"],
    url="https://github.com/Kwisatzzz/homecook",
    packages=find_packages(),
    install_requires=install_requires,
    long_description=(project_root / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
