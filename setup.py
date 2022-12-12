#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Adaptive DeepJSCC",
    author="Selim F. Yilmaz",
    author_email="",
    url="https://github.com/selimfirat/adaptive-deepjscc",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
