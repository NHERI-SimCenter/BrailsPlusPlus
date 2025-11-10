"""Setup file for the BRAILS++ package."""

import re
from setuptools import setup

with open('brails/__init__.py', 'r', encoding='utf-8') as f:
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                        f.read(), re.M).group(1)

setup(
    version=version,
)

