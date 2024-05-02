"""
setup.py file of the `brails` package.

"""

from setuptools import setup, find_packages
import brails


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='brails',
    version=brails.__version__,
    description='BRAILS++: Building Regional Asset Inventories at Large Scale',
    long_description=long_description,
    author='NHERI SimCenter',
    author_email='nheri-simcenter@berkeley.edu',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'requests',
        'wget',
        'shapely',
        'tqdm',
        'torch>=1.12',
        'torchvision>0.13',
        'Pillow',
        'matplotlib',
        'seaborn',
        'plotly',
        'ftfy',
        'supervision'
        'opencv-python'
    ],
    extras_require={
        'development': [
            'flake8',
            'pylint',
            'black',
            'pytest',
            'pytest-cov',
            'jupyter',
            'jupytext',
            'sphinx',
            'sphinx-autoapi',
            'nbsphinx',
            'flake8-rst',
            'flake8-rst-docstrings',
        ],
    },
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
    ],
)

# devguide.python.org: Status of Python versions:
# Use to determine when to drop/add  support
# https://devguide.python.org/versions/
