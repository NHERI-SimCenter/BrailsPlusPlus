"""Setup file for the `BRAILS++` package."""

from setuptools import setup, find_packages
import brails

import glob
import os
import subprocess
import sys

def install_torch():
    try:
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

# Call the function to ensure torch is installed
install_torch()

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

'''
(THIS SECTION IS FOR CUDA GPU USERS ONLY)
Before running pip install -e ., check the following:
- Set CUDA_HOME variable to the location of your cuda server
- GCC version must be >9 (because msdeformattn module explicitly requires it)

Install Error
- If there is problem between Pytorch and CUDA, check nvcc version. CUDA_HOME can be different to nvcc
- If CUDA server > 11.7, cpp_extension error will arise. Got to cpp_extension.py(in conda env folder), search for CUDA_GCC_VERSIONS, copy the setting of 11.7 to your version
Some useful links 
- {conda_root}/envs/{conda_env_name}/lib/{python3.xx}/site-packages/torch/utils/cpp_extension.py
- https://github.com/vllm-project/vllm/issues/1453 (Pytorch & CUDA incompatability)
- https://github.com/IDEA-Research/GroundingDINO (Setting CUDA_HOME)
'''

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "brails/filters/house_view","groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, 'groundingdino')
os.makedirs(temp_dir, exist_ok = True) #temp dir to allow groundingdino installation to copy file(error might trigger if not set)

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
        'supervision',
        'opencv-python',
        'addict',
        'yapf',
        'timm',
        'transformers',
        'pycocotools'
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
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
)

# devguide.python.org: Status of Python versions:
# Use to determine when to drop/add  support
# https://devguide.python.org/versions/
