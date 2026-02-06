"""
Setup script for RNS-ROCm Python bindings.

To install:
    pip install .

To install in development mode:
    pip install -e .

To build wheel:
    pip wheel . --no-deps
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DRNS_BUILD_PYTHON=ON",
            "-DRNS_ENABLE_TESTS=OFF",
            "-DRNS_ENABLE_EXAMPLES=OFF",
        ]
        
        # Check if HIP/ROCm is available
        if os.environ.get("RNS_ENABLE_GPU", "OFF").upper() == "ON":
            cmake_args.append("-DRNS_ENABLE_GPU=ON")
        else:
            cmake_args.append("-DRNS_ENABLE_GPU=OFF")

        build_args = ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL if not set
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )

setup(
    name="rns-rocm",
    version="0.2.0",
    author="VesterlundCoder",
    author_email="",
    description="RNS-ROCm: Residue Number System library for exact modular arithmetic",
    long_description=open("../README.md").read() if os.path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/VesterlundCoder/RNS-ROCm",
    packages=find_packages(),
    ext_modules=[CMakeExtension("rns_rocm", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
        ],
        "sympy": [
            "sympy>=1.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="rns residue-number-system modular-arithmetic gpu rocm hip",
)
