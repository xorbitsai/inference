# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="dacvae",
    version="1.0.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="A high-quality general neural audio codec.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yi-Chiao Wu",
    author_email="yichiaowu@meta.com",
    url="https://github.com/facebookresearch/dacvae",
    license_files = ("LICENSE.txt",),
    packages=find_packages(),
    keywords=["audio", "compression", "machine learning"],
    install_requires=[
        "argbind>=0.3.7",
        "descript-audiotools>=0.7.2",
        "einops",
        "huggingface-hub",
        "numpy",
        "torch",
        "torchaudio",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pynvml",
            "psutil",
            "pandas",
            "onnx",
            "onnx-simplifier",
            "seaborn",
            "jupyterlab",
            "pandas",
            "watchdog",
            "pesq",
            "tabulate",
            "encodec",
        ],
    },
)
