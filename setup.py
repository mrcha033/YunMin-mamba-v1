#!/usr/bin/env python3
"""
Hardware-Data-Parameter Co-Design Framework for State Space Models
Setup script for package installation and distribution
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("src", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="hardware-data-parameter-codesign",
    version=get_version(),
    author="Yunmin Cha",
    author_email="yunmin.cha@example.com",  # Update with actual email
    description="Hardware-Data-Parameter Co-Design Framework for State Space Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yunmin-cha/hardware-data-parameter-codesign",  # Update with actual repo
    project_urls={
        "Bug Reports": "https://github.com/yunmin-cha/hardware-data-parameter-codesign/issues",
        "Source": "https://github.com/yunmin-cha/hardware-data-parameter-codesign",
        "Documentation": "https://github.com/yunmin-cha/hardware-data-parameter-codesign/blob/main/README.md",
    },
    packages=find_packages(exclude=["tests", "experiments", "checkpoints", "results"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
            "fairscale>=0.4.13",
        ],
        "optimization": [
            "triton>=2.0.0",
            "flash-attn>=2.0.0",
        ],
        "all": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "deepspeed>=0.9.0",
            "fairscale>=0.4.13",
            "triton>=2.0.0",
            "flash-attn>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "codesign-train=scripts.pretrain:main",
            "codesign-validate=scripts.run_full_scale_validation:main",
            "codesign-demo=demo_full_scale_validation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "deep learning",
        "state space models",
        "mamba",
        "hardware optimization",
        "parameter efficiency",
        "sparsity",
        "co-design",
        "machine learning",
        "transformers",
        "natural language processing",
    ],
) 