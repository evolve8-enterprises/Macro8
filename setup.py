"""
setup.py — Macro8 subnet package
"""
from setuptools import setup, find_packages

setup(
    name="macro8-subnet",
    version="1.0.0",
    description="Decentralized alpha research network on Bittensor",
    long_description=open("README_DEPLOYMENT.md").read(),
    long_description_content_type="text/markdown",
    author="Macro8 Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "bittensor>=10.0.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "yfinance>=0.2.30",
        "requests>=2.31",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "macro8-miner=macro8_subnet.neurons.miner:main",
            "macro8-validator=macro8_subnet.neurons.validator:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
