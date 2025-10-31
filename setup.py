"""
DFS Meta-Optimizer Setup
Professional-grade Daily Fantasy Sports lineup optimizer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="dfs-meta-optimizer",
    version="8.0.0",
    author="DFS Meta-Optimizer Contributors",
    author_email="",
    description="Professional-grade DFS lineup optimizer with AI and advanced mathematics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dfs-meta-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "ai": ["anthropic>=0.34.0"],
        "math": ["scipy>=1.10.0"],
        "api": ["requests>=2.31.0", "schedule>=1.2.0", "python-dotenv>=1.0.0"],
        "monitoring": ["psutil>=5.9.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "anthropic>=0.34.0",
            "scipy>=1.10.0",
            "requests>=2.31.0",
            "schedule>=1.2.0",
            "python-dotenv>=1.0.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dfs-optimizer=app:main",
        ],
    },
    include_package_data=True,
    keywords="dfs daily-fantasy-sports optimizer ai machine-learning",
    project_urls={
        "Documentation": "https://github.com/yourusername/dfs-meta-optimizer/blob/main/INTEGRATION_GUIDE.md",
        "Bug Reports": "https://github.com/yourusername/dfs-meta-optimizer/issues",
        "Source": "https://github.com/yourusername/dfs-meta-optimizer",
    },
)
