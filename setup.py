"""Setup configuration for Arabic RAG Toolkit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arabic-rag-toolkit",
    version="0.1.0",
    author="Aziz Alzahrani",
    author_email="aziz@example.com",
    description="An Arabic-first Retrieval-Augmented Generation (RAG) toolkit using LangChain and CrewAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/azizalzahrani/arabic-rag-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "isort>=5.0",
            "sphinx>=5.0",
        ],
        "faiss": [
            "faiss-cpu>=1.7.4",
        ],
        "all": [
            "faiss-cpu>=1.7.4",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "isort>=5.0",
            "sphinx>=5.0",
        ],
    },
    keywords="arabic nlp rag retrieval-augmented-generation langchain crewai",
    project_urls={
        "Bug Reports": "https://github.com/azizalzahrani/arabic-rag-toolkit/issues",
        "Source": "https://github.com/azizalzahrani/arabic-rag-toolkit",
        "Documentation": "https://github.com/azizalzahrani/arabic-rag-toolkit#readme",
    },
)
