from setuptools import setup, find_packages

setup(
    name="minrag",
    version="0.1.0",
    description="A lightweight RAG library built from scratch — no LangChain, no ChromaDB",
    author="vamsikumarGedela",
    packages=find_packages(),
    install_requires=[
        "pypdf>=4.0.0",
        "sentence-transformers>=3.0.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.24.0",
        "openai>=1.40.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
)
