"""
أدوات RAG العربية - Arabic-first Retrieval-Augmented Generation Toolkit

A comprehensive toolkit for building RAG systems optimized for Arabic text processing.
Features Arabic-aware text chunking, normalization, embeddings, and multi-agent orchestration.

العربية:
    مجموعة أدوات شاملة لبناء أنظمة Retrieval-Augmented Generation متخصصة في
    معالجة النصوص العربية مع تقطيع ذكي وتطبيع نصوص وتضمين عربي.

English:
    A comprehensive toolkit for building RAG systems optimized for Arabic text processing
    with intelligent chunking, normalization, and Arabic embeddings.
"""

__version__ = "0.1.0"
__author__ = "Aziz Alzahrani"
__email__ = "aziz@example.com"

from arabic_rag.preprocessor import ArabicTextPreprocessor
from arabic_rag.chunker import ArabicTextChunker
from arabic_rag.embeddings import ArabicEmbeddings
from arabic_rag.retriever import ArabicRetriever
from arabic_rag.generator import ArabicResponseGenerator
from arabic_rag.pipeline import ArabicRAGPipeline

__all__ = [
    "ArabicTextPreprocessor",
    "ArabicTextChunker",
    "ArabicEmbeddings",
    "ArabicRetriever",
    "ArabicResponseGenerator",
    "ArabicRAGPipeline",
]
