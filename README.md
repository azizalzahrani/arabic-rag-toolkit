# أدوات RAG العربية

[![Tests](https://github.com/azizalzahrani/arabic-rag-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/azizalzahrani/arabic-rag-toolkit/actions/workflows/tests.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## نظرة عامة
مجموعة أدوات شاملة لبناء أنظمة Retrieval-Augmented Generation (RAG) متخصصة في معالجة النصوص العربية بشكل احترافي. تحل هذه الأداة المشاكل الرئيسية التي تواجه أنظمة RAG التقليدية عند التعامل مع اللغة العربية مثل التقطيع الذكي للنصوص، والبحث في الكلمات ذات التشكيل، ومعالجة الأحرف من اليمين لليسار (RTL).

### المميزات الرئيسية
- **تقطيع ذكي للنصوص العربية**: معالجة التصريفات والبادئات واللواحق العربية بكفاءة
- **تطبيع النصوص العربية**: إزالة التشكيل، توحيد أشكال الألف، ومعالجة التطويل
- **نماذج تضمين عربية**: دعم نماذج متخصصة مثل CAMeL وAraBART والنماذج متعددة اللغات
- **أنظمة متعددة الوكلاء**: أدوات مدمجة لتنسيق أدوار البحث والتحقق والكتابة
- **مرونة في اختيار النماذج**: دعم OpenAI و Anthropic والنماذج المحلية
- **تشغيل محلي افتراضي**: استرجاع وإجابة محليان بدون الحاجة إلى مفاتيح API عند البداية
- **قواعد بيانات متعددة**: دعم الذاكرة المحلية و FAISS و ChromaDB
- **أمثلة عملية**: أمثلة حقيقية تطبق على وثائق سعودية ونظام معالجة متكامل

---

# Arabic RAG Toolkit

Arabic-first building blocks for retrieval, chunking, normalization, and answer generation with a local-first default path that works before you wire in hosted AI services.

## Overview
A comprehensive suite of tools for building Retrieval-Augmented Generation (RAG) systems specifically optimized for Arabic text processing. This toolkit solves critical challenges faced by traditional RAG systems when handling Arabic: intelligent text chunking, diacritic-aware search, and proper right-to-left (RTL) text handling.

### Key Features
- **Arabic-Aware Text Chunking**: Intelligently handles Arabic morphology, prefixes, and suffixes
- **Arabic Text Normalization**: Removes diacritics, normalizes alef variants, and handles tatweel
- **Arabic Embedding Models**: Supports CAMeL, AraBART, and multilingual embedding models
- **Multi-Agent Utilities**: Built-in research, validation, and writing agents
- **Model Flexibility**: Support for OpenAI, Anthropic, and local LLMs
- **Local-First Defaults**: Works without API keys on day one using local fallbacks
- **Multiple Vector Stores**: In-memory, FAISS, and ChromaDB support
- **Practical Examples**: Real-world examples with Saudi regulatory documents

## Why This Repo Exists

- Most general-purpose RAG demos ignore Arabic normalization and chunking details.
- New users should be able to run the project locally before configuring external APIs.
- Hosted providers and external vector stores should be optional upgrades, not installation blockers.

---

## البدء السريع | Quick Start

### المتطلبات | Requirements
- Python 3.9+
- pip

### التثبيت | Installation

```bash
git clone https://github.com/azizalzahrani/arabic-rag-toolkit.git
cd arabic-rag-toolkit
pip install .
```

### النشر من PyPI | PyPI Install

After the first PyPI release:

```bash
pip install arabic-rag-toolkit
```

### إعداد البيئة | Environment Setup

```bash
cp .env.example .env
# Optional: edit .env if you want to use OpenAI / Anthropic / Chroma / FAISS
```

### التثبيت مع الإضافات | Optional Extras

```bash
# Development tools
pip install -e ".[dev]"

# Sentence-transformers embeddings
pip install ".[embeddings]"

# OpenAI + Chroma example stack
pip install ".[openai,chroma,embeddings]"
```

---

## أمثلة الاستخدام | Usage Examples

### مثال 1: نظام RAG بسيط | Simple RAG System

```python
from arabic_rag.pipeline import ArabicRAGPipeline

# إعداد خط أنابيب RAG يعمل محلياً بدون مفاتيح API
pipeline = ArabicRAGPipeline(
    vector_store="memory",
    llm_provider="local"
)

# إضافة وثائق
documents = [
    "نظام الشركات السعودي ينص على أن رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي",
    "يجب أن يكون لدى الشركة مجلس إدارة يتكون من ثلاثة أعضاء على الأقل",
    "للمساهمين الحق في حضور الجمعية العامة والتصويت على القرارات"
]
pipeline.add_documents(documents)

# البحث والاسترجاع
results = pipeline.retrieve("كم هو الحد الأدنى لرأس مال الشركة المساهمة؟")
answer = pipeline.generate_answer(results, "كم هو الحد الأدنى لرأس مال الشركة المساهمة؟")
print(f"الإجابة: {answer}")
```

### مثال 2: نظام متعدد الوكلاء | Multi-Agent RAG

```python
from arabic_rag.agents.multi_agent_crew import setup_crew
from arabic_rag.pipeline import ArabicRAGPipeline

# إعداد خط الأنابيب الأساسي
pipeline = ArabicRAGPipeline(
    vector_store="memory",
    llm_provider="local",
    verbose=True,
)

# إعداد فريق الوكلاء
crew = setup_crew(pipeline)

# تنفيذ مهمة البحث
task = "ابحث عن المتطلبات القانونية لتسجيل شركة جديدة في السعودية وقدم ملخصاً شاملاً"
result = crew.execute_task(task, top_k=3)
print(result["final_answer"])
```

### مثال 3: معالجة النصوص العربية | Arabic Text Processing

```python
from arabic_rag.preprocessor import ArabicTextPreprocessor
from arabic_rag.chunker import ArabicTextChunker

# تطبيع النص
preprocessor = ArabicTextPreprocessor()
text = "اَلسَّلامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكاتُهُ"
normalized = preprocessor.normalize(text)
print(f"النص المطبّع: {normalized}")  # السلام عليكم ورحمة الله وبركاته

# تقطيع ذكي
chunker = ArabicTextChunker()
document = "القانون التجاري السعودي يحدد الأطر القانونية لجميع العمليات التجارية. المادة الأولى تنص على حقوق التجار..."
chunks = chunker.chunk(document)
for i, chunk in enumerate(chunks):
    print(f"الجزء {i+1}: {chunk}")
```

---

## بنية المشروع | Project Structure

```
arabic-rag-toolkit/
├── .github/
│   └── workflows/
│       └── tests.yml         # GitHub Actions CI
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .gitignore
├── .env.example
├── arabic_rag/
│   ├── __init__.py
│   ├── chunker.py              # تقطيع النصوص العربية
│   ├── embeddings.py           # نماذج التضمين العربية
│   ├── retriever.py            # استرجاع الوثائق
│   ├── generator.py            # توليد الإجابات
│   ├── pipeline.py             # خط أنابيب RAG متكامل
│   ├── preprocessor.py         # تطبيع النصوص العربية
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── research_agent.py   # وكيل البحث
│   │   ├── validator_agent.py  # وكيل التحقق
│   │   ├── writer_agent.py     # وكيل الكتابة
│   │   └── multi_agent_crew.py # تنسيق فريق الوكلاء
│   └── utils/
│       ├── __init__.py
│       └── arabic_utils.py     # أدوات عربية مساعدة
├── examples/
│   ├── basic_rag.py            # مثال RAG بسيط
│   ├── multi_agent_rag.py      # مثال متعدد الوكلاء
│   └── saudi_regulations.py    # معالجة الوثائق السعودية
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_preprocessor.py
│   └── test_pipeline.py
└── docs/
    └── ARCHITECTURE_AR.md      # التوثيق المعماري
```

---

## المعمارية | Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Query (Arabic)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Arabic Preprocessor      │
                │  (Normalize, Remove Tash.) │
                └────────────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │  Arabic Chunker  │      │ Arabic Embeddings│
         │ (RTL-Aware)      │      │ (CAMeL/AraBART)  │
         └──────────┬───────┘      └────────┬─────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                      ┌──────────────────┐
                      │  Vector Store    │
                      │ (FAISS/ChromaDB) │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │   Retriever      │
                      └────────┬─────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │ Researcher │      │ Validator  │      │   Writer   │
    │   Agent    │      │   Agent    │      │   Agent    │
    └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
          │                   │                    │
          └───────────────────┼────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   LLM Response   │
                    │ (OpenAI/Anthropic)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Final Answer    │
                    │    (Arabic)      │
                    └──────────────────┘
```

---

## البيئة والإعدادات | Configuration

### `.env.example`
```bash
# LLM APIs
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Vector Store Choice
VECTOR_STORE=memory  # Options: memory, chroma, faiss

# LLM Provider
LLM_PROVIDER=local  # Options: local, openai, anthropic

# Model Names
OPENAI_MODEL=your-openai-model-name
ANTHROPIC_MODEL=your-anthropic-model-name

# Vector Store Path
VECTOR_STORE_PATH=./data/vector_store

# Chunk Settings
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

---

## المتطلبات | Requirements

- Python 3.9+
- `numpy` for the core local/offline path
- `sentence-transformers` for real embedding models
- `chromadb` or `faiss-cpu` for external vector stores
- `openai` or `anthropic` only if you want hosted LLM generation

Package metadata and optional extras are defined in `pyproject.toml`.

---

## المساهمة | Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on our code of conduct and the process for submitting pull requests.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/azizalzahrani/arabic-rag-toolkit.git
cd arabic-rag-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt

# Run tests
pytest -v
```

### Releases

Maintainer release steps are documented in [RELEASING.md](RELEASING.md).

---

## الترخيص | License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## التواصل والدعم | Support & Contact

- **Issues**: [GitHub Issues](https://github.com/azizalzahrani/arabic-rag-toolkit/issues)
- **Author**: [@azizalzahrani](https://github.com/azizalzahrani)
- **Questions**: Open a GitHub issue with a minimal reproduction and expected behavior

---

## الشكر والاعتراف | Acknowledgments

- CAMeL Lab for Arabic NLP research
- Hugging Face for transformer models
- All contributors and users

---

## خارطة الطريق | Roadmap

- [ ] Arabic-specific fine-tuned embedding models
- [ ] Support for dialects (Egyptian, Levantine, Gulf)
- [ ] Integration with more Arabic NLP libraries (Farasa, RichArabic)
- [ ] Multilingual RAG support
- [ ] Web UI for document management
- [ ] Benchmark suite for Arabic RAG systems

---

**Version**: 0.1.0
**Last Updated**: 2026-03-22
**Status**: Active Development
