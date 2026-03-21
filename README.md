# أدوات RAG العربية

## نظرة عامة
مجموعة أدوات شاملة لبناء أنظمة Retrieval-Augmented Generation (RAG) متخصصة في معالجة النصوص العربية بشكل احترافي. تحل هذه الأداة المشاكل الرئيسية التي تواجه أنظمة RAG التقليدية عند التعامل مع اللغة العربية مثل التقطيع الذكي للنصوص، والبحث في الكلمات ذات التشكيل، ومعالجة الأحرف من اليمين لليسار (RTL).

### المميزات الرئيسية
- **تقطيع ذكي للنصوص العربية**: معالجة التصريفات والبادئات واللواحق العربية بكفاءة
- **تطبيع النصوص العربية**: إزالة التشكيل، توحيد أشكال الألف، ومعالجة التطويل
- **نماذج تضمين عربية**: دعم نماذج متخصصة مثل CAMeL وAraBART والنماذج متعددة اللغات
- **أنظمة متعددة الوكلاء**: استخدام CrewAI لإنشاء فريق من الوكلاء الذكيين (محلل، مدقق، كاتب)
- **مرونة في اختيار النماذج**: دعم OpenAI و Anthropic والنماذج المحلية
- **قواعد بيانات متعددة**: دعم FAISS و ChromaDB
- **أمثلة عملية**: أمثلة حقيقية تطبق على وثائق سعودية ونظام معالجة متكامل

---

# Arabic RAG Toolkit

## Overview
A comprehensive suite of tools for building Retrieval-Augmented Generation (RAG) systems specifically optimized for Arabic text processing. This toolkit solves critical challenges faced by traditional RAG systems when handling Arabic: intelligent text chunking, diacritic-aware search, and proper right-to-left (RTL) text handling.

### Key Features
- **Arabic-Aware Text Chunking**: Intelligently handles Arabic morphology, prefixes, and suffixes
- **Arabic Text Normalization**: Removes diacritics, normalizes alef variants, and handles tatweel
- **Arabic Embedding Models**: Supports CAMeL, AraBART, and multilingual embedding models
- **Multi-Agent System**: Leverages CrewAI for orchestrating research, validation, and writing agents
- **Model Flexibility**: Support for OpenAI, Anthropic, and local LLMs
- **Multiple Vector Stores**: FAISS and ChromaDB support
- **Production-Ready Examples**: Real-world examples with Saudi regulatory documents

---

## البدء السريع | Quick Start

### المتطلبات | Requirements
- Python 3.9+
- pip

### التثبيت | Installation

```bash
git clone https://github.com/azizalzahrani/arabic-rag-toolkit.git
cd arabic-rag-toolkit
pip install -e .
```

### إعداد البيئة | Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

---

## أمثلة الاستخدام | Usage Examples

### مثال 1: نظام RAG بسيط | Simple RAG System

```python
from arabic_rag.pipeline import ArabicRAGPipeline

# إعداد خط أنابيب RAG
pipeline = ArabicRAGPipeline(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    vector_store="chroma",
    llm_provider="openai"
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
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    vector_store="faiss",
    llm_provider="openai"
)

# إعداد فريق الوكلاء
crew = setup_crew(pipeline)

# تنفيذ مهمة البحث
task = "ابحث عن المتطلبات القانونية لتسجيل شركة جديدة في السعودية وقدم ملخصاً شاملاً"
result = crew.kickoff(inputs={"task": task})
print(result)
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
chunker = ArabicTextChunker(chunk_size=300, overlap=50)
document = "القانون التجاري السعودي يحدد الأطر القانونية لجميع العمليات التجارية. المادة الأولى تنص على حقوق التجار..."
chunks = chunker.chunk(document)
for i, chunk in enumerate(chunks):
    print(f"الجزء {i+1}: {chunk}")
```

---

## بنية المشروع | Project Structure

```
arabic-rag-toolkit/
├── README.md
├── LICENSE
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
│   │   └── multi_agent_crew.py # إعداد فريق CrewAI
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
```
# LLM APIs
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Vector Store Choice
VECTOR_STORE=chroma  # Options: chroma, faiss

# LLM Provider
LLM_PROVIDER=openai  # Options: openai, anthropic

# Model Names
OPENAI_MODEL=gpt-4
ANTHROPIC_MODEL=claude-3-opus-20240229

# Vector Store Path
VECTOR_STORE_PATH=./data/vector_store

# Chunk Settings
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

---

## المتطلبات | Requirements

- Python 3.9+
- LangChain 0.1+
- CrewAI 0.1+
- Sentence-Transformers
- FAISS-CPU or ChromaDB
- OpenAI or Anthropic API keys (optional for local models)

See `requirements.txt` for complete list.

---

## المساهمة | Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on our code of conduct and the process for submitting pull requests.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/azizalzahrani/arabic-rag-toolkit.git
cd arabic-rag-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## الترخيص | License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## التواصل والدعم | Support & Contact

- **Issues**: [GitHub Issues](https://github.com/azizalzahrani/arabic-rag-toolkit/issues)
- **Author**: [@azizalzahrani](https://github.com/azizalzahrani)
- **Email**: support@example.com

---

## الشكر والاعتراف | Acknowledgments

- CAMeL Lab for Arabic NLP research
- Hugging Face for transformer models
- LangChain and CrewAI communities
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
**Last Updated**: 2026-03-21
**Status**: Active Development
