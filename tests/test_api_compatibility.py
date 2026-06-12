"""
اختبارات التوافق مع الواجهة العامة - Public API compatibility tests

العربية:
    اختبارات تغطي الواجهات الموثقة وسلوك التشغيل المحلي بدون تبعيات خارجية.

English:
    Tests covering documented APIs and the local offline-compatible execution path.
"""

import re
from pathlib import Path

from arabic_rag.generator import ArabicResponseGenerator
from arabic_rag.pipeline import ArabicRAGPipeline
from arabic_rag import __version__


def test_pipeline_supports_documented_shortcuts():
    """يدعم اختصارات البناء المذكورة في الأمثلة."""
    pipeline = ArabicRAGPipeline(
        embedding_model="demo-model",
        vector_store="memory",
        llm_provider="local",
        verbose=True,
        chunk_size=128,
        chunk_overlap=16,
    )

    assert pipeline.config.embedding_config.model_name == "demo-model"
    assert pipeline.config.retrieval_config.vector_store_type == "memory"
    assert pipeline.config.generation_config.llm_provider == "local"
    assert pipeline.config.verbose is True
    assert pipeline.config.chunking_config.chunk_size == 128
    assert pipeline.config.chunking_config.chunk_overlap == 16


def test_pipeline_defaults_are_local_first():
    """التهيئة الافتراضية يجب أن تعمل مباشرة في بيئة نظيفة."""
    pipeline = ArabicRAGPipeline()

    assert pipeline.config.retrieval_config.vector_store_type == "memory"
    assert pipeline.config.generation_config.llm_provider == "local"
    assert pipeline.get_pipeline_stats()["vector_store_type"] == "memory"


def test_pipeline_readme_style_flow_works_without_external_services():
    """يمكن تنفيذ التدفق الموثق محلياً بدون مفاتيح API أو قواعد بيانات خارجية."""
    pipeline = ArabicRAGPipeline(vector_store="memory", llm_provider="local")
    pipeline.add_documents(
        [
            "نظام الشركات السعودي ينص على أن رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي.",
            "مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير.",
        ]
    )

    results = pipeline.retrieve("كم هو الحد الأدنى لرأس مال الشركة المساهمة؟")
    answer = pipeline.generate_answer(results, "كم هو الحد الأدنى لرأس مال الشركة المساهمة؟")

    assert results
    assert "خمسة" in answer or "رأس مال" in answer


def test_generator_defaults_to_local_provider_without_api_keys():
    """يستخدم المولد المزود المحلي عند غياب مفاتيح المزودات الخارجية."""
    generator = ArabicResponseGenerator()
    assert generator.provider_name == "local"


def test_package_version_matches_pyproject():
    """يبقى رقم الإصدار متسقاً بين الحزمة وبيانات النشر."""
    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject_text, re.MULTILINE)

    assert match is not None
    assert match.group(1) == __version__


def test_pipeline_from_env_reads_documented_variables(monkeypatch):
    """تُقرأ متغيرات البيئة الموثقة في ‎.env.example‎ فعلياً."""
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("TEMPERATURE", "0.5")
    monkeypatch.setenv("MAX_TOKENS", "512")

    pipeline = ArabicRAGPipeline.from_env()

    assert pipeline.config.retrieval_config.vector_store_type == "memory"
    assert pipeline.config.generation_config.llm_provider == "local"
    assert pipeline.config.chunking_config.chunk_size == 256
    assert pipeline.config.chunking_config.chunk_overlap == 32
    assert pipeline.config.retrieval_config.top_k == 7
    assert pipeline.config.generation_config.temperature == 0.5
    assert pipeline.config.generation_config.max_tokens == 512


def test_reset_keeps_loaded_embedding_model():
    """إعادة التعيين تمسح المستندات دون إعادة تحميل نموذج التضمين."""
    pipeline = ArabicRAGPipeline()
    embeddings_before = pipeline.embeddings

    pipeline.add_documents(["مستند تجريبي للفهرسة والاختبار."])
    assert pipeline.documents

    pipeline.reset()

    assert pipeline.embeddings is embeddings_before
    assert pipeline.documents == []
    assert pipeline.retrieve("مستند") == []


def test_query_sources_include_document_snippets():
    """قسم المصادر يتضمن مقتطفات من المستندات لا الدرجات فقط."""
    pipeline = ArabicRAGPipeline(vector_store="memory", llm_provider="local")
    pipeline.add_documents(["مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير."])

    answer = pipeline.query("من المسؤول عن إدارة الشركة؟", return_sources=True)

    sources_section = answer.split("المصادر:")[-1]
    assert "المصادر" in answer
    assert "[1]" in sources_section
    # المقتطف يتضمن نص المستند وليس الدرجة فقط
    assert "مجلس" in sources_section


def test_retrieved_documents_keep_original_orthography():
    """النص المعروض يحتفظ بالهمزات والرسم الأصلي بينما يتم البحث مطبّعاً."""
    pipeline = ArabicRAGPipeline(vector_store="memory", llm_provider="local")
    original = "مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير."
    pipeline.add_documents([original])

    results = pipeline.retrieve("من المسءول عن اداره الشركه؟")

    assert results
    top_document = results[0][0]
    assert "مسؤول" in top_document  # لم تتحول إلى "مسءول"
    assert "الإدارة" in top_document  # لم تتحول إلى "الاداره"
