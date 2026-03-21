"""
اختبارات التوافق مع الواجهة العامة - Public API compatibility tests

العربية:
    اختبارات تغطي الواجهات الموثقة وسلوك التشغيل المحلي بدون تبعيات خارجية.

English:
    Tests covering documented APIs and the local offline-compatible execution path.
"""

from arabic_rag.generator import ArabicResponseGenerator
from arabic_rag.pipeline import ArabicRAGPipeline


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
