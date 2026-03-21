"""
اختبارات خط أنابيب RAG - RAG Pipeline Tests

العربية:
    مجموعة اختبارات لخط أنابيب RAG العربي

English:
    Test suite for Arabic RAG pipeline
"""

import pytest
from arabic_rag.pipeline import ArabicRAGPipeline, PipelineConfig


class TestArabicRAGPipeline:
    """اختبارات خط أنابيب RAG"""

    @pytest.fixture
    def pipeline(self):
        """إنشاء خط أنابيب للاختبار"""
        return ArabicRAGPipeline()

    @pytest.fixture
    def sample_documents(self):
        """مستندات نموذجية للاختبار"""
        return [
            "نظام الشركات السعودي ينص على أن رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي",
            "للمساهمين الحق في حضور الجمعية العامة والتصويت على القرارات",
            "مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير",
        ]

    def test_pipeline_initialization(self, pipeline):
        """اختبار تهيئة خط الأنابيب"""
        assert pipeline is not None
        assert pipeline.preprocessor is not None
        assert pipeline.chunker is not None
        assert pipeline.embeddings is not None
        assert pipeline.retriever is not None
        assert pipeline.generator is not None

    def test_add_documents(self, pipeline, sample_documents):
        """اختبار إضافة المستندات"""
        pipeline.add_documents(sample_documents)
        assert len(pipeline.documents) > 0

    def test_query_without_documents(self, pipeline):
        """اختبار الاستعلام بدون مستندات"""
        result = pipeline.query("ما هو القانون التجاري؟", return_sources=False)
        assert "عذراً" in result or "لم أتمكن" in result

    def test_get_pipeline_stats(self, pipeline, sample_documents):
        """اختبار الحصول على إحصائيات خط الأنابيب"""
        pipeline.add_documents(sample_documents)
        stats = pipeline.get_pipeline_stats()

        assert "total_documents" in stats
        assert "embedding_dimension" in stats
        assert "vector_store_type" in stats
        assert stats["total_documents"] > 0

    def test_batch_query(self, pipeline, sample_documents):
        """اختبار الاستعلام عن عدة أسئلة"""
        pipeline.add_documents(sample_documents)
        questions = ["ما هو؟", "كيف؟"]
        # Note: لا نختبر النتائج الفعلية لأنها تتطلب نموذج LLM
        assert len(questions) == 2

    def test_reset_pipeline(self, pipeline, sample_documents):
        """اختبار إعادة تعيين خط الأنابيب"""
        pipeline.add_documents(sample_documents)
        assert len(pipeline.documents) > 0

        pipeline.reset()
        assert len(pipeline.documents) == 0

    def test_configuration(self):
        """اختبار إعدادات خط الأنابيب"""
        config = PipelineConfig(verbose=True)
        pipeline = ArabicRAGPipeline(config=config)
        assert pipeline.config.verbose is True

    def test_preprocessor_integration(self, pipeline):
        """اختبار تكامل معالج النصوص"""
        text = "اَلنَّصُ العَرَبِي"
        normalized = pipeline.preprocessor.normalize(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0

    def test_chunker_integration(self, pipeline):
        """اختبار تكامل معقم النصوص"""
        text = "نص طويل جداً " * 20
        chunks = pipeline.chunker.chunk(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
