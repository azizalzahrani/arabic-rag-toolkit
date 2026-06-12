"""
اختبارات نموذج التضمين - Arabic Embeddings Tests

العربية:
    اختبارات لوحدة التضمين بما فيها المسار المحلي الاحتياطي

English:
    Tests for the embeddings module, including the local fallback path
"""

import numpy as np
import pytest

from arabic_rag.embeddings import ArabicEmbeddings, EmbeddingConfig


@pytest.fixture
def embeddings():
    """نموذج تضمين للاختبار"""
    return ArabicEmbeddings()


class TestArabicEmbeddings:
    """اختبارات نموذج التضمين"""

    def test_embed_text_shape(self, embeddings):
        """متجه التضمين له البعد المتوقع"""
        vector = embeddings.embed_text("مرحبا بك")
        assert vector.shape == (embeddings.get_embedding_dimension(),)

    def test_embed_text_rejects_empty(self, embeddings):
        """النص الفارغ يرفع خطأ واضحاً"""
        with pytest.raises(ValueError):
            embeddings.embed_text("   ")

    def test_embed_batch_rejects_all_empty(self, embeddings):
        """قائمة نصوص فارغة بالكامل ترفع خطأ"""
        with pytest.raises(ValueError):
            embeddings.embed_batch(["", "   "])

    def test_similarity_is_symmetric_and_bounded(self, embeddings):
        """درجة التشابه متناظرة ومحصورة"""
        a = embeddings.similarity("المدرسة والتعليم", "التعليم في المدرسة")
        b = embeddings.similarity("التعليم في المدرسة", "المدرسة والتعليم")
        assert a == pytest.approx(b)
        assert -1.0 <= a <= 1.0

    def test_most_similar_alignment_with_empty_entries(self, embeddings):
        """النصوص الفارغة لا تخل بمحاذاة النتائج مع نصوصها

        Regression test: embed_batch silently drops empty strings, so
        most_similar must rank against the same filtered list.
        """
        texts = ["المدرسة والتعليم والدراسة", "", "   ", "الطبخ والمأكولات الشهية"]
        results = embeddings.most_similar("التعليم والدراسة", texts, top_k=2)

        returned_texts = [text for _, text in results]
        assert "" not in returned_texts
        assert returned_texts[0] == "المدرسة والتعليم والدراسة"

    def test_most_similar_scores_are_sorted(self, embeddings):
        """النتائج مرتبة تنازلياً حسب الدرجة"""
        results = embeddings.most_similar(
            "القانون التجاري",
            ["نظام الشركات والقانون التجاري", "الطقس اليوم مشمس", "أحكام القانون التجاري السعودي"],
            top_k=3,
        )
        scores = [score for score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_dimension_matches_fallback_config(self):
        """البعد يطابق الإعداد عند استخدام المسار المحلي"""
        embeddings = ArabicEmbeddings(EmbeddingConfig(fallback_dimension=128))
        if embeddings.backend == "hashing":
            assert embeddings.get_embedding_dimension() == 128
            assert embeddings.embed_text("نص").shape == (128,)

    def test_normalized_vectors_have_unit_length(self, embeddings):
        """التطبيع يجعل طول المتجه يساوي 1"""
        vector = embeddings.embed_text("نص عربي للاختبار")
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_save_and_load_roundtrip(self, embeddings, tmp_path):
        """الحفظ والتحميل يعيدان نفس المصفوفة"""
        vectors = embeddings.embed_batch(["النص الأول", "النص الثاني"])
        filepath = str(tmp_path / "embeddings.npy")
        embeddings.save_embeddings(vectors, filepath)
        loaded = embeddings.load_embeddings(filepath)
        assert np.allclose(vectors, loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
