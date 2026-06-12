"""
اختبارات مُقطِّع النصوص العربية - Arabic Chunker Tests

العربية:
    مجموعة اختبارات لوحدة تقطيع النصوص العربية

English:
    Test suite for Arabic text chunking module
"""

import pytest
from arabic_rag.chunker import ArabicTextChunker, ChunkingConfig


class TestArabicTextChunker:
    """اختبارات مُقطِّع النصوص العربية"""

    @pytest.fixture
    def chunker(self):
        """إنشاء مُقطِّع نصوص للاختبار"""
        return ArabicTextChunker()

    def test_chunk_small_text(self, chunker):
        """اختبار تقطيع نص صغير"""
        text = "هذا نص صغير جداً"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_large_text(self, chunker):
        """اختبار تقطيع نص كبير"""
        text = " ".join(["هذا جزء من النص"] * 50)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_with_sentences(self, chunker):
        """اختبار تقطيع مع احترام الجمل"""
        text = "هذه جملة أولى. هذه جملة ثانية. هذه جملة ثالثة."
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        # يجب أن تكون الأجزاء تحتوي على جمل كاملة
        for chunk in chunks:
            assert len(chunk) > 0

    def test_split_paragraphs(self, chunker):
        """اختبار تقسيم الفقرات"""
        text = "فقرة أولى\n\nفقرة ثانية\n\nفقرة ثالثة"
        paragraphs = chunker._split_paragraphs(text)
        assert len(paragraphs) == 3

    def test_split_sentences(self, chunker):
        """اختبار تقسيم الجمل"""
        text = "جملة أولى. جملة ثانية؟ جملة ثالثة!"
        sentences = chunker._split_sentences(text)
        assert len(sentences) > 0

    def test_chunk_statistics(self, chunker):
        """اختبار إحصائيات التقطيع"""
        text = "هذا النص يستخدم للاختبار" * 20
        chunks = chunker.chunk(text)
        stats = chunker.get_chunk_statistics(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] > 0
        assert stats["average_chunk_size"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] > 0

    def test_overlap_application(self, chunker):
        """اختبار تطبيق التداخل: كل جزء يبدأ بنهاية الجزء السابق"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = ArabicTextChunker(config)
        text = "النص الطويل " * 50
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        for previous, current in zip(chunks, chunks[1:]):
            leading_words = current.split()[:2]
            tail = previous[-60:]
            # كلمات بداية الجزء الحالي موجودة في نهاية الجزء السابق
            assert all(word in tail for word in leading_words)

    def test_overlap_respects_word_boundaries(self):
        """التداخل لا يقطع الكلمات في منتصفها"""
        config = ChunkingConfig(chunk_size=60, chunk_overlap=15)
        chunker = ArabicTextChunker(config)
        text = "كلمات عربية متنوعة للتجربة والاختبار " * 10
        chunks = chunker.chunk(text)

        vocabulary = set(text.split())
        for chunk in chunks[1:]:
            first_word = chunk.split()[0]
            assert first_word in vocabulary, f"كلمة مقطوعة: {first_word}"

    def test_constructor_keyword_shortcuts(self):
        """اختصارات البناء الموثقة تعمل"""
        chunker = ArabicTextChunker(chunk_size=150, chunk_overlap=30)
        assert chunker.config.chunk_size == 150
        assert chunker.config.chunk_overlap == 30

    def test_constructor_rejects_config_and_kwargs(self):
        """تمرير تكوين واختصارات معاً يرفع خطأ"""
        with pytest.raises(TypeError):
            ArabicTextChunker(ChunkingConfig(), chunk_size=100)

    def test_pipe_character_is_not_a_delimiter(self, chunker):
        """حرف '|' ليس فاصل جمل (إصلاح بناء النمط السابق)"""
        sentences = chunker._split_sentences("هذا نص | يحتوي على خط عمودي")
        assert len(sentences) == 1

    def test_empty_text(self, chunker):
        """اختبار معالجة نص فارغ"""
        chunks = chunker.chunk("")
        assert chunks == []

    def test_min_chunk_size(self):
        """اختبار الحد الأدنى لحجم الجزء"""
        config = ChunkingConfig(chunk_size=100, min_chunk_size=50)
        chunker = ArabicTextChunker(config)
        text = "نص قصير"
        chunks = chunker.chunk(text)
        # يجب ألا يحتوي على أجزاء أصغر من الحد الأدنى
        for chunk in chunks:
            assert len(chunk) >= config.min_chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
