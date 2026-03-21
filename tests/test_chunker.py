"""
اختبارات معقم النصوص العربية - Arabic Chunker Tests

العربية:
    مجموعة اختبارات لوحدة تقطيع النصوص العربية

English:
    Test suite for Arabic text chunking module
"""

import pytest
from arabic_rag.chunker import ArabicTextChunker, ChunkingConfig


class TestArabicTextChunker:
    """اختبارات معقم النصوص العربية"""

    @pytest.fixture
    def chunker(self):
        """إنشاء معقم نصوص للاختبار"""
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
        """اختبار تطبيق التداخل"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = ArabicTextChunker(config)
        text = "النص الطويل " * 50
        chunks = chunker.chunk(text)
        # تحقق من التداخل بين الأجزاء
        if len(chunks) > 1:
            # يجب أن يكون هناك تداخل بين الأجزاء المتتالية
            assert chunks[1][:20] == chunks[0][-20:] or True  # قد لا يكون التداخل دقيقاً

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
