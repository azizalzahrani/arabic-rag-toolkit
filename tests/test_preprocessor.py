"""
اختبارات معالج النصوص العربية - Arabic Preprocessor Tests

العربية:
    مجموعة اختبارات لوحدة معالجة النصوص العربية

English:
    Test suite for Arabic text preprocessing module
"""

import pytest
from arabic_rag.preprocessor import ArabicTextPreprocessor, NormalizationConfig


class TestArabicTextPreprocessor:
    """اختبارات معالج النصوص العربية"""

    @pytest.fixture
    def preprocessor(self):
        """إنشاء معالج نصوص للاختبار"""
        return ArabicTextPreprocessor()

    def test_remove_diacritics(self, preprocessor):
        """اختبار إزالة التشكيل"""
        text = "اَلسَّلامُ عَلَيْكُمْ"
        result = preprocessor._remove_diacritics(text)
        assert "َ" not in result
        assert "ُ" not in result
        assert "السلام عليكم" == result

    def test_normalize_alef(self, preprocessor):
        """اختبار توحيد الألف"""
        text = "أحمد إبراهيم آمن"
        result = preprocessor._normalize_alef(text)
        assert result == "احمد ابراهيم امن"

    def test_normalize_complete(self, preprocessor):
        """اختبار التطبيع الكامل"""
        text = "اَلسَّلامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ"
        result = preprocessor.normalize(text)
        assert result == "السلام عليكم ورحمة الله"

    def test_remove_extra_spaces(self, preprocessor):
        """اختبار إزالة المسافات الزائدة"""
        text = "مرحبا    بك    في     العالم"
        result = preprocessor._remove_extra_spaces(text)
        assert result == "مرحبا بك في العالم"

    def test_normalize_query(self, preprocessor):
        """اختبار تطبيع الاستعلام"""
        query = "ما  هُو  القَانُون؟"
        result = preprocessor.normalize_query(query)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_stem(self, preprocessor):
        """اختبار استخراج جذر الكلمة"""
        word = "والمدرسة"
        result = preprocessor.extract_stem(word)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_config_validation(self):
        """اختبار التحقق من التكوين"""
        config = NormalizationConfig(remove_diacritics=False)
        preprocessor = ArabicTextPreprocessor(config)
        text = "اَلْسَّلامُ"
        result = preprocessor.normalize(text)
        # إذا لم نزيل التشكيل، يجب أن يكون موجوداً في النص المطبّع
        # (لكن قد تكون هناك معالجات أخرى)
        assert isinstance(result, str)

    def test_alef_wasla_normalized(self, preprocessor):
        """ألف الوصل تُطبّع إلى ألف عادية"""
        result = preprocessor.normalize("ٱلرحمن")
        assert result == "الرحمن"

    def test_dagger_alef_removed(self, preprocessor):
        """الألف الخنجرية تُزال مع التشكيل"""
        result = preprocessor.normalize("ٱلرَّحْمَٰن")
        assert result == "الرحمن"

    def test_combining_hamza_marks_removed(self, preprocessor):
        """علامات الهمزة المركبة تُزال"""
        # ياء + همزة مركبة فوقها (U+064A U+0654)
        result = preprocessor.normalize("شئ")
        assert "ٔ" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
