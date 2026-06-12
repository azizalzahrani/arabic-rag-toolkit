"""
اختبارات منشئ الإجابات - Response Generator Tests

العربية:
    اختبارات لاختيار النموذج والمزود وسلوك التراجع المحلي

English:
    Tests for model resolution, provider selection, and local fallback behavior
"""

import pytest

from arabic_rag.generator import (
    DEFAULT_MODELS,
    ArabicResponseGenerator,
    GenerationConfig,
    resolve_model_name,
)


class TestModelResolution:
    """اختبارات تحديد اسم النموذج"""

    def test_explicit_model_wins(self, monkeypatch):
        """الاسم الصريح في التكوين له الأولوية"""
        monkeypatch.setenv("OPENAI_MODEL", "env-model")
        config = GenerationConfig(model_name="explicit-model")
        assert resolve_model_name(config, "openai") == "explicit-model"

    def test_env_model_used_when_config_empty(self, monkeypatch):
        """متغير البيئة يستخدم عند غياب الاسم الصريح"""
        monkeypatch.setenv("ANTHROPIC_MODEL", "env-anthropic-model")
        config = GenerationConfig()
        assert resolve_model_name(config, "anthropic") == "env-anthropic-model"

    def test_provider_default_used_last(self, monkeypatch):
        """الافتراضي الخاص بالمزود يستخدم أخيراً"""
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        config = GenerationConfig()
        assert resolve_model_name(config, "openai") == DEFAULT_MODELS["openai"]
        assert resolve_model_name(config, "anthropic") == DEFAULT_MODELS["anthropic"]

    def test_no_cross_provider_model_leakage(self, monkeypatch):
        """افتراضي OpenAI لا يتسرب إلى Anthropic (إصلاح gpt-4 السابق)"""
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        config = GenerationConfig()
        assert "gpt" not in resolve_model_name(config, "anthropic")


class TestProviderFallback:
    """اختبارات التراجع إلى المزود المحلي"""

    def test_missing_api_key_falls_back_with_warning(self, monkeypatch):
        """غياب مفتاح API يتراجع محلياً مع تحذير واضح"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.warns(RuntimeWarning, match="OPENAI_API_KEY"):
            generator = ArabicResponseGenerator(GenerationConfig(llm_provider="openai"))
        assert generator.provider_name == "local"

    def test_unknown_provider_raises(self):
        """مزود غير معروف يرفع خطأ"""
        with pytest.raises(ValueError):
            ArabicResponseGenerator(GenerationConfig(llm_provider="mystery"))

    def test_local_provider_no_warning(self, recwarn):
        """اختيار المزود المحلي صراحةً لا يصدر تحذيرات"""
        generator = ArabicResponseGenerator(GenerationConfig(llm_provider="local"))
        assert generator.provider_name == "local"
        runtime_warnings = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings


class TestLocalExtractiveAnswers:
    """اختبارات الإجابات الاستخراجية المحلية"""

    @pytest.fixture
    def generator(self):
        return ArabicResponseGenerator(GenerationConfig(llm_provider="local"))

    def test_answer_uses_context(self, generator):
        """الإجابة تستند إلى السياق المقدم"""
        answer = generator.generate_answer(
            "كم الحد الأدنى لرأس المال؟",
            context="رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي.",
        )
        assert "خمسة ملايين" in answer

    def test_answer_without_context_is_honest(self, generator):
        """بدون سياق، الإجابة توضح عدم توفر معلومات"""
        answer = generator.generate_answer("ما هو القانون التجاري؟")
        assert "لا يتوفر" in answer or "لا توجد" in answer

    def test_generate_with_references_counts_documents(self, generator):
        """عدد المراجع يطابق عدد المستندات"""
        result = generator.generate_with_references(
            "ما حقوق المساهمين؟",
            ["للمساهمين حق حضور الجمعية العامة.", "لهم حق التصويت على القرارات."],
        )
        assert result["document_count"] == 2
        assert len(result["references"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
