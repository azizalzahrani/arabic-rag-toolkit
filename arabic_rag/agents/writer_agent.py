"""
وكيل الكتابة - Writer Agent

العربية:
    وكيل متخصص في صياغة وتحسين الإجابات النهائية

English:
    Specialized agent for crafting and improving final responses
"""

from typing import Optional, List


class WriterAgent:
    """
    وكيل الكتابة - Writer Agent

    العربية:
        وكيل متخصص في صياغة الإجابات النهائية بشكل احترافي
        واضح ومتناسق مع المحتوى العربي.

    English:
        Specialized agent for crafting final responses professionally,
        clearly, and consistent with Arabic content standards.

    Example:
        ```python
        agent = WriterAgent()
        refined = agent.refine_answer("إجابة أولية")
        ```
    """

    def __init__(self, style: str = "formal"):
        """
        تهيئة وكيل الكتابة - Initialize writer agent

        Args:
            style: str - نمط الكتابة (formal, casual, technical)
        """
        self.style = style
        self.writing_history = []

    def refine_answer(self, answer: str, context: Optional[str] = None) -> str:
        """
        تحسين الإجابة - Refine answer

        العربية:
            تحسين صياغة الإجابة وجعلها أكثر وضوحاً وتماسكاً

        English:
            Improve answer phrasing to make it clearer and more coherent.

        Args:
            answer: str - الإجابة الأولية
            context: str - السياق (اختياري)

        Returns:
            str - الإجابة المحسّنة
        """
        # تنسيق الإجابة
        refined = answer.strip()

        # إضافة الفواصل المناسبة
        refined = self._add_proper_punctuation(refined)

        # تحسين البنية
        refined = self._improve_structure(refined)

        # التحقق من الاتساق
        refined = self._ensure_consistency(refined)

        self.writing_history.append({
            "original": answer,
            "refined": refined
        })

        return refined

    def format_structured_answer(self, answer: str, include_introduction: bool = True,
                                include_conclusion: bool = True) -> str:
        """
        تنسيق إجابة منظمة - Format structured answer

        العربية:
            تنسيق الإجابة بشكل منظم مع مقدمة وخاتمة

        English:
            Format answer with introduction, body, and conclusion.

        Args:
            answer: str - الإجابة
            include_introduction: bool - إضافة مقدمة
            include_conclusion: bool - إضافة خاتمة

        Returns:
            str - الإجابة المنسقة
        """
        formatted = ""

        if include_introduction:
            formatted += self._generate_introduction(answer) + "\n\n"

        formatted += answer + "\n"

        if include_conclusion:
            formatted += "\n" + self._generate_conclusion(answer)

        return formatted.strip()

    def add_citations(self, answer: str, sources: List[str]) -> str:
        """
        إضافة الاستشهادات - Add citations

        العربية:
            إضافة الاستشهادات من المصادر إلى الإجابة

        English:
            Add citations from sources to the answer.

        Args:
            answer: str - الإجابة
            sources: List[str] - قائمة المصادر

        Returns:
            str - الإجابة مع الاستشهادات
        """
        cited_answer = answer + "\n\n" + "المصادر:\n"
        for i, source in enumerate(sources, 1):
            # اختصار المصدر إذا كان طويلاً
            short_source = source[:100] + "..." if len(source) > 100 else source
            cited_answer += f"[{i}] {short_source}\n"

        return cited_answer

    def create_summary(self, answer: str, max_sentences: int = 3) -> str:
        """
        إنشاء ملخص - Create summary

        العربية:
            إنشاء ملخص موجز للإجابة

        English:
            Create a concise summary of the answer.

        Args:
            answer: str - الإجابة
            max_sentences: int - الحد الأقصى للجمل

        Returns:
            str - الملخص
        """
        # تقسيم إلى جمل
        sentences = answer.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        # اختيار أهم الجمل
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences)

        return summary.strip()

    def check_arabic_quality(self, text: str) -> dict:
        """
        التحقق من جودة النص العربي - Check Arabic quality

        العربية:
            التحقق من جودة ودقة النص العربي

        English:
            Verify quality and accuracy of Arabic text.

        Args:
            text: str - النص

        Returns:
            dict - نتائج الفحص
        """
        issues = []

        # التحقق من المسافات الزائدة
        if '  ' in text:
            issues.append("مسافات متعددة غير ضرورية")

        # التحقق من علامات الترقيم
        if not text.endswith(('।', '؟', '!', '.')):
            issues.append("النص لا ينتهي بعلامة ترقيم مناسبة")

        # التحقق من الأقواس المتطابقة
        if text.count('(') != text.count(')'):
            issues.append("الأقواس غير متطابقة")

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "quality_score": 1.0 - (len(issues) * 0.2)
        }

    def get_writing_history(self) -> List[dict]:
        """
        الحصول على سجل الكتابة - Get writing history

        العربية:
            الحصول على سجل جميع الإجابات المكتوبة

        English:
            Get history of all written responses.

        Returns:
            List[dict] - سجل الكتابة
        """
        return self.writing_history

    def _add_proper_punctuation(self, text: str) -> str:
        """إضافة علامات الترقيم المناسبة"""
        # تصحيح المسافات قبل علامات الترقيم
        text = text.replace(' .', '.')
        text = text.replace(' ؟', '؟')
        text = text.replace(' !', '!')
        text = text.replace(' ،', '،')
        return text

    def _improve_structure(self, text: str) -> str:
        """تحسين بنية النص"""
        # إزالة المسافات الزائدة
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _ensure_consistency(self, text: str) -> str:
        """ضمان اتساق النص"""
        # توحيد علامات الترقيم
        text = text.replace('،،', '،')
        text = text.replace('..', '.')
        return text

    def _generate_introduction(self, answer: str) -> str:
        """توليد مقدمة"""
        # استخراج أول جملة
        first_sentence = answer.split('.')[0] if '.' in answer else answer[:50]
        return f"فيما يلي الإجابة على سؤالك:"

    def _generate_conclusion(self, answer: str) -> str:
        """توليد خاتمة"""
        return "نتمنى أن تكون هذه المعلومات مفيدة لك."

    def clear_history(self) -> None:
        """
        مسح السجل - Clear history

        العربية:
            مسح جميع سجلات الكتابة السابقة

        English:
            Clear all writing history.
        """
        self.writing_history = []
