"""
وكيل التحقق - Validator Agent

العربية:
    وكيل متخصص في التحقق من مدى صلة المستندات والإجابات المقترحة

English:
    Specialized agent for validating relevance of documents and proposed answers
"""

from typing import List, Tuple, Optional


class ValidatorAgent:
    """
    وكيل التحقق - Validator Agent

    العربية:
        وكيل متخصص في التحقق من جودة ودقة المستندات والإجابات.
        يقيم مدى ملاءمة النتائج للاستعلام الأصلي.

    English:
        Specialized agent for validating quality and accuracy of documents
        and answers. Assesses result relevance to original query.

    Example:
        ```python
        agent = ValidatorAgent()
        is_valid = agent.validate_document("النص", "السؤال")
        ```
    """

    def __init__(self, similarity_threshold: float = 0.5):
        """
        تهيئة وكيل التحقق - Initialize validator agent

        Args:
            similarity_threshold: float - الحد الأدنى للتشابه
        """
        self.similarity_threshold = similarity_threshold
        self.validation_log = []

    def validate_document(self, document: str, query: str,
                         similarity_score: Optional[float] = None) -> dict:
        """
        التحقق من صلة المستند - Validate document relevance

        العربية:
            التحقق من أن المستند ذو صلة بالاستعلام

        English:
            Verify that document is relevant to the query.

        Args:
            document: str - المستند
            query: str - الاستعلام
            similarity_score: float - درجة التشابه (اختياري)

        Returns:
            dict - نتائج التحقق
        """
        # التحقق من الطول الأدنى
        min_length_valid = len(document.strip()) > 10

        # التحقق من درجة التشابه
        similarity_valid = True
        if similarity_score is not None:
            similarity_valid = similarity_score >= self.similarity_threshold

        # عد الكلمات المشتركة
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        common_words = query_words & doc_words
        common_word_ratio = len(common_words) / len(query_words) if query_words else 0

        is_valid = min_length_valid and similarity_valid

        result = {
            "is_valid": is_valid,
            "document_length": len(document),
            "min_length_valid": min_length_valid,
            "similarity_valid": similarity_valid,
            "similarity_score": similarity_score,
            "common_word_count": len(common_words),
            "common_word_ratio": common_word_ratio,
            "validation_reason": self._get_validation_reason(is_valid, min_length_valid, similarity_valid)
        }

        self.validation_log.append(result)
        return result

    def validate_answer(self, answer: str, query: str, source_documents: List[str]) -> dict:
        """
        التحقق من الإجابة - Validate answer

        العربية:
            التحقق من جودة الإجابة ومدى استنادها على المستندات المصدر

        English:
            Verify answer quality and grounding in source documents.

        Args:
            answer: str - الإجابة
            query: str - السؤال الأصلي
            source_documents: List[str] - المستندات المستخدمة

        Returns:
            dict - نتائج التحقق
        """
        # التحقق من طول الإجابة
        min_answer_length = len(answer.strip()) > 20
        max_answer_length = len(answer) < 5000

        # التحقق من وجود المراجع
        has_references = len(source_documents) > 0

        # التحقق من الاستكمالية
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words & answer_words) / len(query_words) if query_words else 0

        is_complete = query_coverage >= 0.3
        is_valid = min_answer_length and max_answer_length and has_references and is_complete

        return {
            "is_valid": is_valid,
            "answer_length": len(answer),
            "length_valid": min_answer_length and max_answer_length,
            "has_sources": has_references,
            "source_count": len(source_documents),
            "query_coverage": query_coverage,
            "completeness": "كاملة" if is_complete else "ناقصة",
            "validation_scores": {
                "length": 1.0 if (min_answer_length and max_answer_length) else 0.0,
                "sources": 1.0 if has_references else 0.0,
                "coverage": query_coverage,
                "overall": sum([
                    1.0 if (min_answer_length and max_answer_length) else 0.0,
                    1.0 if has_references else 0.0,
                    query_coverage
                ]) / 3
            }
        }

    def filter_results(self, documents: List[Tuple[str, float]], query: str) -> List[Tuple[str, float]]:
        """
        تصفية النتائج - Filter results

        العربية:
            تصفية النتائج بناءً على معايير الصلة والجودة

        English:
            Filter results based on relevance and quality criteria.

        Args:
            documents: List[Tuple[str, float]] - المستندات مع درجات التشابه
            query: str - الاستعلام

        Returns:
            List[Tuple[str, float]] - المستندات المصفاة
        """
        filtered = []
        for doc, score in documents:
            validation = self.validate_document(doc, query, score)
            if validation["is_valid"]:
                filtered.append((doc, score))

        return filtered

    def get_validation_report(self) -> dict:
        """
        الحصول على تقرير التحقق - Get validation report

        العربية:
            الحصول على تقرير شامل عن جميع عمليات التحقق

        English:
            Get comprehensive report of all validations.

        Returns:
            dict - التقرير الشامل
        """
        if not self.validation_log:
            return {"message": "لا توجد عمليات تحقق سابقة"}

        valid_count = sum(1 for v in self.validation_log if v["is_valid"])
        total_count = len(self.validation_log)

        return {
            "total_validations": total_count,
            "valid_count": valid_count,
            "invalid_count": total_count - valid_count,
            "validity_rate": valid_count / total_count if total_count > 0 else 0,
            "average_similarity": sum(
                v["similarity_score"] for v in self.validation_log
                if v.get("similarity_score") is not None
            ) / max(len([
                v for v in self.validation_log if v.get("similarity_score") is not None
            ]), 1),
            "average_document_length": sum(v["document_length"] for v in self.validation_log) / total_count,
        }

    def _get_validation_reason(self, is_valid: bool, length_valid: bool, similarity_valid: bool) -> str:
        """
        الحصول على سبب التحقق - Get validation reason

        العربية:
            الحصول على شرح تفصيلي لنتيجة التحقق

        English:
            Get detailed explanation for validation result.

        Args:
            is_valid: bool - هل النتيجة صحيحة
            length_valid: bool - هل الطول صحيح
            similarity_valid: bool - هل التشابه صحيح

        Returns:
            str - شرح التحقق
        """
        if not is_valid:
            reasons = []
            if not length_valid:
                reasons.append("طول المستند أقل من المتوقع")
            if not similarity_valid:
                reasons.append("درجة التشابه أقل من الحد الأدنى")
            return "، ".join(reasons) if reasons else "فشل التحقق"
        return "المستند صحيح ومقبول"

    def clear_log(self) -> None:
        """
        مسح السجل - Clear validation log

        العربية:
            مسح جميع سجلات التحقق السابقة

        English:
            Clear all previous validation logs.
        """
        self.validation_log = []
