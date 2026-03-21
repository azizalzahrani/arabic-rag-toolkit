"""
فريق الوكلاء متعددي الأدوار - Multi-Agent CrewAI Setup

العربية:
    إعداد فريق من الوكلاء المتخصصة يعملون معاً لحل المهام المعقدة

English:
    Setup a team of specialized agents working together to solve complex tasks
"""

from typing import Optional, Dict, Any
from arabic_rag.agents.research_agent import ResearchAgent
from arabic_rag.agents.validator_agent import ValidatorAgent
from arabic_rag.agents.writer_agent import WriterAgent


class ArabicRAGCrew:
    """
    فريق RAG العربي - Arabic RAG Crew

    العربية:
        فريق متكامل من الوكلاء المتخصصة:
        - وكيل البحث: للبحث عن المستندات ذات الصلة
        - وكيل التحقق: للتحقق من جودة النتائج
        - وكيل الكتابة: لصياغة الإجابة النهائية

    English:
        Integrated team of specialized agents:
        - Research Agent: for finding relevant documents
        - Validator Agent: for verifying result quality
        - Writer Agent: for crafting final response

    Example:
        ```python
        crew = ArabicRAGCrew(retriever=my_retriever)
        result = crew.execute_task("ما هو القانون التجاري؟")
        ```
    """

    def __init__(self, retriever=None, verbose: bool = False):
        """
        تهيئة الفريق - Initialize the crew

        Args:
            retriever: ArabicRetriever - محرك البحث
            verbose: bool - إظهار التفاصيل
        """
        self.verbose = verbose
        self.research_agent = ResearchAgent(retriever)
        self.validator_agent = ValidatorAgent()
        self.writer_agent = WriterAgent(style="formal")

    def execute_task(self, task: str, top_k: int = 5) -> Dict[str, Any]:
        """
        تنفيذ مهمة - Execute a task

        العربية:
            تنفيذ مهمة كاملة من البحث إلى التحقق إلى الكتابة

        English:
            Execute a complete task from research to validation to writing.

        Args:
            task: str - المهمة / السؤال
            top_k: int - عدد المستندات المراد استرجاعها

        Returns:
            Dict - نتائج تنفيذ المهمة

        Example:
            ```python
            result = crew.execute_task("شرح مفهوم الشركة المساهمة")
            print(result['final_answer'])
            print(result['execution_report'])
            ```
        """
        self._log("بدء تنفيذ المهمة", task)

        # المرحلة 1: البحث
        self._log("المرحلة 1", "البحث عن المستندات ذات الصلة")
        search_results = self.research_agent.search(task, top_k=top_k)

        if not search_results:
            return {
                "task": task,
                "success": False,
                "final_answer": "عذراً، لم أتمكن من العثور على معلومات ذات صلة.",
                "execution_report": {"stage": "البحث", "status": "فشل"}
            }

        # استخراج المستندات
        documents = [doc for doc, _ in search_results]

        # المرحلة 2: التحقق
        self._log("المرحلة 2", "التحقق من جودة النتائج")
        validated_docs = self.validator_agent.filter_results(search_results, task)

        if not validated_docs:
            self._log("تحذير", "جميع النتائج فشلت التحقق، استخدام النتائج الأصلية")
            validated_docs = search_results

        # المرحلة 3: الصياغة
        self._log("المرحلة 3", "صياغة الإجابة النهائية")
        answer_body = self._generate_answer_body(validated_docs)
        final_answer = self.writer_agent.format_structured_answer(answer_body)
        final_answer = self.writer_agent.add_citations(final_answer, documents)

        # إنشاء التقرير
        validation_report = self.validator_agent.get_validation_report()
        execution_report = {
            "task": task,
            "stages": {
                "research": {
                    "status": "نجح",
                    "documents_found": len(search_results),
                    "documents_retrieved": len(validated_docs)
                },
                "validation": {
                    "status": "نجح",
                    "valid_count": len(validated_docs),
                    "validity_rate": len(validated_docs) / len(search_results) if search_results else 0
                },
                "writing": {
                    "status": "نجح",
                    "answer_length": len(final_answer),
                    "quality_check": self.writer_agent.check_arabic_quality(final_answer)
                }
            }
        }

        self._log("إتمام", "المهمة نفذت بنجاح")

        return {
            "task": task,
            "success": True,
            "final_answer": final_answer,
            "source_documents": documents,
            "validation_summary": validation_report,
            "execution_report": execution_report
        }

    def execute_complex_task(self, main_task: str, subtasks: list,
                            top_k: int = 5) -> Dict[str, Any]:
        """
        تنفيذ مهمة معقدة - Execute complex task with subtasks

        العربية:
            تنفيذ مهمة معقدة تتكون من عدة مهام فرعية

        English:
            Execute a complex task composed of multiple subtasks.

        Args:
            main_task: str - المهمة الرئيسية
            subtasks: list - المهام الفرعية
            top_k: int - عدد المستندات

        Returns:
            Dict - النتائج الشاملة
        """
        self._log("مهمة معقدة", f"المهمة الرئيسية: {main_task}")

        subtask_results = {}
        for i, subtask in enumerate(subtasks, 1):
            self._log(f"المهمة الفرعية {i}/{len(subtasks)}", subtask)
            result = self.execute_task(subtask, top_k=top_k)
            subtask_results[subtask] = result

        # توحيد النتائج
        comprehensive_answer = self._synthesize_answers(subtask_results)

        return {
            "main_task": main_task,
            "subtasks_count": len(subtasks),
            "subtask_results": subtask_results,
            "comprehensive_answer": comprehensive_answer,
            "overall_success": all(r["success"] for r in subtask_results.values())
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        الحصول على ملخص التنفيذ - Get execution summary

        العربية:
            الحصول على ملخص شامل لجميع عمليات التنفيذ

        English:
            Get comprehensive summary of all executions.

        Returns:
            Dict - الملخص الشامل
        """
        return {
            "research_history": self.research_agent.get_search_history(),
            "validation_report": self.validator_agent.get_validation_report(),
            "writing_history": self.writer_agent.get_writing_history(),
        }

    def _generate_answer_body(self, documents: list) -> str:
        """توليد جسم الإجابة من المستندات"""
        body = ""
        for i, (doc, score) in enumerate(documents, 1):
            body += f"({i}) {doc}\n\n"
        return body.strip()

    def _synthesize_answers(self, results: dict) -> str:
        """دمج الإجابات من المهام الفرعية"""
        synthesis = "تم دمج النتائج من جميع المهام الفرعية:\n\n"
        for task, result in results.items():
            synthesis += f"بخصوص '{task}':\n"
            synthesis += result["final_answer"][:200] + "...\n\n"
        return synthesis

    def _log(self, stage: str, message: str) -> None:
        """تسجيل الرسائل"""
        if self.verbose:
            print(f"[{stage}] {message}")

    def clear_all_history(self) -> None:
        """
        مسح جميع السجلات - Clear all history

        العربية:
            مسح جميع سجلات جميع الوكلاء

        English:
            Clear all history from all agents.
        """
        self.research_agent.clear_history()
        self.validator_agent.clear_log()
        self.writer_agent.clear_history()


def setup_crew(retriever=None, verbose: bool = False) -> ArabicRAGCrew:
    """
    إعداد الفريق - Setup the crew

    العربية:
        دالة مساعدة لإعداد فريق RAG العربي

    English:
        Helper function to setup the Arabic RAG crew.

    Args:
        retriever: ArabicRetriever - محرك البحث
        verbose: bool - الوضع التفصيلي

    Returns:
        ArabicRAGCrew - الفريق المُعد

    Example:
        ```python
        crew = setup_crew(my_retriever, verbose=True)
        result = crew.execute_task("سؤالي")
        ```
    """
    return ArabicRAGCrew(retriever=retriever, verbose=verbose)
