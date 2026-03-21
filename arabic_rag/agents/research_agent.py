"""
وكيل البحث - Research Agent

العربية:
    وكيل متخصص في البحث عن المستندات ذات الصلة باستخدام نظام RAG

English:
    Specialized agent for researching relevant documents using RAG system
"""

from typing import Optional, List, Tuple


class ResearchAgent:
    """
    وكيل البحث - Research Agent

    العربية:
        وكيل متخصص في البحث عن المستندات والمعلومات ذات الصلة
        بناءً على استعلام معين.

    English:
        Specialized agent for researching relevant documents and information
        based on a given query.

    Example:
        ```python
        agent = ResearchAgent(retriever)
        docs = agent.search("ما هو القانون التجاري؟")
        ```
    """

    def __init__(self, retriever=None):
        """
        تهيئة وكيل البحث - Initialize research agent

        Args:
            retriever: ArabicRetriever - محرك البحث
        """
        self.retriever = retriever
        self.search_history = []

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        البحث عن معلومات - Search for information

        العربية:
            البحث عن المستندات والمعلومات ذات الصلة بالاستعلام

        English:
            Search for relevant documents and information.

        Args:
            query: str - الاستعلام
            top_k: int - عدد النتائج

        Returns:
            List[Tuple[str, float]] - المستندات مع درجات التشابه
        """
        if not self.retriever:
            return []

        results = self.retriever.retrieve(query, top_k=top_k)
        self.search_history.append({"query": query, "results_count": len(results)})
        return results

    def analyze_query(self, query: str) -> dict:
        """
        تحليل الاستعلام - Analyze the query

        العربية:
            تحليل الاستعلام لاستخراج الكلمات المفتاحية والمفاهيم الرئيسية

        English:
            Analyze the query to extract keywords and main concepts.

        Args:
            query: str - الاستعلام

        Returns:
            dict - معلومات التحليل
        """
        keywords = query.split()
        return {
            "query": query,
            "keywords": keywords,
            "query_length": len(keywords),
            "main_terms": [kw for kw in keywords if len(kw) > 2]
        }

    def get_search_history(self) -> List[dict]:
        """
        الحصول على سجل البحث - Get search history

        العربية:
            الحصول على قائمة الاستعلامات السابقة ونتائجها

        English:
            Get list of previous queries and their results.

        Returns:
            List[dict] - سجل البحث
        """
        return self.search_history

    def clear_history(self) -> None:
        """
        مسح السجل - Clear history

        العربية:
            مسح جميع السجلات السابقة

        English:
            Clear all search history.
        """
        self.search_history = []
