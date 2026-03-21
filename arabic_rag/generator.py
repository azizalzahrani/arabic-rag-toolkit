"""
توليد الإجابات - Arabic Response Generation Module

العربية:
    وحدة متخصصة في توليد الإجابات من خلال نماذج اللغة الكبيرة.
    تدعم OpenAI و Anthropic والنماذج المحلية.

English:
    Specialized module for generating responses using large language models.
    Supports OpenAI, Anthropic, and local models.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import re


@dataclass
class GenerationConfig:
    """
    إعدادات التوليد - Generation configuration

    العربية:
        تكوين معاملات توليد الإجابات

    English:
        Configuration for response generation parameters.
    """
    llm_provider: str = "local"  # openai, anthropic, local
    model_name: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 0.9
    timeout: int = 30


class LLMProvider(ABC):
    """
    واجهة مزود النماذج - LLM Provider Interface

    العربية:
        واجهة مجردة لمزودي نماذج اللغة

    English:
        Abstract interface for LLM providers.
    """

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """توليد إجابة"""
        pass


class OpenAIProvider(LLMProvider):
    """
    مزود OpenAI - OpenAI Provider

    العربية:
        توليد الإجابات باستخدام OpenAI API

    English:
        Generate responses using OpenAI API.
    """

    def __init__(self):
        """تهيئة مزود OpenAI"""
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("openai is required. Install it with: pip install openai")

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """
        توليد إجابة - Generate response

        Args:
            prompt: str - المحفزات
            config: GenerationConfig - التكوين

        Returns:
            str - الإجابة المولدة
        """
        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بشكل دقيق وشامل."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout=config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """
    مزود Anthropic - Anthropic Provider

    العربية:
        توليد الإجابات باستخدام Anthropic API

    English:
        Generate responses using Anthropic API.
    """

    def __init__(self):
        """تهيئة مزود Anthropic"""
        try:
            import anthropic
            self.client = anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic is required. Install it with: pip install anthropic")

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """
        توليد إجابة - Generate response

        Args:
            prompt: str - المحفزات
            config: GenerationConfig - التكوين

        Returns:
            str - الإجابة المولدة
        """
        try:
            message = self.client.messages.create(
                model=config.model_name,
                max_tokens=config.max_tokens,
                system="أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بشكل دقيق وشامل.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")


class LocalExtractiveProvider(LLMProvider):
    """
    مزود محلي - Local fallback provider

    العربية:
        مزود خفيف يعتمد على استخراج أفضل المقاطع من السياق عند غياب مزودي LLM.

    English:
        Lightweight fallback provider that extracts the best context snippets when
        external LLM providers are unavailable.
    """

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if "التعليقات والتحسينات المطلوبة:" in prompt:
            return self._extract_section(prompt, "الإجابة الحالية:", "التعليقات والتحسينات المطلوبة:").strip()

        if "سؤال المتابعة:" in prompt:
            follow_up = self._extract_section(prompt, "سؤال المتابعة:", "الرجاء").strip()
            original_answer = self._extract_section(prompt, "الإجابة على السؤال الأول:", "سؤال المتابعة:").strip()
            if original_answer:
                return f"{original_answer}\n\nإضافةً إلى ذلك، بخصوص سؤال المتابعة: {follow_up}"
            return f"لا تتوفر معلومات كافية للإجابة عن سؤال المتابعة: {follow_up}"

        if "النص:" in prompt and "ملخص" in prompt:
            text = self._extract_section(prompt, "النص:", "")
            return self._summarize_text(text)

        question = self._extract_section(prompt, "السؤال:", "الرجاء").strip()
        context = self._extract_section(prompt, "السياق والمعلومات ذات الصلة:", "السؤال:").strip()

        if context:
            return self._answer_from_context(question, context)

        if question:
            return f"لا يتوفر سياق كافٍ للإجابة بشكل موثوق عن السؤال: {question}"

        return self._summarize_text(prompt)

    def _answer_from_context(self, question: str, context: str) -> str:
        sentences = self._split_sentences(context)
        if not sentences:
            return "لا توجد معلومات كافية في السياق المتاح."

        query_terms = self._tokenize(question)
        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: (self._score_sentence(sentence, query_terms), len(sentence)),
            reverse=True,
        )

        top_sentences = []
        for sentence in ranked_sentences:
            if sentence not in top_sentences:
                top_sentences.append(sentence)
            if len(top_sentences) == 2:
                break

        body = " ".join(top_sentences).strip()
        if not body:
            body = sentences[0]

        return f"استناداً إلى المستندات المتاحة: {body}"

    def _summarize_text(self, text: str, max_sentences: int = 2) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text.strip()
        return " ".join(sentences[:max_sentences]).strip()

    def _split_sentences(self, text: str) -> List[str]:
        candidates = re.split(r'(?<=[\.\!\؟\n])\s+', text.strip())
        return [candidate.strip() for candidate in candidates if candidate.strip()]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[\w\u0600-\u06FF]+', text.lower())

    def _score_sentence(self, sentence: str, query_terms: List[str]) -> int:
        sentence_terms = set(self._tokenize(sentence))
        return sum(1 for term in query_terms if term in sentence_terms)

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        if start_marker not in text:
            return ""

        section = text.split(start_marker, 1)[1]
        if end_marker and end_marker in section:
            section = section.split(end_marker, 1)[0]

        return section.strip()


class ArabicResponseGenerator:
    """
    منشئ الإجابات العربي - Arabic Response Generator

    العربية:
        فئة متخصصة في توليد إجابات عربية باستخدام نماذج لغة متقدمة.
        تدعم عدة مزودي خدمات.

    English:
        Specialized class for generating Arabic responses using advanced language models.
        Supports multiple service providers.

    Attributes:
        config: GenerationConfig - التكوين المستخدم
        provider: LLMProvider - مزود النموذج

    Example:
        ```python
        generator = ArabicResponseGenerator()
        answer = generator.generate_answer(
            "ما هو القانون التجاري؟",
            context="القانون التجاري هو..."
        )
        ```
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        تهيئة منشئ الإجابات - Initialize generator

        Args:
            config: GenerationConfig - التكوين
        """
        self.config = config or GenerationConfig()
        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """
        تهيئة مزود النموذج - Initialize LLM provider

        العربية:
            إنشاء مزود النموذج المطلوب

        English:
            Initialize the required LLM provider.
        """
        provider_type = self.config.llm_provider.lower()

        if provider_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                self.provider = LocalExtractiveProvider()
                self.provider_name = "local"
                return

            try:
                self.provider = OpenAIProvider()
                self.provider_name = "openai"
            except ImportError:
                self.provider = LocalExtractiveProvider()
                self.provider_name = "local"
        elif provider_type == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                self.provider = LocalExtractiveProvider()
                self.provider_name = "local"
                return

            try:
                self.provider = AnthropicProvider()
                self.provider_name = "anthropic"
            except ImportError:
                self.provider = LocalExtractiveProvider()
                self.provider_name = "local"
        elif provider_type == "local":
            self.provider = LocalExtractiveProvider()
            self.provider_name = "local"
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    def generate_answer(self, question: str, context: str = "",
                       instructions: Optional[str] = None) -> str:
        """
        توليد إجابة - Generate an answer

        العربية:
            توليد إجابة شاملة على سؤال بناءً على السياق المتوفر

        English:
            Generate a comprehensive answer to a question based on provided context.

        Args:
            question: str - السؤال
            context: str - السياق / المستندات ذات الصلة
            instructions: str - تعليمات إضافية

        Returns:
            str - الإجابة المولدة

        Example:
            ```python
            answer = generator.generate_answer(
                question="ما هي شروط تسجيل شركة جديدة؟",
                context="وفقاً لنظام الشركات السعودي..."
            )
            ```
        """
        prompt = self._build_prompt(question, context, instructions)
        return self.provider.generate(prompt, self.config)

    def generate_with_references(self, question: str,
                                context_documents: List[str]) -> Dict[str, Any]:
        """
        توليد إجابة مع المراجع - Generate answer with references

        العربية:
            توليد إجابة مع الإشارة إلى المستندات المستخدمة

        English:
            Generate an answer while tracking which documents were used.

        Args:
            question: str - السؤال
            context_documents: List[str] - المستندات المستخدمة

        Returns:
            Dict - قاموس يحتوي على الإجابة والمراجع

        Example:
            ```python
            result = generator.generate_with_references(
                "ما حقوق المساهمين؟",
                ["نص المادة 1...", "نص المادة 2..."]
            )
            print(result['answer'])
            print(result['references'])
            ```
        """
        context = self._format_context_with_references(context_documents)
        answer = self.generate_answer(question, context)

        return {
            "answer": answer,
            "references": context_documents,
            "document_count": len(context_documents)
        }

    def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        """
        توليد ملخص - Generate a summary

        العربية:
            توليد ملخص موجز للنص

        English:
            Generate a concise summary of the text.

        Args:
            text: str - النص المراد تلخيصه
            max_length: int - الطول الأقصى للملخص

        Returns:
            str - الملخص

        Example:
            ```python
            summary = generator.generate_summary(long_text, max_length=100)
            ```
        """
        prompt = f"""
        استخرج ملخصاً دقيقاً وموجزاً للنص التالي باللغة العربية.
        الملخص يجب أن يكون واضحاً ومختصراً ويغطي النقاط الرئيسية.

        النص:
        {text}
        """
        return self.provider.generate(prompt, self.config)

    def refine_answer(self, original_answer: str, feedback: str) -> str:
        """
        تحسين الإجابة - Refine an answer

        العربية:
            تحسين إجابة سابقة بناءً على التغذية الراجعة

        English:
            Improve a previous answer based on feedback.

        Args:
            original_answer: str - الإجابة الأصلية
            feedback: str - التعليقات والتحسينات

        Returns:
            str - الإجابة المحسّنة

        Example:
            ```python
            refined = generator.refine_answer(
                original_answer="الإجابة الأولية...",
                feedback="الإجابة صحيحة لكن يجب إضافة المزيد من التفاصيل"
            )
            ```
        """
        prompt = f"""
        الإجابة الحالية:
        {original_answer}

        التعليقات والتحسينات المطلوبة:
        {feedback}

        الرجاء تحسين الإجابة بناءً على التعليقات المذكورة أعلاه، مع الحفاظ على الدقة والموثوقية.
        """
        return self.provider.generate(prompt, self.config)

    def answer_follow_up(self, original_question: str, original_answer: str,
                        follow_up_question: str) -> str:
        """
        الإجابة على سؤال متابعة - Answer a follow-up question

        العربية:
            الإجابة على سؤال متابعة بناءً على الإجابة السابقة

        English:
            Answer a follow-up question based on the previous answer.

        Args:
            original_question: str - السؤال الأول
            original_answer: str - الإجابة الأولى
            follow_up_question: str - سؤال المتابعة

        Returns:
            str - الإجابة على السؤال الجديد

        Example:
            ```python
            follow_up = generator.answer_follow_up(
                "ما هو القانون التجاري؟",
                "القانون التجاري هو...",
                "هل يوجد استثناءات في القانون التجاري؟"
            )
            ```
        """
        prompt = f"""
        السؤال الأول: {original_question}
        الإجابة على السؤال الأول: {original_answer}

        سؤال المتابعة: {follow_up_question}

        الرجاء الإجابة على سؤال المتابعة بناءً على الإجابة السابقة، مع الحفاظ على الاتساق والدقة.
        """
        return self.provider.generate(prompt, self.config)

    def _build_prompt(self, question: str, context: str = "",
                     instructions: Optional[str] = None) -> str:
        """
        بناء المحفز - Build the prompt

        العربية:
            بناء المحفز الكامل من السؤال والسياق والتعليمات

        English:
            Build a complete prompt from question, context, and instructions.

        Args:
            question: str - السؤال
            context: str - السياق
            instructions: str - التعليمات الإضافية

        Returns:
            str - المحفز الكامل
        """
        prompt = ""

        if instructions:
            prompt += f"التعليمات: {instructions}\n\n"

        if context:
            prompt += f"السياق والمعلومات ذات الصلة:\n{context}\n\n"

        prompt += f"السؤال: {question}\n\nالرجاء تقديم إجابة شاملة ودقيقة باللغة العربية."

        return prompt

    def _format_context_with_references(self, documents: List[str]) -> str:
        """
        تنسيق السياق مع المراجع - Format context with references

        العربية:
            تنسيق المستندات مع ترقيم المراجع

        English:
            Format documents with reference numbering.

        Args:
            documents: List[str] - المستندات

        Returns:
            str - السياق المنسق
        """
        formatted = ""
        for i, doc in enumerate(documents, 1):
            formatted += f"[{i}] {doc}\n"
        return formatted
