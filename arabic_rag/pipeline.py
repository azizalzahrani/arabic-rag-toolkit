"""
خط أنابيب RAG - Arabic RAG Pipeline Module

العربية:
    وحدة متخصصة في تنسيق جميع مكونات نظام RAG العربي في خط أنابيب واحد متكامل.

English:
    Specialized module for orchestrating all components of the Arabic RAG system
    into a single integrated pipeline.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from arabic_rag.preprocessor import ArabicTextPreprocessor, NormalizationConfig
from arabic_rag.chunker import ArabicTextChunker, ChunkingConfig
from arabic_rag.embeddings import ArabicEmbeddings, EmbeddingConfig
from arabic_rag.retriever import ArabicRetriever, RetrievalConfig
from arabic_rag.generator import ArabicResponseGenerator, GenerationConfig


@dataclass
class PipelineConfig:
    """
    إعدادات خط الأنابيب - Pipeline configuration

    العربية:
        تكوين شامل لجميع مكونات خط الأنابيب

    English:
        Comprehensive configuration for all pipeline components.
    """
    normalization_config: NormalizationConfig = None
    chunking_config: ChunkingConfig = None
    embedding_config: EmbeddingConfig = None
    retrieval_config: RetrievalConfig = None
    generation_config: GenerationConfig = None
    verbose: bool = False

    def __post_init__(self):
        """تهيئة التكوينات الافتراضية"""
        if self.normalization_config is None:
            self.normalization_config = NormalizationConfig()
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig()
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()
        if self.retrieval_config is None:
            self.retrieval_config = RetrievalConfig()
        if self.generation_config is None:
            self.generation_config = GenerationConfig()


class ArabicRAGPipeline:
    """
    خط أنابيب RAG العربي - Arabic RAG Pipeline

    العربية:
        فئة متخصصة في تنسيق جميع مكونات نظام RAG العربي في خط أنابيب واحد.
        تقوم بمعالجة المستندات والبحث والتوليد في تسلسل موحد.

    English:
        Specialized class orchestrating the complete Arabic RAG system.
        Handles document processing, retrieval, and generation in a unified flow.

    Attributes:
        config: PipelineConfig - التكوين الشامل
        preprocessor: ArabicTextPreprocessor - معالج النصوص
        chunker: ArabicTextChunker - معقم النصوص
        embeddings: ArabicEmbeddings - نموذج التضمين
        retriever: ArabicRetriever - محرك البحث
        generator: ArabicResponseGenerator - منشئ الإجابات

    Example:
        ```python
        pipeline = ArabicRAGPipeline()
        pipeline.add_documents(["وثيقة 1", "وثيقة 2"])
        answer = pipeline.query("سؤالي هنا")
        ```
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        تهيئة خط الأنابيب - Initialize the pipeline

        Args:
            config: PipelineConfig - التكوين
        """
        self.config = config or PipelineConfig()
        self._setup_logger()
        self._initialize_components()

    def _setup_logger(self) -> None:
        """إعداد نظام السجلات - Setup logging"""
        self.logger = logging.getLogger(__name__)
        if self.config.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def _initialize_components(self) -> None:
        """
        تهيئة جميع مكونات خط الأنابيب - Initialize all pipeline components

        العربية:
            إنشاء واحدة من كل مكون من مكونات النظام

        English:
            Create instances of all system components.
        """
        self.logger.debug("Initializing pipeline components...")

        self.preprocessor = ArabicTextPreprocessor(self.config.normalization_config)
        self.chunker = ArabicTextChunker(self.config.chunking_config)
        self.embeddings = ArabicEmbeddings(self.config.embedding_config)
        self.retriever = ArabicRetriever(self.config.retrieval_config, self.embeddings)
        self.generator = ArabicResponseGenerator(self.config.generation_config)

        self.documents = []
        self.logger.debug("Pipeline components initialized successfully")

    def add_documents(self, documents: List[str], preprocess: bool = True) -> None:
        """
        إضافة مستندات - Add documents to the pipeline

        العربية:
            إضافة مستندات جديدة إلى خط الأنابيب.
            يتم معالجة المستندات وتقطيعها وإضافتها إلى قاعدة البيانات المتجهة.

        English:
            Add new documents to the pipeline.
            Documents are processed, chunked, and added to vector database.

        Args:
            documents: List[str] - قائمة المستندات
            preprocess: bool - هل يتم معالجة المستندات مسبقاً

        Example:
            ```python
            docs = ["المستند الأول", "المستند الثاني"]
            pipeline.add_documents(docs)
            ```
        """
        self.logger.info(f"Adding {len(documents)} documents to pipeline...")

        # معالجة المستندات
        if preprocess:
            processed_docs = [
                self.preprocessor.normalize(doc) for doc in documents
            ]
        else:
            processed_docs = documents

        # تقطيع المستندات
        chunks = []
        for doc in processed_docs:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)

        self.logger.info(f"Documents split into {len(chunks)} chunks")

        # إضافة الأجزاء إلى المحرك
        self.retriever.add_documents(chunks)
        self.documents.extend(chunks)

        self.logger.info("Documents added and indexed successfully")

    def query(self, question: str, top_k: Optional[int] = None,
              return_sources: bool = True) -> str:
        """
        الاستعلام عن سؤال - Query the RAG system

        العربية:
            الاستعلام عن سؤال والحصول على إجابة موحدة من خلال
            استرجاع المستندات ذات الصلة وتوليد إجابة.

        English:
            Query the RAG system with a question and get an integrated answer
            by retrieving relevant documents and generating a response.

        Args:
            question: str - السؤال
            top_k: int - عدد المستندات المراد استرجاعها
            return_sources: bool - هل يتم إرجاع المصادر

        Returns:
            str - الإجابة

        Example:
            ```python
            answer = pipeline.query("ما هو القانون التجاري؟")
            print(answer)
            ```
        """
        self.logger.info(f"Processing query: {question}")

        # معالجة السؤال
        processed_question = self.preprocessor.normalize(question)
        self.logger.debug(f"Processed question: {processed_question}")

        # البحث عن المستندات ذات الصلة
        relevant_docs = self.retriever.retrieve(processed_question, top_k=top_k)
        self.logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            return "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك."

        # تنسيق السياق
        context = "\n".join([doc for doc, _ in relevant_docs])

        # توليد الإجابة
        self.logger.debug("Generating response...")
        answer = self.generator.generate_answer(question, context=context)

        if return_sources:
            answer += "\n\n---\n**المصادر:**\n"
            for i, (doc, score) in enumerate(relevant_docs, 1):
                answer += f"[{i}] (درجة التشابه: {score:.2f})\n"

        self.logger.info("Response generated successfully")
        return answer

    def query_with_context(self, question: str, additional_context: str = "",
                          top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        الاستعلام مع السياق الإضافي - Query with additional context

        العربية:
            الاستعلام مع توفير سياق إضافي لتحسين دقة الإجابة

        English:
            Query the system with additional context for better accuracy.

        Args:
            question: str - السؤال
            additional_context: str - سياق إضافي
            top_k: int - عدد المستندات المراد استرجاعها

        Returns:
            Dict - قاموس يحتوي على الإجابة والمعلومات المتعلقة بها
        """
        self.logger.info("Processing query with additional context...")

        # البحث عن المستندات
        processed_question = self.preprocessor.normalize(question)
        relevant_docs = self.retriever.retrieve(processed_question, top_k=top_k)

        # تجميع السياق
        context = ""
        if additional_context:
            context += f"سياق إضافي: {additional_context}\n\n"

        context += "\n".join([doc for doc, _ in relevant_docs])

        # توليد الإجابة
        answer = self.generator.generate_answer(question, context=context)

        return {
            "question": question,
            "answer": answer,
            "source_documents": [doc for doc, _ in relevant_docs],
            "similarity_scores": [score for _, score in relevant_docs],
            "document_count": len(relevant_docs)
        }

    def batch_query(self, questions: List[str], top_k: Optional[int] = None) -> List[str]:
        """
        الاستعلام عن عدة أسئلة - Query multiple questions

        العربية:
            الاستعلام عن عدة أسئلة دفعة واحدة

        English:
            Query multiple questions in batch.

        Args:
            questions: List[str] - قائمة الأسئلة
            top_k: int - عدد المستندات المراد استرجاعها

        Returns:
            List[str] - قائمة الإجابات

        Example:
            ```python
            questions = ["سؤال 1؟", "سؤال 2؟"]
            answers = pipeline.batch_query(questions)
            ```
        """
        self.logger.info(f"Processing batch of {len(questions)} questions...")
        answers = []
        for i, question in enumerate(questions, 1):
            self.logger.debug(f"Processing question {i}/{len(questions)}")
            answer = self.query(question, top_k=top_k, return_sources=False)
            answers.append(answer)
        return answers

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        الحصول على إحصائيات خط الأنابيب - Get pipeline statistics

        العربية:
            الحصول على معلومات إحصائية عن حالة خط الأنابيب

        English:
            Get statistical information about the pipeline state.

        Returns:
            Dict - إحصائيات

        Example:
            ```python
            stats = pipeline.get_pipeline_stats()
            print(f"عدد المستندات: {stats['total_documents']}")
            ```
        """
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embeddings.get_embedding_dimension(),
            "vector_store_type": self.config.retrieval_config.vector_store_type,
            "preprocessing_config": {
                "remove_diacritics": self.config.normalization_config.remove_diacritics,
                "normalize_alef": self.config.normalization_config.normalize_alef,
            },
            "chunking_config": {
                "chunk_size": self.config.chunking_config.chunk_size,
                "chunk_overlap": self.config.chunking_config.chunk_overlap,
            }
        }

    def save_pipeline(self, path: str) -> None:
        """
        حفظ خط الأنابيب - Save the pipeline

        العربية:
            حفظ حالة خط الأنابيب بما في ذلك المستندات والمتجهات

        English:
            Save the pipeline state including documents and vectors.

        Args:
            path: str - مسار الحفظ
        """
        self.logger.info(f"Saving pipeline to {path}...")
        self.retriever.save(path)
        self.logger.info("Pipeline saved successfully")

    def load_pipeline(self, path: str) -> None:
        """
        تحميل خط الأنابيب - Load the pipeline

        العربية:
            تحميل حالة خط الأنابيب المحفوظة سابقاً

        English:
            Load a previously saved pipeline state.

        Args:
            path: str - مسار التحميل
        """
        self.logger.info(f"Loading pipeline from {path}...")
        self.retriever.load(path)
        self.logger.info("Pipeline loaded successfully")

    def reset(self) -> None:
        """
        إعادة تعيين خط الأنابيب - Reset the pipeline

        العربية:
            مسح جميع المستندات والمتجهات وإعادة تهيئة النظام

        English:
            Clear all documents and vectors and reinitialize the system.
        """
        self.logger.info("Resetting pipeline...")
        self.documents = []
        self._initialize_components()
        self.logger.info("Pipeline reset successfully")
