"""
خط أنابيب RAG - Arabic RAG Pipeline Module

العربية:
    وحدة متخصصة في تنسيق جميع مكونات نظام RAG العربي في خط أنابيب واحد متكامل.

English:
    Specialized module for orchestrating all components of the Arabic RAG system
    into a single integrated pipeline.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

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
    normalization_config: Optional[NormalizationConfig] = None
    chunking_config: Optional[ChunkingConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    retrieval_config: Optional[RetrievalConfig] = None
    generation_config: Optional[GenerationConfig] = None
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


def config_from_env() -> PipelineConfig:
    """
    بناء التكوين من متغيرات البيئة - Build a PipelineConfig from environment variables

    العربية:
        قراءة المتغيرات الموثقة في ‎.env.example‎ وبناء تكوين كامل منها.
        أي متغير غير معرّف يستخدم القيمة الافتراضية.

    English:
        Read the variables documented in `.env.example` and build a full config.
        Any unset variable falls back to its default value.

    Supported variables:
        EMBEDDING_MODEL, VECTOR_STORE, VECTOR_STORE_PATH, LLM_PROVIDER,
        OPENAI_MODEL / ANTHROPIC_MODEL (read by the generator),
        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, TEMPERATURE, MAX_TOKENS

    Example:
        ```python
        from arabic_rag.pipeline import ArabicRAGPipeline, config_from_env

        pipeline = ArabicRAGPipeline(config=config_from_env())
        # أو ببساطة - or simply:
        pipeline = ArabicRAGPipeline.from_env()
        ```
    """
    config = PipelineConfig()

    if os.getenv("EMBEDDING_MODEL"):
        config.embedding_config.model_name = os.environ["EMBEDDING_MODEL"]
    if os.getenv("VECTOR_STORE"):
        config.retrieval_config.vector_store_type = os.environ["VECTOR_STORE"]
    if os.getenv("VECTOR_STORE_PATH"):
        config.retrieval_config.vector_store_path = os.environ["VECTOR_STORE_PATH"]
    if os.getenv("LLM_PROVIDER"):
        config.generation_config.llm_provider = os.environ["LLM_PROVIDER"]
    if os.getenv("CHUNK_SIZE"):
        config.chunking_config.chunk_size = int(os.environ["CHUNK_SIZE"])
    if os.getenv("CHUNK_OVERLAP"):
        config.chunking_config.chunk_overlap = int(os.environ["CHUNK_OVERLAP"])
    if os.getenv("TOP_K"):
        config.retrieval_config.top_k = int(os.environ["TOP_K"])
    if os.getenv("TEMPERATURE"):
        config.generation_config.temperature = float(os.environ["TEMPERATURE"])
    if os.getenv("MAX_TOKENS"):
        config.generation_config.max_tokens = int(os.environ["MAX_TOKENS"])

    return config


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
        chunker: ArabicTextChunker - مُقطِّع النصوص
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

    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        """
        تهيئة خط الأنابيب - Initialize the pipeline

        Args:
            config: PipelineConfig - التكوين
        """
        self.config = self._build_config(config, kwargs)
        self._setup_logger()
        self._initialize_components()

    @classmethod
    def from_env(cls, **kwargs) -> "ArabicRAGPipeline":
        """
        إنشاء خط أنابيب من متغيرات البيئة - Build a pipeline from environment variables

        العربية:
            إنشاء خط الأنابيب اعتماداً على المتغيرات الموثقة في ‎.env.example‎،
            مع إمكانية تمرير اختصارات إضافية تتجاوز قيم البيئة.

        English:
            Build the pipeline from the variables documented in `.env.example`.
            Extra keyword shortcuts override environment values.

        Example:
            ```python
            pipeline = ArabicRAGPipeline.from_env()
            ```
        """
        return cls(config=config_from_env(), **kwargs)

    def _build_config(self, config: Optional[PipelineConfig], overrides: Dict[str, Any]) -> PipelineConfig:
        """
        بناء التكوين - Build configuration

        العربية:
            دعم واجهة التكوين الكاملة مع اختصارات سهلة متوافقة مع الأمثلة.

        English:
            Support both full PipelineConfig and shortcut keyword arguments used by examples.
        """
        pipeline_config = config or PipelineConfig()
        overrides = dict(overrides)

        if "verbose" in overrides:
            pipeline_config.verbose = overrides.pop("verbose")

        if "embedding_model" in overrides:
            pipeline_config.embedding_config.model_name = overrides.pop("embedding_model")

        if "vector_store" in overrides:
            pipeline_config.retrieval_config.vector_store_type = overrides.pop("vector_store")

        if "vector_store_path" in overrides:
            pipeline_config.retrieval_config.vector_store_path = overrides.pop("vector_store_path")

        if "llm_provider" in overrides:
            pipeline_config.generation_config.llm_provider = overrides.pop("llm_provider")

        if "llm_model" in overrides:
            pipeline_config.generation_config.model_name = overrides.pop("llm_model")

        if "chunk_size" in overrides:
            pipeline_config.chunking_config.chunk_size = overrides.pop("chunk_size")

        if "chunk_overlap" in overrides:
            pipeline_config.chunking_config.chunk_overlap = overrides.pop("chunk_overlap")

        if "min_chunk_size" in overrides:
            pipeline_config.chunking_config.min_chunk_size = overrides.pop("min_chunk_size")

        if "similarity_threshold" in overrides:
            pipeline_config.retrieval_config.similarity_threshold = overrides.pop("similarity_threshold")

        if overrides:
            unknown_arguments = ", ".join(sorted(overrides.keys()))
            raise TypeError(f"Unknown pipeline arguments: {unknown_arguments}")

        return pipeline_config

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
            preprocess: bool - استخدام نسخة مطبّعة للتضمين والبحث
                (يبقى النص الأصلي محفوظاً كما هو للعرض في الإجابات والمصادر)

        Example:
            ```python
            docs = ["المستند الأول", "المستند الثاني"]
            pipeline.add_documents(docs)
            ```
        """
        self.logger.info(f"Adding {len(documents)} documents to pipeline...")

        # تقطيع المستندات الأصلية (للحفاظ على التشكيل والهمزات عند العرض)
        # Chunk the ORIGINAL documents so the stored text keeps its
        # diacritics and hamzas for display.
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)

        self.logger.info(f"Documents split into {len(chunks)} chunks")

        # التطبيع يُستخدم للتضمين فقط حتى يتطابق مع الاستعلامات المطبّعة
        # Normalization is applied to the embedded copy only, matching
        # the normalized queries used at retrieval time.
        if preprocess:
            embedding_texts = [self.preprocessor.normalize(chunk) for chunk in chunks]
        else:
            embedding_texts = None

        # إضافة الأجزاء إلى المحرك
        self.retriever.add_documents(chunks, embedding_texts=embedding_texts)
        self.documents.extend(chunks)

        self.logger.info("Documents added and indexed successfully")

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[tuple]:
        """
        استرجاع المستندات - Retrieve relevant documents

        العربية:
            واجهة متوافقة مع الأمثلة لاسترجاع المستندات فقط بدون توليد إجابة.

        English:
            Example-friendly API for retrieving documents without generating an answer.
        """
        processed_question = self.preprocessor.normalize_query(question)
        return self.retriever.retrieve(processed_question, top_k=top_k)

    def generate_answer(
        self,
        retrieval_results: List[tuple],
        question: str,
        instructions: Optional[str] = None,
        return_sources: bool = False,
    ) -> str:
        """
        توليد إجابة من نتائج الاسترجاع - Generate an answer from retrieval results

        العربية:
            واجهة متوافقة مع الأمثلة لبناء إجابة من النتائج المسترجعة.

        English:
            Example-friendly API to build an answer from retrieved results.
        """
        documents = []
        for item in retrieval_results:
            if isinstance(item, tuple):
                documents.append(item[0])
            else:
                documents.append(item)

        context = "\n".join(documents)
        answer = self.generator.generate_answer(question, context=context, instructions=instructions)

        if return_sources and retrieval_results:
            answer += "\n\n---\n**المصادر:**\n"
            for index, (document, score) in enumerate(
                zip(documents, retrieval_results), 1
            ):
                snippet = document if len(document) <= 100 else document[:100].rstrip() + "..."
                score_value = score[1] if isinstance(score, tuple) and len(score) > 1 else None
                if score_value is None:
                    answer += f"[{index}] {snippet}\n"
                else:
                    answer += f"[{index}] {snippet} (درجة التشابه: {score_value:.2f})\n"

        return answer

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
        relevant_docs = self.retrieve(question, top_k=top_k)
        self.logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            return "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك."

        self.logger.debug("Generating response...")
        answer = self.generate_answer(relevant_docs, question, return_sources=return_sources)

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
            "vector_store_type": getattr(self.retriever, "vector_store_type", self.config.retrieval_config.vector_store_type),
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
            مسح جميع المستندات والمتجهات وإعادة تهيئة المخزن المتجه
            مع الاحتفاظ بنموذج التضمين المحمّل لتجنب إعادة التحميل البطيئة.

        English:
            Clear all documents and vectors and reinitialize the vector store,
            keeping the already-loaded embedding model to avoid a slow reload.
        """
        self.logger.info("Resetting pipeline...")
        self.documents = []
        self.retriever = ArabicRetriever(self.config.retrieval_config, self.embeddings)
        self.logger.info("Pipeline reset successfully")
