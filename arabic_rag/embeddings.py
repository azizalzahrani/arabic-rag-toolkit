"""
نماذج التضمين العربية - Arabic Embeddings Module

العربية:
    وحدة متخصصة في استخدام نماذج التضمين المتخصصة في اللغة العربية.
    تدعم نماذج متعددة بما فيها CAMeL وAraBART والنماذج متعددة اللغات.

English:
    Specialized module for Arabic-optimized embedding models.
    Supports multiple models including CAMeL, AraBART, and multilingual models.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np


@dataclass
class EmbeddingConfig:
    """
    إعدادات التضمين - Embedding configuration

    العربية:
        تكوين معاملات التضمين والنماذج

    English:
        Configuration for embedding parameters and models.
    """
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"
    fallback_dimension: int = 384


class ArabicEmbeddings:
    """
    نموذج التضمين العربي - Arabic Embeddings Model

    العربية:
        فئة متخصصة في توليد تضمينات (embeddings) للنصوص العربية.
        تستخدم نماذج متطورة مدربة على نصوص عربية.

    English:
        Specialized class for generating embeddings for Arabic text.
        Uses advanced models trained on Arabic content.

    Attributes:
        config: EmbeddingConfig - التكوين المستخدم
        model: SentenceTransformer - النموذج

    Example:
        ```python
        embeddings = ArabicEmbeddings()
        vector = embeddings.embed_text("مرحبا بك")
        vectors = embeddings.embed_batch(["النص الأول", "النص الثاني"])
        ```
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        تهيئة نموذج التضمين - Initialize embedding model

        Args:
            config: EmbeddingConfig - التكوين
        """
        self.config = config or EmbeddingConfig()
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        تهيئة النموذج - Initialize the model

        العربية:
            تحميل نموذج التضمين المطلوب

        English:
            Load the required embedding model.
        """
        self._dimension: Optional[int] = None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.model = None
            self.backend = "hashing"
            return

        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            self.backend = "sentence-transformers"
        except Exception as error:  # network failures, missing model files, etc.
            warnings.warn(
                f"Could not load sentence-transformers model "
                f"'{self.config.model_name}' ({error}). "
                "Falling back to the local hashing backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.model = None
            self.backend = "hashing"

    def embed_text(self, text: str) -> np.ndarray:
        """
        توليد تضمين لنص واحد - Generate embedding for a single text

        العربية:
            توليد متجه التضمين لنص واحد

        English:
            Generate embedding vector for a single text.

        Args:
            text: str - النص

        Returns:
            np.ndarray - متجه التضمين

        Example:
            ```python
            vector = embeddings.embed_text("هذا نص عربي")
            print(vector.shape)  # (384,)
            ```
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if self.backend == "sentence-transformers":
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._hash_embed(text)

        if self.config.normalize:
            embedding = self._normalize_vector(embedding)

        return embedding

    def embed_batch(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        توليد تضمينات لقائمة نصوص - Generate embeddings for multiple texts

        العربية:
            توليد متجهات التضمين لعدة نصوص دفعة واحدة

        English:
            Generate embedding vectors for multiple texts at once.

        Args:
            texts: List[str] - قائمة النصوص
            show_progress_bar: bool - عرض شريط التقدم

        Returns:
            np.ndarray - مصفوفة التضمينات

        Example:
            ```python
            texts = ["النص الأول", "النص الثاني"]
            vectors = embeddings.embed_batch(texts)
            print(vectors.shape)  # (2, 384)
            ```
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty or whitespace")

        if self.backend == "sentence-transformers":
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress_bar
            )
        else:
            embeddings = np.array([self._hash_embed(text) for text in valid_texts])

        if self.config.normalize:
            embeddings = np.array([self._normalize_vector(e) for e in embeddings])

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        حساب التشابه بين نصين - Calculate similarity between two texts

        العربية:
            حساب درجة التشابه (من 0 إلى 1) بين نصين باستخدام تضميناتهما

        English:
            Calculate similarity score (0-1) between two texts using cosine similarity.

        Args:
            text1: str - النص الأول
            text2: str - النص الثاني

        Returns:
            float - درجة التشابه (0-1)

        Example:
            ```python
            score = embeddings.similarity("كتب", "كتابة")
            print(score)  # 0.85
            ```
        """
        if not text1 or not text1.strip() or not text2 or not text2.strip():
            raise ValueError("Both texts must be non-empty")

        vec1 = self.embed_text(text1)
        vec2 = self.embed_text(text2)

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / denominator)

    def most_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
        """
        إيجاد أكثر النصوص تشابهاً - Find most similar texts

        العربية:
            إيجاد أكثر النصوص تشابهاً مع نص الاستعلام

        English:
            Find the most similar texts to a query text.

        Args:
            query: str - نص الاستعلام
            texts: List[str] - قائمة النصوص للمقارنة
            top_k: int - عدد النتائج المطلوبة

        Returns:
            List[tuple] - قائمة (نص، درجة) مرتبة

        Example:
            ```python
            results = embeddings.most_similar(
                "تعليم",
                ["المدرسة", "الجامعة", "البيت"],
                top_k=2
            )
            # [(0.9, "المدرسة"), (0.85, "الجامعة")]
            ```
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # نحافظ على نفس التصفية المستخدمة في embed_batch حتى تبقى الفهارس متطابقة
        # Keep the same filtering used by embed_batch so indices stay aligned.
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty or whitespace")

        query_embedding = self.embed_text(query)
        text_embeddings = self.embed_batch(valid_texts)
        query_norm = np.linalg.norm(query_embedding)

        # حساب درجات التشابه
        scores = []
        for i, text_embedding in enumerate(text_embeddings):
            denominator = query_norm * np.linalg.norm(text_embedding)
            score = float(np.dot(query_embedding, text_embedding) / denominator) if denominator else 0.0
            scores.append((score, valid_texts[i]))

        # ترتيب تنازلي
        scores.sort(reverse=True, key=lambda x: x[0])

        return scores[:top_k]

    def get_embedding_dimension(self) -> int:
        """
        الحصول على بُعد التضمين - Get embedding dimension

        العربية:
            الحصول على حجم متجه التضمين (عدد الأبعاد)

        English:
            Get the dimensionality of the embedding vectors.

        Returns:
            int - عدد الأبعاد

        Example:
            ```python
            dim = embeddings.get_embedding_dimension()
            print(dim)  # 384
            ```
        """
        if self._dimension is not None:
            return self._dimension

        if self.backend == "sentence-transformers":
            model_dimension = self.model.get_sentence_embedding_dimension()
            self._dimension = model_dimension or len(self.embed_text("اختبار"))
        else:
            self._dimension = self.config.fallback_dimension

        return self._dimension

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        تطبيع المتجه - Normalize vector

        العربية:
            تطبيع المتجه بحيث يصبح طوله 1

        English:
            Normalize vector to unit length.

        Args:
            vector: np.ndarray - المتجه

        Returns:
            np.ndarray - المتجه المطبّع
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        تضمين محلي خفيف - Lightweight local embedding

        العربية:
            إنشاء تضمين حتمي بدون الاعتماد على مكتبات خارجية.

        English:
            Build a deterministic embedding without external model dependencies.
        """
        dimension = self.config.fallback_dimension
        vector = np.zeros(dimension, dtype=np.float32)

        tokens = self._extract_features(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in (0, 5):
                index = int.from_bytes(digest[offset:offset + 4], "big") % dimension
                sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
                vector[index] += sign

        return vector

    def _extract_features(self, text: str) -> List[str]:
        """
        استخراج سمات نصية - Extract text features

        العربية:
            مزج الكلمات وثلاثيات الأحرف لتحسين الاسترجاع المحلي البسيط.

        English:
            Blend tokens and character trigrams for simple local retrieval quality.
        """
        normalized = " ".join(text.strip().split())
        if not normalized:
            return []

        tokens = normalized.split()
        compact = normalized.replace(" ", "")
        trigrams = [compact[i:i + 3] for i in range(max(len(compact) - 2, 0))]

        if not trigrams and compact:
            trigrams = [compact]

        return tokens + trigrams

    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """
        حفظ التضمينات - Save embeddings to file

        العربية:
            حفظ مصفوفة التضمينات في ملف

        English:
            Save embeddings matrix to a file.

        Args:
            embeddings: np.ndarray - مصفوفة التضمينات
            filepath: str - مسار الملف
        """
        np.save(filepath, embeddings)

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        تحميل التضمينات - Load embeddings from file

        العربية:
            تحميل مصفوفة التضمينات من ملف

        English:
            Load embeddings matrix from a file.

        Args:
            filepath: str - مسار الملف

        Returns:
            np.ndarray - مصفوفة التضمينات
        """
        return np.load(filepath)
