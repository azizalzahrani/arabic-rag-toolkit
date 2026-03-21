"""
نماذج التضمين العربية - Arabic Embeddings Module

العربية:
    وحدة متخصصة في استخدام نماذج التضمين المتخصصة في اللغة العربية.
    تدعم نماذج متعددة بما فيها CAMeL وAraBART والنماذج متعددة اللغات.

English:
    Specialized module for Arabic-optimized embedding models.
    Supports multiple models including CAMeL, AraBART, and multilingual models.
"""

from typing import List, Optional
from dataclasses import dataclass
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
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )

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

        embedding = self.model.encode(text, convert_to_numpy=True)

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

        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar
        )

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

        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def most_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[tuple]:
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

        query_embedding = self.embed_text(query)
        text_embeddings = self.embed_batch(texts)

        # حساب درجات التشابه
        scores = []
        for i, text_embedding in enumerate(text_embeddings):
            score = float(
                np.dot(query_embedding, text_embedding) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding))
            )
            scores.append((score, texts[i]))

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
        # نموذج تجريبي بسيط
        test_embedding = self.embed_text("test")
        return len(test_embedding)

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
