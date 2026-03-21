"""
استرجاع الوثائق - Arabic Document Retriever Module

العربية:
    وحدة متخصصة في استرجاع الوثائق ذات الصلة من قاعدة البيانات المتجهة.
    تدعم FAISS و ChromaDB كمخازن متجهات.

English:
    Specialized module for retrieving relevant documents from vector databases.
    Supports FAISS and ChromaDB as vector store backends.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os


@dataclass
class RetrievalConfig:
    """
    إعدادات الاسترجاع - Retrieval configuration

    العربية:
        تكوين معاملات الاسترجاع والبحث

    English:
        Configuration for retrieval and search parameters.
    """
    vector_store_type: str = "chroma"  # chroma or faiss
    top_k: int = 5
    similarity_threshold: float = 0.0
    vector_store_path: str = "./data/vector_store"


class VectorStore(ABC):
    """
    واجهة قاعدة البيانات المتجهة - Vector Store Interface

    العربية:
        واجهة مجردة لقواعد البيانات المتجهة

    English:
        Abstract interface for vector store implementations.
    """

    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List) -> None:
        """إضافة مستندات"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """البحث عن مستندات متشابهة"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """حفظ قاعدة البيانات"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """تحميل قاعدة البيانات"""
        pass


class ChromaVectorStore(VectorStore):
    """
    متجر ChromaDB - ChromaDB Vector Store

    العربية:
        تطبيق ChromaDB كقاعدة بيانات متجهة

    English:
        ChromaDB implementation for vector storage.
    """

    def __init__(self, persist_directory: str = "./data/vector_store"):
        """
        تهيئة ChromaDB - Initialize ChromaDB

        Args:
            persist_directory: str - مسار حفظ البيانات
        """
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            raise ImportError("chromadb is required. Install it with: pip install chromadb")

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.documents = []
        self.embeddings = []

    def create_collection(self, collection_name: str = "arabic_documents") -> None:
        """
        إنشاء مجموعة - Create a collection

        Args:
            collection_name: str - اسم المجموعة
        """
        # حذف المجموعة القديمة إن وجدت
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], embeddings: List) -> None:
        """
        إضافة مستندات - Add documents

        Args:
            documents: List[str] - قائمة المستندات
            embeddings: List - قائمة التضمينات
        """
        if not self.collection:
            self.create_collection()

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            self.collection.add(
                documents=[doc],
                embeddings=[embedding.tolist()],
                ids=[f"doc_{i}"]
            )

        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        البحث - Search for similar documents

        Args:
            query_embedding: List[float] - متجه الاستعلام
            top_k: int - عدد النتائج

        Returns:
            List[Tuple[str, float]] - قائمة (مستند، درجة)
        """
        if not self.collection or not self.documents:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        if results['documents'] and results['distances']:
            for doc, distance in zip(results['documents'][0], results['distances'][0]):
                # تحويل المسافة إلى درجة تشابه
                similarity = 1 - distance
                output.append((doc, similarity))

        return output

    def save(self, path: str) -> None:
        """
        حفظ - Save data

        Args:
            path: str - مسار الحفظ
        """
        # ChromaDB يحفظ تلقائياً في مجلد persist_directory
        pass

    def load(self, path: str) -> None:
        """
        تحميل - Load data

        Args:
            path: str - مسار التحميل
        """
        self.persist_directory = path
        self.client = self.chromadb.PersistentClient(path=path)


class FAISSVectorStore(VectorStore):
    """
    متجر FAISS - FAISS Vector Store

    العربية:
        تطبيق FAISS كقاعدة بيانات متجهة

    English:
        FAISS implementation for vector storage.
    """

    def __init__(self, dimension: int = 384):
        """
        تهيئة FAISS - Initialize FAISS

        Args:
            dimension: int - بُعد التضمينات
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss is required. Install it with: pip install faiss-cpu")

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: List[str], embeddings: List) -> None:
        """
        إضافة مستندات - Add documents

        Args:
            documents: List[str] - قائمة المستندات
            embeddings: List - قائمة التضمينات
        """
        import numpy as np

        embeddings_array = np.array([e if isinstance(e, np.ndarray) else np.array(e) for e in embeddings]).astype('float32')
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        البحث - Search for similar documents

        Args:
            query_embedding: List[float] - متجه الاستعلام
            top_k: int - عدد النتائج

        Returns:
            List[Tuple[str, float]] - قائمة (مستند، درجة)
        """
        import numpy as np

        if not self.documents:
            return []

        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)

        output = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # تحويل المسافة إلى درجة تشابه (1 / (1 + distance))
                similarity = 1 / (1 + distance)
                output.append((self.documents[idx], float(similarity)))

        return output

    def save(self, path: str) -> None:
        """
        حفظ - Save data

        Args:
            path: str - مسار الحفظ
        """
        import pickle
        import os

        os.makedirs(path, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))

        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        """
        تحميل - Load data

        Args:
            path: str - مسار التحميل
        """
        import pickle
        import os

        self.index = self.faiss.read_index(os.path.join(path, "faiss_index.bin"))

        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)


class ArabicRetriever:
    """
    محرك البحث العربي - Arabic Document Retriever

    العربية:
        فئة متخصصة في استرجاع الوثائق ذات الصلة من قاعدة البيانات المتجهة.
        تدعم محركات بحث متعددة.

    English:
        Specialized class for retrieving relevant documents from vector databases.
        Supports multiple search engines.

    Attributes:
        config: RetrievalConfig - التكوين المستخدم
        vector_store: VectorStore - قاعدة البيانات المتجهة

    Example:
        ```python
        retriever = ArabicRetriever()
        results = retriever.retrieve("ما هو القانون التجاري؟", embeddings)
        ```
    """

    def __init__(self, config: Optional[RetrievalConfig] = None, embeddings=None):
        """
        تهيئة محرك البحث - Initialize retriever

        Args:
            config: RetrievalConfig - التكوين
            embeddings: ArabicEmbeddings - نموذج التضمين
        """
        self.config = config or RetrievalConfig()
        self.embeddings = embeddings
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """
        تهيئة قاعدة البيانات المتجهة - Initialize vector store

        العربية:
            إنشاء قاعدة البيانات المتجهة المطلوبة

        English:
            Initialize the required vector store.
        """
        if self.config.vector_store_type.lower() == "chroma":
            self.vector_store = ChromaVectorStore(self.config.vector_store_path)
        elif self.config.vector_store_type.lower() == "faiss":
            dimension = 384  # الحجم الافتراضي
            if self.embeddings:
                dimension = self.embeddings.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(dimension=dimension)
        else:
            raise ValueError(f"Unknown vector store type: {self.config.vector_store_type}")

    def add_documents(self, documents: List[str]) -> None:
        """
        إضافة مستندات - Add documents to the retriever

        العربية:
            إضافة مستندات إلى قاعدة البيانات المتجهة

        English:
            Add documents to the vector database.

        Args:
            documents: List[str] - قائمة المستندات
        """
        if not self.embeddings:
            raise ValueError("Embeddings model is required to add documents")

        embeddings = self.embeddings.embed_batch(documents, show_progress_bar=True)
        self.vector_store.add_documents(documents, embeddings)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        استرجاع مستندات - Retrieve relevant documents

        العربية:
            استرجاع أكثر المستندات تشابهاً مع الاستعلام

        English:
            Retrieve the most relevant documents for a query.

        Args:
            query: str - نص الاستعلام
            top_k: int - عدد النتائج (يستخدم الإعداد الافتراضي إن لم يُحدد)

        Returns:
            List[Tuple[str, float]] - قائمة (مستند، درجة تشابه)

        Example:
            ```python
            results = retriever.retrieve("تعريف الشركة المساهمة")
            for doc, score in results:
                print(f"{doc} (Score: {score:.2f})")
            ```
        """
        if not self.embeddings:
            raise ValueError("Embeddings model is required for retrieval")

        if not top_k:
            top_k = self.config.top_k

        # الحصول على تضمين الاستعلام
        query_embedding = self.embeddings.embed_text(query)

        # البحث في قاعدة البيانات
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # تصفية حسب العتبة
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= self.config.similarity_threshold
        ]

        return filtered_results

    def save(self, path: str) -> None:
        """
        حفظ محرك البحث - Save retriever

        Args:
            path: str - مسار الحفظ
        """
        self.vector_store.save(path)

    def load(self, path: str) -> None:
        """
        تحميل محرك البحث - Load retriever

        Args:
            path: str - مسار التحميل
        """
        self.vector_store.load(path)
