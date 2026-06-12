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
import warnings


@dataclass
class RetrievalConfig:
    """
    إعدادات الاسترجاع - Retrieval configuration

    العربية:
        تكوين معاملات الاسترجاع والبحث

    English:
        Configuration for retrieval and search parameters.
    """
    vector_store_type: str = "memory"  # memory, chroma or faiss
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


class MemoryVectorStore(VectorStore):
    """
    متجر ذاكرة محلي - In-memory vector store

    العربية:
        تخزين بسيط داخل الذاكرة لا يحتاج إلى تبعيات خارجية.

    English:
        Lightweight in-memory store that does not require external dependencies.
    """

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: List[str], embeddings: List) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(list(embeddings))

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        import numpy as np

        if not self.documents:
            return []

        query = np.array(query_embedding, dtype="float32")
        query_norm = np.linalg.norm(query)

        matrix = np.array(self.embeddings, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1) * query_norm
        dots = matrix @ query
        similarities = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)

        order = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], float(similarities[i])) for i in order]

    def save(self, path: str) -> None:
        import pickle

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "memory_store.pkl"), "wb") as file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                },
                file,
            )

    def load(self, path: str) -> None:
        import pickle

        with open(os.path.join(path, "memory_store.pkl"), "rb") as file:
            payload = pickle.load(file)

        self.documents = payload.get("documents", [])
        self.embeddings = payload.get("embeddings", [])


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
        self.collection_name = "arabic_documents"
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
        except Exception:
            pass

        self.collection_name = collection_name
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

        if len(documents) == 0:
            return

        start_index = len(self.documents)
        embedding_lists = [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]
        self.collection.add(
            documents=list(documents),
            embeddings=embedding_lists,
            ids=[f"doc_{i}" for i in range(start_index, start_index + len(documents))],
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

        query_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=min(top_k, len(self.documents))
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

        العربية:
            إعادة فتح المجموعة المحفوظة واستعادة قائمة المستندات حتى يعمل البحث مباشرة.

        English:
            Reopen the persisted collection and restore the documents list
            so that search works immediately after loading.

        Args:
            path: str - مسار التحميل
        """
        self.persist_directory = path
        self.client = self.chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        payload = self.collection.get(include=["documents", "embeddings"])
        self.documents = payload.get("documents") or []
        embeddings = payload.get("embeddings")
        self.embeddings = list(embeddings) if embeddings is not None else []


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
            if 0 <= idx < len(self.documents):
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
        requested_store = self.config.vector_store_type.lower()

        if requested_store == "memory":
            self.vector_store = MemoryVectorStore()
            self.vector_store_type = "memory"
        elif requested_store == "chroma":
            try:
                self.vector_store = ChromaVectorStore(self.config.vector_store_path)
                self.vector_store_type = "chroma"
            except ImportError as error:
                self._fall_back_to_memory("chroma", error)
        elif requested_store == "faiss":
            dimension = 384  # الحجم الافتراضي
            if self.embeddings:
                dimension = self.embeddings.get_embedding_dimension()
            try:
                self.vector_store = FAISSVectorStore(dimension=dimension)
                self.vector_store_type = "faiss"
            except ImportError as error:
                self._fall_back_to_memory("faiss", error)
        else:
            raise ValueError(f"Unknown vector store type: {self.config.vector_store_type}")

    def _fall_back_to_memory(self, requested_store: str, error: Exception) -> None:
        """
        التراجع إلى متجر الذاكرة - Fall back to the in-memory store

        العربية:
            استخدام متجر الذاكرة مع تحذير يوضح السبب.

        English:
            Use the in-memory store and warn about why.
        """
        warnings.warn(
            f"Requested vector store '{requested_store}' is unavailable ({error}). "
            "Falling back to the in-memory store.",
            RuntimeWarning,
            stacklevel=3,
        )
        self.vector_store = MemoryVectorStore()
        self.vector_store_type = "memory"

    def add_documents(self, documents: List[str],
                      embedding_texts: Optional[List[str]] = None) -> None:
        """
        إضافة مستندات - Add documents to the retriever

        العربية:
            إضافة مستندات إلى قاعدة البيانات المتجهة. يمكن تمرير نسخ مطبّعة
            من النصوص لاستخدامها في التضمين مع الاحتفاظ بالنص الأصلي للعرض.

        English:
            Add documents to the vector database. Optionally pass normalized
            variants used only for embedding, while the original text is what
            gets stored and returned by search.

        Args:
            documents: List[str] - قائمة المستندات (كما ستُعرض)
            embedding_texts: Optional[List[str]] - نصوص التضمين المطبّعة
                (نفس الطول؛ يستخدم documents إن لم تُحدد)
        """
        if not self.embeddings:
            raise ValueError("Embeddings model is required to add documents")

        if embedding_texts is not None and len(embedding_texts) != len(documents):
            raise ValueError("embedding_texts must match documents in length")

        texts_to_embed = embedding_texts if embedding_texts is not None else documents
        embeddings = self.embeddings.embed_batch(texts_to_embed, show_progress_bar=True)
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
