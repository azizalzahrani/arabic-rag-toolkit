"""
تقطيع النصوص العربية - Arabic Text Chunking Module

العربية:
    وحدة متخصصة في تقطيع النصوص العربية بذكاء، مع الحفاظ على سلامة الكلمات والجمل.
    تتعامل مع خصائص اللغة العربية مثل التصريفات والبادئات واللواحق.

English:
    Specialized module for intelligent Arabic text chunking that preserves word
    and sentence integrity. Handles Arabic morphology, prefixes, and suffixes.
"""

from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class ChunkingConfig:
    """
    إعدادات التقطيع - Chunking configuration

    العربية:
        تكوين خيارات التقطيع مثل حجم الجزء والتداخل

    English:
        Configuration options for chunk size and overlap.
    """
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 1
    split_on_sentences: bool = True
    preserve_paragraphs: bool = True


class ArabicTextChunker:
    """
    مُقطِّع النصوص العربية - Arabic Text Chunker

    العربية:
        فئة متخصصة في تقطيع النصوص العربية بشكل ذكي. تحاول تقسيم النص عند
        حدود الجمل بدلاً من منتصف الكلمات، مما يحافظ على السياق.

    English:
        Specialized class for intelligent Arabic text chunking. Attempts to split
        at sentence boundaries rather than mid-word, preserving context.

    Attributes:
        config: ChunkingConfig - التكوين المستخدم

    Example:
        ```python
        chunker = ArabicTextChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(long_text)
        for chunk in chunks:
            print(chunk)
        ```
    """

    # علامات نهاية الجملة - Sentence-ending punctuation
    SENTENCE_DELIMITERS = (
        '۔',      # Arabic full stop
        '؟',      # Arabic question mark
        '؛',      # Arabic semicolon
        '!',      # Exclamation
        '.',      # Period
        '?',      # Question mark
    )

    # علامات الفواصل - Clause punctuation
    CLAUSE_DELIMITERS = (
        '،',      # Arabic comma
        ',',      # Regular comma
    )

    def __init__(self, config: Optional[ChunkingConfig] = None, **kwargs):
        """
        تهيئة مُقطِّع النصوص - Initialize the chunker

        Args:
            config: ChunkingConfig - التكوين (يستخدم الافتراضي إن لم يُحدد)
            **kwargs: اختصارات تكوين مثل chunk_size و chunk_overlap -
                ChunkingConfig field shortcuts, e.g. chunk_size, chunk_overlap

        Example:
            ```python
            chunker = ArabicTextChunker(chunk_size=300, chunk_overlap=50)
            ```
        """
        if config is not None and kwargs:
            raise TypeError("Pass either a ChunkingConfig or keyword shortcuts, not both")

        if kwargs:
            config = ChunkingConfig(**kwargs)

        self.config = config or ChunkingConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """
        التحقق من صحة التكوين - Validate configuration

        العربية:
            التحقق من أن قيم التكوين منطقية

        English:
            Ensure configuration values are logically sound.
        """
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.config.min_chunk_size > self.config.chunk_size:
            raise ValueError("min_chunk_size must be less than or equal to chunk_size")

    def chunk(self, text: str) -> List[str]:
        """
        تقطيع النص - Chunk the text

        العربية:
            تقسيم النص إلى أجزاء بحجم محدد مع الحفاظ على سلامة الكلمات والجمل.

        English:
            Split text into chunks while preserving word and sentence integrity.

        Args:
            text: str - النص المراد تقطيعه

        Returns:
            List[str] - قائمة الأجزاء

        Example:
            ```python
            chunks = chunker.chunk("نص طويل جداً...")
            # Returns: ["جزء 1", "جزء 2", ...]
            ```
        """
        if not text or len(text.strip()) == 0:
            return []

        # الخطوة 1: تقسيم إلى فقرات إن كانت مفعلة
        if self.config.preserve_paragraphs:
            paragraphs = self._split_paragraphs(text)
            chunks = []
            for paragraph in paragraphs:
                para_chunks = self._chunk_text(paragraph)
                chunks.extend(para_chunks)
            return chunks
        else:
            return self._chunk_text(text)

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        تقسيم النص إلى فقرات - Split text into paragraphs

        العربية:
            تقسيم النص الأولي إلى فقرات بناءً على الأسطر الفارغة

        English:
            Initial split of text into paragraphs based on blank lines.
        """
        # تقسيم بناءً على الأسطر الفارغة (سطر واحد أو أكثر)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_text(self, text: str) -> List[str]:
        """
        تقطيع نص واحد - Chunk a single text block

        العربية:
            تقطيع نص واحد إلى أجزاء بحسب الحجم والتداخل

        English:
            Chunk a single text block by size and overlap.
        """
        if len(text) <= self.config.chunk_size:
            stripped = text.strip()
            return [stripped] if len(stripped) >= self.config.min_chunk_size else []

        chunks = []
        sentences = self._split_sentences(text) if self.config.split_on_sentences else [text]

        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # إذا كانت الجملة الواحدة أكبر من حجم الجزء، قطّعها
            if sentence_size > self.config.chunk_size:
                # أولاً، أضف الجزء الحالي إن كان غير فارغ
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # ثم قطّع الجملة الطويلة
                long_sentence_chunks = self._chunk_long_sentence(sentence)
                chunks.extend(long_sentence_chunks)
                current_chunk = ""
                current_size = 0
            # إذا كان إضافة الجملة سيتجاوز الحد
            elif current_size + sentence_size > self.config.chunk_size:
                # أضف الجزء الحالي
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # ابدأ جزء جديد
                current_chunk = sentence
                current_size = sentence_size
            else:
                # أضف الجملة إلى الجزء الحالي
                if current_chunk:
                    current_chunk += " " + sentence
                    current_size += 1 + sentence_size
                else:
                    current_chunk = sentence
                    current_size = sentence_size

        # أضف الجزء الأخير
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())

        # تطبيق التداخل
        if len(chunks) > 1 and self.config.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        تقسيم النص إلى جمل - Split text into sentences

        العربية:
            تقسيم النص عند علامات الترقيم مع الحفاظ على العلامة

        English:
            Split text at punctuation marks while preserving delimiters.
        """
        # إنشاء نمط يطابق أي علامة ترقيم
        delimiter_pattern = '([' + ''.join(re.escape(d) for d in self.SENTENCE_DELIMITERS) + '])'

        sentences = re.split(delimiter_pattern, text)

        # إعادة بناء الجمل مع علاماتها
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in self.SENTENCE_DELIMITERS:
                # جملة + علامة ترقيم
                sentence = sentences[i] + sentences[i + 1]
                result.append(sentence.strip())
                i += 2
            elif sentences[i].strip():
                result.append(sentences[i].strip())
                i += 1
            else:
                i += 1

        return [s for s in result if s]  # إزالة العناصر الفارغة

    def _chunk_long_sentence(self, sentence: str) -> List[str]:
        """
        تقطيع جملة طويلة - Chunk a long sentence

        العربية:
            عندما تكون جملة واحدة أطول من حجم الجزء المطلوب،
            نقطعها حسب الفواصل والكلمات

        English:
            When a single sentence exceeds chunk_size, split it by clauses and words.
        """
        if len(sentence) <= self.config.chunk_size:
            return [sentence]

        chunks = []
        current = ""

        # محاولة التقسيم بالفواصل أولاً
        clauses = self._split_clauses(sentence)

        for clause in clauses:
            clause_size = len(clause)

            if clause_size > self.config.chunk_size:
                # قطّع الفاصلة الطويلة حسب الكلمات
                if current:
                    chunks.append(current.strip())
                    current = ""

                words = clause.split()
                for word in words:
                    if len(current) + len(word) + 1 > self.config.chunk_size:
                        if current:
                            chunks.append(current.strip())
                        current = word
                    else:
                        if current:
                            current += " " + word
                        else:
                            current = word
            else:
                if len(current) + clause_size + 1 > self.config.chunk_size:
                    if current:
                        chunks.append(current.strip())
                    current = clause
                else:
                    if current:
                        current += " " + clause
                    else:
                        current = clause

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if len(c) >= self.config.min_chunk_size]

    def _split_clauses(self, text: str) -> List[str]:
        """
        تقسيم النص إلى فواصل - Split text into clauses

        العربية:
            تقسيم النص عند الفواصل (الفاصلة)

        English:
            Split text at clause delimiters (commas, semicolons).
        """
        delimiter_pattern = '([' + ''.join(re.escape(d) for d in self.CLAUSE_DELIMITERS) + '])'

        clauses = re.split(delimiter_pattern, text)

        result = []
        i = 0
        while i < len(clauses):
            if i + 1 < len(clauses) and clauses[i + 1] in self.CLAUSE_DELIMITERS:
                clause = clauses[i] + clauses[i + 1]
                result.append(clause.strip())
                i += 2
            elif clauses[i].strip():
                result.append(clauses[i].strip())
                i += 1
            else:
                i += 1

        return [c for c in result if c.strip()]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        تطبيق التداخل بين الأجزاء - Apply overlap between chunks

        العربية:
            إضافة تداخل بين الأجزاء المتتالية للحفاظ على السياق

        English:
            Add overlap between consecutive chunks to preserve context.
        """
        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            # استخراج آخر جزء من الجزء السابق مع احترام حدود الكلمات
            # Take the tail of the previous chunk, respecting word boundaries
            # so Arabic words are never cut in half.
            overlap_text = self._tail_at_word_boundary(chunks[i - 1], self.config.chunk_overlap)

            # إضافة الجزء الحالي مع التداخل
            if overlap_text:
                overlapped_chunks.append(overlap_text + " " + chunks[i])
            else:
                overlapped_chunks.append(chunks[i])

        return overlapped_chunks

    @staticmethod
    def _tail_at_word_boundary(text: str, max_length: int) -> str:
        """
        اقتطاع نهاية النص عند حدود الكلمات - Tail of text cut at a word boundary

        العربية:
            إرجاع آخر جزء من النص بطول أقصاه max_length دون قطع أي كلمة.

        English:
            Return the trailing part of the text, at most max_length characters,
            without splitting any word.
        """
        if len(text) <= max_length:
            return text

        tail = text[-max_length:]
        if text[-max_length - 1] != ' ':
            # كنا في منتصف كلمة، تجاوزها - We landed mid-word; skip past it.
            space_index = tail.find(' ')
            if space_index == -1:
                return ""
            tail = tail[space_index + 1:]

        return tail.strip()

    def get_chunk_statistics(self, chunks: List[str]) -> dict:
        """
        الحصول على إحصائيات التقطيع - Get chunking statistics

        العربية:
            حساب إحصائيات عن الأجزاء المقطعة

        English:
            Calculate statistics about the chunked text.

        Returns:
            dict - إحصائيات عن الأجزاء
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        sizes = [len(chunk) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "average_chunk_size": sum(sizes) / len(chunks),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        }
