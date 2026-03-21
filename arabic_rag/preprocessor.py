"""
تطبيع النصوص العربية - Arabic Text Preprocessing Module

العربية:
    وحدة متخصصة في تطبيع النصوص العربية بما في ذلك إزالة التشكيل وتوحيد أشكال الألف
    ومعالجة التطويل والأحرف الخاصة. تحافظ على المعنى الأساسي مع تحسين جودة البحث.

English:
    Specialized module for normalizing Arabic text including diacritic removal,
    alef normalization, tatweel handling, and special character processing.
    Preserves meaning while improving search quality.
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """
    إعدادات التطبيع - Normalization configuration

    العربية:
        تكوين خيارات التطبيع المختلفة لمعالجة النصوص العربية

    English:
        Configuration options for various Arabic text normalization steps
    """
    remove_diacritics: bool = True
    normalize_alef: bool = True
    normalize_yaa: bool = True
    normalize_ha: bool = True
    remove_tatweel: bool = True
    remove_extra_spaces: bool = True
    remove_punctuation: bool = False
    lowercase: bool = False


class ArabicTextPreprocessor:
    """
    معالج النصوص العربية - Arabic Text Preprocessor

    العربية:
        فئة متخصصة في معالجة وتطبيع النصوص العربية. تتعامل مع التشكيل والألف
        والياء والهاء والتطويل والمسافات الزائدة.

    English:
        Specialized class for processing and normalizing Arabic text.
        Handles diacritics, alef variants, yaa, ha, tatweel, and extra spaces.

    Attributes:
        config: NormalizationConfig - التكوين المستخدم

    Example:
        ```python
        processor = ArabicTextPreprocessor()
        normalized = processor.normalize("اَلسَّلامُ عَلَيْكُمْ")
        # Output: "السلام عليكم"
        ```
    """

    # Arabic diacritics (تشكيل)
    DIACRITICS = {
        'َ': '',    # Fatha
        'ً': '',    # Fathatan
        'ُ': '',    # Damma
        'ٌ': '',    # Dammatan
        'ِ': '',    # Kasra
        'ٍ': '',    # Kasratan
        'ْ': '',    # Sukun
        'َّ': '',   # Shadda with Fatha
        'ّ': '',    # Shadda
        'ـ': '',    # Tatweel
    }

    # Alef normalization
    ALEF_VARIANTS = {
        'أ': 'ا',   # Alef with Hamza above
        'إ': 'ا',   # Alef with Hamza below
        'آ': 'ا',   # Alef with Madda
    }

    # Yaa normalization
    YAA_VARIANTS = {
        'ى': 'ي',   # Alef Maksura
        'ؤ': 'ء',   # Waw with Hamza
    }

    # Ha normalization
    HA_VARIANTS = {
        'ة': 'ه',   # Taa Marbuta
    }

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        تهيئة معالج النصوص - Initialize the preprocessor

        Args:
            config: NormalizationConfig - تكوين التطبيع (يستخدم الافتراضي إن لم يُحدد)
        """
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> str:
        """
        تطبيع النص الكامل - Normalize complete text

        العربية:
            تطبيع النص من خلال تطبيق جميع خطوات المعالجة بالترتيب المناسب.

        English:
            Normalize text by applying all processing steps in proper order.

        Args:
            text: str - النص المراد تطبيعه

        Returns:
            str - النص المطبّع

        Example:
            ```python
            text = "اَلسَّلامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ"
            normalized = processor.normalize(text)
            # Returns: "السلام عليكم ورحمة الله"
            ```
        """
        if not text:
            return text

        # خطوة 1: إزالة التشكيل
        if self.config.remove_diacritics:
            text = self._remove_diacritics(text)

        # خطوة 2: توحيد أشكال الألف
        if self.config.normalize_alef:
            text = self._normalize_alef(text)

        # خطوة 3: توحيد الياء
        if self.config.normalize_yaa:
            text = self._normalize_yaa(text)

        # خطوة 4: توحيد الهاء
        if self.config.normalize_ha:
            text = self._normalize_ha(text)

        # خطوة 5: إزالة التطويل
        if self.config.remove_tatweel:
            text = self._remove_tatweel(text)

        # خطوة 6: إزالة المسافات الزائدة
        if self.config.remove_extra_spaces:
            text = self._remove_extra_spaces(text)

        # خطوة 7: إزالة علامات الترقيم (اختياري)
        if self.config.remove_punctuation:
            text = self._remove_punctuation(text)

        # خطوة 8: تحويل إلى أحرف صغيرة (اختياري)
        if self.config.lowercase:
            text = text.lower()

        return text

    def _remove_diacritics(self, text: str) -> str:
        """
        إزالة التشكيل - Remove Arabic diacritics

        العربية:
            إزالة جميع علامات التشكيل (الفتحة والكسرة والضمة والسكون والشدة وغيرها)

        English:
            Remove all Arabic diacritical marks.
        """
        for diacritic, replacement in self.DIACRITICS.items():
            text = text.replace(diacritic, replacement)
        return text

    def _normalize_alef(self, text: str) -> str:
        """
        توحيد أشكال الألف - Normalize alef variants to standard alef

        العربية:
            توحيد الألف بهمزة أعلاها والألف بهمزة أسفلها والألف بالمد إلى الألف العادية

        English:
            Normalize alef with hamza above, alef with hamza below, and alef with madda.
        """
        for variant, standard in self.ALEF_VARIANTS.items():
            text = text.replace(variant, standard)
        return text

    def _normalize_yaa(self, text: str) -> str:
        """
        توحيد الياء - Normalize yaa variants

        العربية:
            توحيد الألف المقصورة والواو بهمزة إلى الأشكال القياسية

        English:
            Normalize alef maksura and waw with hamza variants.
        """
        for variant, standard in self.YAA_VARIANTS.items():
            text = text.replace(variant, standard)
        return text

    def _normalize_ha(self, text: str) -> str:
        """
        توحيد الهاء - Normalize ha variants

        العربية:
            توحيد التاء المربوطة إلى الهاء العادية

        English:
            Normalize taa marbuta to regular ha.
        """
        for variant, standard in self.HA_VARIANTS.items():
            text = text.replace(variant, standard)
        return text

    def _remove_tatweel(self, text: str) -> str:
        """
        إزالة التطويل - Remove tatweel character

        العربية:
            إزالة حرف التطويل (الخط الطويل) الذي يستخدم للتأكيد البصري

        English:
            Remove tatweel character used for visual emphasis.
        """
        return text.replace('ـ', '')

    def _remove_extra_spaces(self, text: str) -> str:
        """
        إزالة المسافات الزائدة - Remove extra spaces

        العربية:
            إزالة المسافات المتعددة وتحويلها إلى مسافة واحدة، وإزالة المسافات من البدايات والنهايات

        English:
            Remove multiple spaces and convert to single space, trim edges.
        """
        # إزالة المسافات المتعددة
        text = re.sub(r'\s+', ' ', text)
        # إزالة المسافات من البدايات والنهايات
        return text.strip()

    def _remove_punctuation(self, text: str) -> str:
        """
        إزالة علامات الترقيم - Remove punctuation marks

        العربية:
            إزالة العلامات الخاصة والترقيم

        English:
            Remove special characters and punctuation marks.
        """
        # الاحتفاظ بالأحرف العربية والأرقام والمسافات فقط
        arabic_chars = r'[\u0600-\u06FF\u0750-\u077F\s\d\-]'
        return re.sub(f'[^{arabic_chars}]', '', text)

    def normalize_query(self, query: str) -> str:
        """
        تطبيع الاستعلام - Normalize search query

        العربية:
            تطبيع استعلام البحث بنفس طريقة تطبيع المستندات
            لضمان تطابق أفضل في البحث

        English:
            Normalize search query the same way as documents for better matching.

        Args:
            query: str - الاستعلام

        Returns:
            str - الاستعلام المطبّع
        """
        return self.normalize(query)

    def extract_stem(self, word: str) -> str:
        """
        استخراج جذر الكلمة - Extract word stem (basic stemming)

        العربية:
            محاولة استخراج جذر الكلمة بإزالة البادئات واللواحق الشائعة

        English:
            Extract basic word stem by removing common prefixes and suffixes.

        Args:
            word: str - الكلمة

        Returns:
            str - جذر الكلمة المتوقع

        Example:
            ```python
            stem = processor.extract_stem("والمدرسة")
            # Potential output: "درس" or similar
            ```
        """
        word = self.normalize(word)

        # إزالة البادئات الشائعة
        prefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]

        # إزالة اللواحق الشائعة
        suffixes = ['ها', 'ان', 'ات', 'ون', 'ين', 'ة', 'ه']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break

        return word


def preprocess_documents(documents: list[str], config: Optional[NormalizationConfig] = None) -> list[str]:
    """
    معالجة قائمة من المستندات - Process a list of documents

    العربية:
        دالة مساعدة لتطبيع قائمة من المستندات دفعة واحدة

    English:
        Helper function to normalize a batch of documents.

    Args:
        documents: list[str] - قائمة المستندات
        config: NormalizationConfig - التكوين

    Returns:
        list[str] - المستندات المطبّعة
    """
    preprocessor = ArabicTextPreprocessor(config)
    return [preprocessor.normalize(doc) for doc in documents]
