"""
أدوات عربية - Arabic Utility Functions

العربية:
    مجموعة من الدوال المساعدة المتخصصة في معالجة النصوص العربية

English:
    Collection of helper functions specialized for Arabic text processing
"""

import re
from typing import Optional


def is_arabic_text(text: str) -> bool:
    """
    التحقق من وجود نص عربي - Check if text contains Arabic

    العربية:
        التحقق من وجود أحرف عربية في النص

    English:
        Check if text contains Arabic characters.

    Args:
        text: str - النص المراد التحقق منه

    Returns:
        bool - True إذا كان النص يحتوي على عربي

    Example:
        ```python
        assert is_arabic_text("مرحبا بك")
        assert not is_arabic_text("Hello")
        ```
    """
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F]'
    return bool(re.search(arabic_pattern, text))


def count_arabic_words(text: str) -> int:
    """
    عد الكلمات العربية - Count Arabic words

    العربية:
        عد عدد الكلمات العربية في النص

    English:
        Count the number of Arabic words in the text.

    Args:
        text: str - النص

    Returns:
        int - عدد الكلمات

    Example:
        ```python
        count = count_arabic_words("مرحبا بك في العالم")
        assert count == 4
        ```
    """
    # إزالة المسافات الزائدة
    text = text.strip()
    if not text:
        return 0

    # تقسيم حسب المسافات
    words = text.split()

    # عد الكلمات التي تحتوي على عربي
    count = sum(1 for word in words if is_arabic_text(word))
    return count


def normalize_spaces(text: str) -> str:
    """
    تطبيع المسافات - Normalize spaces

    العربية:
        إزالة المسافات الزائدة والأحرف غير المرئية

    English:
        Remove extra spaces and invisible characters.

    Args:
        text: str - النص

    Returns:
        str - النص المطبّع

    Example:
        ```python
        result = normalize_spaces("مرحبا    بك")
        assert result == "مرحبا بك"
        ```
    """
    # إزالة جميع أنواع المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    # إزالة المسافات من البدايات والنهايات
    return text.strip()


def reverse_text_direction(text: str) -> str:
    """
    عكس اتجاه النص - Reverse text direction

    العربية:
        عكس ترتيب الأحرف في النص (للاستخدام في السياقات البرمجية)

    English:
        Reverse character order in text (for programmatic contexts).

    Args:
        text: str - النص

    Returns:
        str - النص معكوس الاتجاه

    Example:
        ```python
        result = reverse_text_direction("مرحبا")
        assert result == "ابحرم"
        ```
    """
    return text[::-1]


def extract_numbers(text: str) -> list:
    """
    استخراج الأرقام - Extract numbers

    العربية:
        استخراج جميع الأرقام من النص

    English:
        Extract all numbers from text.

    Args:
        text: str - النص

    Returns:
        list - قائمة الأرقام المستخرجة

    Example:
        ```python
        numbers = extract_numbers("السعر 100 ريال و 50 هللة")
        assert numbers == [100, 50]
        ```
    """
    # البحث عن جميع الأرقام
    numbers = re.findall(r'\d+', text)
    return [int(num) for num in numbers]


def extract_urls(text: str) -> list:
    """
    استخراج الروابط - Extract URLs

    العربية:
        استخراج جميع الروابط من النص

    English:
        Extract all URLs from text.

    Args:
        text: str - النص

    Returns:
        list - قائمة الروابط

    Example:
        ```python
        urls = extract_urls("زر الموقع https://example.com هنا")
        assert "https://example.com" in urls
        ```
    """
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)


def extract_emails(text: str) -> list:
    """
    استخراج البريد الإلكتروني - Extract emails

    العربية:
        استخراج جميع عناوين البريد الإلكتروني من النص

    English:
        Extract all email addresses from text.

    Args:
        text: str - النص

    Returns:
        list - قائمة عناوين البريد

    Example:
        ```python
        emails = extract_emails("البريد: test@example.com")
        assert "test@example.com" in emails
        ```
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)


def split_into_paragraphs(text: str, min_length: int = 50) -> list:
    """
    تقسيم إلى فقرات - Split into paragraphs

    العربية:
        تقسيم النص إلى فقرات بناءً على الفواصل الفارغة

    English:
        Split text into paragraphs based on blank lines.

    Args:
        text: str - النص
        min_length: int - الطول الأدنى للفقرة

    Returns:
        list - قائمة الفقرات

    Example:
        ```python
        paragraphs = split_into_paragraphs("فقرة 1\\n\\nفقرة 2")
        assert len(paragraphs) == 2
        ```
    """
    # تقسيم بناءً على الأسطر الفارغة
    paragraphs = re.split(r'\n\s*\n', text)
    # تصفية الفقرات الفارغة والقصيرة جداً
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    return paragraphs


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    قطع النص - Truncate text

    العربية:
        قطع النص إذا تجاوز الطول المحدد

    English:
        Truncate text if it exceeds maximum length.

    Args:
        text: str - النص
        max_length: int - الطول الأقصى
        suffix: str - اللاحقة

    Returns:
        str - النص المقطوع

    Example:
        ```python
        result = truncate_text("نص طويل جداً", 8)
        assert result == "نص طو..."
        ```
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length - len(suffix)]
    return truncated + suffix


def get_text_statistics(text: str) -> dict:
    """
    الحصول على إحصائيات النص - Get text statistics

    العربية:
        الحصول على إحصائيات عن النص مثل عدد الأحرف والكلمات والفقرات

    English:
        Get statistics about text including character and word counts.

    Args:
        text: str - النص

    Returns:
        dict - قاموس الإحصائيات

    Example:
        ```python
        stats = get_text_statistics("مرحبا بك")
        assert stats['character_count'] == 8
        assert stats['word_count'] == 2
        ```
    """
    # إزالة المسافات الزائدة
    text = normalize_spaces(text)

    return {
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(" ", "")),
        "word_count": len(text.split()),
        "paragraph_count": len(split_into_paragraphs(text, min_length=1)),
        "arabic_word_count": count_arabic_words(text),
        "has_arabic": is_arabic_text(text),
        "has_numbers": bool(extract_numbers(text)),
        "has_urls": bool(extract_urls(text)),
        "has_emails": bool(extract_emails(text)),
    }
