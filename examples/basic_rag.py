"""
مثال بسيط على نظام RAG - Basic RAG Example

العربية:
    مثال بسيط يوضح كيفية استخدام أدوات RAG العربية

English:
    Simple example demonstrating how to use Arabic RAG toolkit
"""

from arabic_rag.pipeline import ArabicRAGPipeline


def main():
    """
    الدالة الرئيسية - Main function

    العربية:
        مثال عملي لاستخدام خط أنابيب RAG العربي

    English:
        Practical example of using Arabic RAG pipeline
    """

    # تهيئة خط الأنابيب
    print("تهيئة خط أنابيب RAG...")
    pipeline = ArabicRAGPipeline()

    # إضافة مستندات
    documents = [
        "نظام الشركات السعودي ينص على أن رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي. يجب أن يكون لدى الشركة مجلس إدارة يتكون من ثلاثة أعضاء على الأقل.",
        "للمساهمين الحق في حضور الجمعية العامة والتصويت على القرارات. يحق لهم الحصول على حصتهم من الأرباح حسب عدد أسهمهم.",
        "القانون التجاري السعودي يحدد الأطر القانونية لجميع العمليات التجارية. يجب على التجار الالتزام بقوانين الدولة والأنظمة المحلية.",
        "الشركة المساهمة هي شركة تجارية يتكون رأسمالها من أسهم متساوية القيمة. يمكن تداول الأسهم بين المساهمين.",
        "مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير. يتخذ المجلس القرارات المهمة المتعلقة بعمل الشركة.",
    ]

    print(f"إضافة {len(documents)} مستندات...")
    pipeline.add_documents(documents)

    # الاستعلام الأول
    print("\n" + "=" * 50)
    question1 = "ما هو الحد الأدنى لرأس مال الشركة المساهمة؟"
    print(f"السؤال: {question1}")
    print("-" * 50)
    answer1 = pipeline.query(question1)
    print(f"الإجابة:\n{answer1}")

    # الاستعلام الثاني
    print("\n" + "=" * 50)
    question2 = "ما هي مسؤوليات مجلس الإدارة؟"
    print(f"السؤال: {question2}")
    print("-" * 50)
    answer2 = pipeline.query(question2)
    print(f"الإجابة:\n{answer2}")

    # الاستعلام الثالث
    print("\n" + "=" * 50)
    question3 = "حقوق المساهمين في الشركة المساهمة"
    print(f"السؤال: {question3}")
    print("-" * 50)
    answer3 = pipeline.query(question3)
    print(f"الإجابة:\n{answer3}")

    # الحصول على الإحصائيات
    print("\n" + "=" * 50)
    print("إحصائيات خط الأنابيب:")
    print("-" * 50)
    stats = pipeline.get_pipeline_stats()
    print(f"عدد المستندات: {stats['total_documents']}")
    print(f"بُعد التضمين: {stats['embedding_dimension']}")
    print(f"نوع قاعدة البيانات المتجهة: {stats['vector_store_type']}")


if __name__ == "__main__":
    main()
