"""
مثال نظام متعدد الوكلاء - Multi-Agent RAG Example

العربية:
    مثال متقدم يوضح استخدام فريق الوكلاء المتخصصة

English:
    Advanced example demonstrating multi-agent RAG system
"""

from arabic_rag.pipeline import ArabicRAGPipeline
from arabic_rag.agents.multi_agent_crew import setup_crew


def main():
    """
    الدالة الرئيسية - Main function

    العربية:
        مثال متقدم لاستخدام فريق الوكلاء

    English:
        Advanced example using multi-agent team
    """

    # تهيئة خط الأنابيب
    print("تهيئة نظام RAG متعدد الوكلاء...")
    pipeline = ArabicRAGPipeline(vector_store="memory", llm_provider="local", verbose=True)

    # إضافة المستندات
    documents = [
        "نظام الشركات السعودي ينص على أن رأس مال الشركة المساهمة لا يقل عن خمسة ملايين ريال سعودي. يجب أن يكون لدى الشركة مجلس إدارة يتكون من ثلاثة أعضاء على الأقل.",
        "للمساهمين الحق في حضور الجمعية العامة والتصويت على القرارات. يحق لهم الحصول على حصتهم من الأرباح حسب عدد أسهمهم.",
        "القانون التجاري السعودي يحدد الأطر القانونية لجميع العمليات التجارية. يجب على التجار الالتزام بقوانين الدولة والأنظمة المحلية.",
        "الشركة المساهمة هي شركة تجارية يتكون رأسمالها من أسهم متساوية القيمة. يمكن تداول الأسهم بين المساهمين.",
        "مجلس الإدارة مسؤول عن إدارة الشركة وتمثيلها أمام الغير. يتخذ المجلس القرارات المهمة المتعلقة بعمل الشركة.",
        "الجمعية العامة للمساهمين تجتمع مرة واحدة على الأقل في السنة. يتم فيها انتخاب أعضاء مجلس الإدارة والموافقة على الحسابات.",
        "يجب نشر القوائم المالية للشركة بشكل دوري. يتم فحص الحسابات من قبل مدقق خارجي مستقل.",
    ]

    print(f"إضافة {len(documents)} مستند...")
    pipeline.add_documents(documents)

    # إعداد فريق الوكلاء
    print("\nإعداد فريق الوكلاء...")
    crew = setup_crew(retriever=pipeline.retriever, verbose=True)

    # تنفيذ مهام باستخدام الفريق
    print("\n" + "=" * 70)
    print("تنفيذ مهام باستخدام فريق الوكلاء")
    print("=" * 70)

    tasks = [
        "شرح مفهوم الشركة المساهمة وخصائصها الرئيسية",
        "ما هي حقوق والتزامات المساهمين؟",
        "ما دور مجلس الإدارة في الشركة المساهمة؟",
    ]

    results = {}
    for i, task in enumerate(tasks, 1):
        print(f"\n[المهمة {i}] {task}")
        print("-" * 70)
        result = crew.execute_task(task, top_k=3)

        if result["success"]:
            print(f"\n✓ نجحت المهمة")
            print(f"\nالإجابة النهائية:\n{result['final_answer'][:500]}...")
            print(f"\nالمستندات المستخدمة: {len(result['source_documents'])}")
        else:
            print(f"\n✗ فشلت المهمة")

        results[task] = result

    # عرض ملخص التنفيذ
    print("\n" + "=" * 70)
    print("ملخص التنفيذ")
    print("=" * 70)

    summary = crew.get_execution_summary()
    print(f"\nعدد عمليات البحث: {len(summary['research_history'])}")
    print(f"عدد عمليات التحقق: {len(summary['validation_report'].get('issues', []))}")
    print(f"عدد الإجابات المكتوبة: {len(summary['writing_history'])}")


if __name__ == "__main__":
    main()
