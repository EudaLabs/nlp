"""Demo examples for Question Answering system."""
from question_answering import QASystem


def basic_qa_demo():
    """Demo basic question answering."""
    print("=" * 80)
    print("BASIC QUESTION ANSWERING DEMO")
    print("=" * 80)

    qa = QASystem()

    context = """
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability with its notable use
    of significant indentation. It supports multiple programming paradigms, including
    procedural, object-oriented, and functional programming.
    """

    questions = [
        "Who created Python?",
        "When was Python first released?",
        "What does Python emphasize?",
        "What programming paradigms does Python support?",
    ]

    for question in questions:
        answer = qa.answer(question, context)
        print(f"\nQ: {question}")
        print(f"A: {answer['answer']}")
        print(f"Confidence: {answer['score']:.2f}")


def multi_context_demo():
    """Demo QA with multiple contexts."""
    print("\n" + "=" * 80)
    print("MULTI-CONTEXT QA DEMO")
    print("=" * 80)

    qa = QASystem()

    contexts = [
        """
        The Eiffel Tower was designed by Gustave Eiffel's company and built between 1887
        and 1889. It stands 330 meters tall and is located in Paris, France.
        """,
        """
        The Great Wall of China was built over many centuries, primarily during the Ming
        Dynasty (1368-1644). It stretches over 13,000 miles across northern China.
        """,
    ]

    questions = [
        "Who designed the Eiffel Tower?",
        "When was the Great Wall built?",
    ]

    for i, (question, context) in enumerate(zip(questions, contexts), 1):
        answer = qa.answer(question, context)
        print(f"\n{i}. Question: {question}")
        print(f"   Answer: {answer['answer']} (score: {answer['score']:.2f})")


def confidence_filtering_demo():
    """Demo confidence-based filtering."""
    print("\n" + "=" * 80)
    print("CONFIDENCE FILTERING DEMO")
    print("=" * 80)

    qa = QASystem()

    context = "The sky is blue during the day."

    # High confidence question
    q1 = "What color is the sky?"
    a1 = qa.answer(q1, context, min_confidence=0.5)
    print(f"\nQ: {q1}")
    print(f"A: {a1['answer']} (score: {a1['score']:.2f})")

    # Low confidence question
    q2 = "What is the capital of France?"
    a2 = qa.answer(q2, context, min_confidence=0.5)
    print(f"\nQ: {q2}")
    print(f"A: {a2['answer']} (score: {a2['score']:.2f})")


def main():
    """Run all demos."""
    print("\n❓ Question Answering System Demos\n")

    try:
        basic_qa_demo()
        multi_context_demo()
        confidence_filtering_demo()

        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
