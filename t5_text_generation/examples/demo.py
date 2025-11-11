"""
Demo script for T5 text generation.
"""
from t5_text_generation import T5Generator, T5Config


def summarization_demo():
    """Demonstrate text summarization."""
    print("=" * 80)
    print("TEXT SUMMARIZATION DEMO")
    print("=" * 80)

    generator = T5Generator(model_name="t5-small")

    texts = [
        """
        Artificial Intelligence (AI) is revolutionizing many industries. From healthcare
        to finance, AI systems are being deployed to solve complex problems. Machine learning,
        a subset of AI, enables computers to learn from data without explicit programming.
        Deep learning, using neural networks, has achieved remarkable results in image
        recognition, natural language processing, and game playing. However, ethical
        considerations and responsible AI development remain crucial challenges.
        """,
        """
        Climate change is one of the most pressing issues of our time. Rising global
        temperatures are causing ice caps to melt, sea levels to rise, and extreme weather
        events to become more frequent. Scientists agree that human activities, particularly
        the burning of fossil fuels, are the primary drivers of climate change. Urgent action
        is needed to reduce greenhouse gas emissions and transition to renewable energy sources.
        """,
    ]

    for i, text in enumerate(texts, 1):
        summary = generator.summarize(text.strip(), max_length=60, min_length=20)
        print(f"\nExample {i}:")
        print(f"Original ({len(text.split())} words):\n{text.strip()}")
        print(f"\nSummary ({len(summary.split())} words):\n{summary}")
        print("-" * 80)


def paraphrasing_demo():
    """Demonstrate text paraphrasing."""
    print("\n" + "=" * 80)
    print("TEXT PARAPHRASING DEMO")
    print("=" * 80)

    generator = T5Generator(model_name="t5-small")

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists have discovered a new species of butterfly.",
        "Reading books is a great way to expand your knowledge.",
        "Technology has transformed the way we communicate.",
    ]

    for sentence in sentences:
        paraphrase = generator.paraphrase(sentence)
        print(f"\nOriginal:    {sentence}")
        print(f"Paraphrased: {paraphrase}")


def translation_demo():
    """Demonstrate text translation."""
    print("\n" + "=" * 80)
    print("TEXT TRANSLATION DEMO")
    print("=" * 80)

    generator = T5Generator(model_name="t5-small")

    translations = [
        ("Hello, how are you?", "English", "German"),
        ("Good morning, everyone!", "English", "French"),
        ("Thank you very much.", "English", "Spanish"),
    ]

    for text, source, target in translations:
        translated = generator.translate(text, source, target)
        print(f"\n{source}: {text}")
        print(f"{target}: {translated}")


def batch_processing_demo():
    """Demonstrate batch processing."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING DEMO")
    print("=" * 80)

    generator = T5Generator(model_name="t5-small")

    texts = [
        "This is the first text to summarize.",
        "This is the second text that needs summarization.",
        "Here is another piece of text for our batch demo.",
        "And finally, the last text in our batch.",
    ]

    print(f"\nProcessing {len(texts)} texts in batch...")
    summaries = generator.batch_generate(
        texts, batch_size=2, max_length=20, min_length=5
    )

    for i, (text, summary) in enumerate(zip(texts, summaries), 1):
        print(f"\n{i}. Text: {text}")
        print(f"   Summary: {summary}")


def qa_demo():
    """Demonstrate question answering."""
    print("\n" + "=" * 80)
    print("QUESTION ANSWERING DEMO")
    print("=" * 80)

    generator = T5Generator(model_name="t5-small")

    qa_pairs = [
        {
            "question": "What is machine learning?",
            "context": "Machine learning is a subset of artificial intelligence that "
            "enables computers to learn from data without being explicitly programmed. "
            "It uses algorithms to identify patterns and make decisions.",
        },
        {
            "question": "Who invented the telephone?",
            "context": "Alexander Graham Bell is credited with inventing the telephone "
            "in 1876. He was a Scottish-born scientist and inventor who made significant "
            "contributions to telecommunications.",
        },
    ]

    for i, qa in enumerate(qa_pairs, 1):
        answer = generator.answer_question(qa["question"], qa["context"])
        print(f"\nExample {i}:")
        print(f"Question: {qa['question']}")
        print(f"Context: {qa['context']}")
        print(f"Answer: {answer}")


def main():
    """Run all demos."""
    print("\n🚀 T5 Text Generation Demos\n")

    try:
        summarization_demo()
        paraphrasing_demo()
        translation_demo()
        qa_demo()
        batch_processing_demo()

        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
