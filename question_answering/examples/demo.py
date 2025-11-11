"""
Examples demonstrating question answering capabilities.
"""

from question_answering.model import ExtractiveQA, GenerativeQA, HybridQA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extractive_qa_example():
    """Example: Extractive question answering."""
    print("\n" + "="*80)
    print("EXTRACTIVE QUESTION ANSWERING")
    print("="*80)
    
    qa = ExtractiveQA(model_name="deepset/roberta-base-squad2")
    
    context = """
    The Amazon rainforest, also called the Amazon jungle, is a moist broadleaf tropical 
    rainforest in the Amazon biome that covers most of the Amazon basin of South America. 
    This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. 
    The majority of the forest is contained within Brazil, with 60% of the rainforest, 
    followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, 
    Ecuador, Bolivia, Guyana, Suriname, and French Guiana.
    """
    
    questions = [
        "Where is the Amazon rainforest located?",
        "What percentage of the forest is in Brazil?",
        "How large is the Amazon basin?",
    ]
    
    print(f"Context: {context[:200]}...\n")
    
    for question in questions:
        answers = qa.answer(question, context, top_k=1)
        print(f"Q: {question}")
        print(f"A: {answers[0]['answer']}")
        print(f"Confidence: {answers[0]['score']:.4f}\n")


def generative_qa_example():
    """Example: Generative question answering."""
    print("\n" + "="*80)
    print("GENERATIVE QUESTION ANSWERING")
    print("="*80)
    
    qa = GenerativeQA(model_name="google/flan-t5-base")
    
    context = """
    Python is a high-level, interpreted programming language created by Guido van Rossum 
    and first released in 1991. Python's design philosophy emphasizes code readability 
    with the use of significant indentation. It supports multiple programming paradigms, 
    including structured, object-oriented, and functional programming.
    """
    
    questions = [
        "Who created Python?",
        "When was Python first released?",
        "What programming paradigms does Python support?",
    ]
    
    print(f"Context: {context[:200]}...\n")
    
    for question in questions:
        answer = qa.answer(question, context)
        print(f"Q: {question}")
        print(f"A: {answer}\n")


def hybrid_qa_example():
    """Example: Hybrid question answering."""
    print("\n" + "="*80)
    print("HYBRID QUESTION ANSWERING")
    print("="*80)
    
    qa = HybridQA()
    
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading 
    artists and intellectuals for its design, but it has become a global cultural icon of France 
    and one of the most recognizable structures in the world.
    """
    
    questions = [
        "Who was the Eiffel Tower named after?",
        "When was it constructed?",
        "Why is the Eiffel Tower famous?",
    ]
    
    print(f"Context: {context[:200]}...\n")
    
    for question in questions:
        result = qa.answer(question, context, mode="auto")
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Method: {result['method']}")
        if 'score' in result:
            print(f"Confidence: {result['score']:.4f}")
        print()


def batch_processing_example():
    """Example: Batch processing."""
    print("\n" + "="*80)
    print("BATCH PROCESSING")
    print("="*80)
    
    qa = ExtractiveQA()
    
    questions = [
        "What is AI?",
        "What is machine learning?",
        "What is deep learning?",
    ]
    
    contexts = [
        "Artificial Intelligence (AI) is intelligence demonstrated by machines.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
    ]
    
    answers = qa.batch_answer(questions, contexts)
    
    for q, a in zip(questions, answers):
        print(f"Q: {q}")
        print(f"A: {a[0]['answer']}")
        print(f"Score: {a[0]['score']:.4f}\n")


def confidence_filtering_example():
    """Example: Filtering by confidence."""
    print("\n" + "="*80)
    print("CONFIDENCE FILTERING")
    print("="*80)
    
    qa = ExtractiveQA()
    
    context = "The sky appears blue due to Rayleigh scattering of light."
    
    questions = [
        "Why is the sky blue?",  # Should have high confidence
        "What is the temperature of the sky?",  # Should have low confidence
    ]
    
    for question in questions:
        answer = qa.get_answer_with_confidence(
            question,
            context,
            confidence_threshold=0.5
        )
        
        print(f"Q: {question}")
        if answer:
            print(f"A: {answer['answer']}")
            print(f"Confidence: {answer['score']:.4f}")
        else:
            print("A: [No confident answer found]")
        print()


def open_domain_qa_example():
    """Example: Open-domain QA (without context)."""
    print("\n" + "="*80)
    print("OPEN-DOMAIN QUESTION ANSWERING")
    print("="*80)
    
    qa = GenerativeQA(model_name="google/flan-t5-base")
    
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?",
        "What is the speed of light?",
    ]
    
    print("Answering questions without provided context:\n")
    
    for question in questions:
        answer = qa.answer(question)  # No context
        print(f"Q: {question}")
        print(f"A: {answer}\n")


def multi_context_qa_example():
    """Example: QA across multiple contexts."""
    print("\n" + "="*80)
    print("MULTI-CONTEXT QUESTION ANSWERING")
    print("="*80)
    
    qa = ExtractiveQA()
    
    question = "What is the main benefit?"
    
    contexts = [
        "Solar energy provides clean and renewable power.",
        "Wind energy reduces carbon emissions significantly.",
        "Nuclear energy offers high energy density.",
    ]
    
    print(f"Question: {question}\n")
    
    results = []
    for i, context in enumerate(contexts, 1):
        answer = qa.answer(question, context, top_k=1)[0]
        results.append((context[:50] + "...", answer['answer'], answer['score']))
    
    # Sort by confidence
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("Answers from different contexts (sorted by confidence):\n")
    for context, answer, score in results:
        print(f"Context: {context}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.4f}\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("QUESTION ANSWERING EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate various QA capabilities.")
    
    try:
        extractive_qa_example()
        generative_qa_example()
        hybrid_qa_example()
        batch_processing_example()
        confidence_filtering_example()
        open_domain_qa_example()
        multi_context_qa_example()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
