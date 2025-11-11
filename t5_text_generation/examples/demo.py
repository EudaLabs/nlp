"""
Examples demonstrating T5 text generation capabilities.
"""

from t5_text_generation.model import T5TextGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarization_example():
    """Example: Text summarization."""
    print("\n" + "="*80)
    print("TEXT SUMMARIZATION EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    "cognitive" functions that humans associate with the human mind, such as 
    "learning" and "problem solving".
    """
    
    summary = model.summarize(text, max_length=60, min_length=20)
    
    print(f"\nOriginal text ({len(text.split())} words):")
    print(text.strip())
    print(f"\nSummary ({len(summary.split())} words):")
    print(summary)


def translation_example():
    """Example: Translation."""
    print("\n" + "="*80)
    print("TRANSLATION EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    texts = [
        "Hello, how are you today?",
        "The weather is beautiful.",
        "I love learning new things.",
    ]
    
    for text in texts:
        translation = model.translate(
            text,
            source_lang="English",
            target_lang="German"
        )
        print(f"\nEnglish: {text}")
        print(f"German:  {translation}")


def paraphrase_example():
    """Example: Paraphrasing."""
    print("\n" + "="*80)
    print("PARAPHRASING EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    text = "Natural language processing is a fascinating field of study."
    
    paraphrases = model.paraphrase(text, num_return_sequences=3)
    
    print(f"\nOriginal: {text}")
    print("\nParaphrases:")
    for i, paraphrase in enumerate(paraphrases, 1):
        print(f"{i}. {paraphrase}")


def question_generation_example():
    """Example: Question generation."""
    print("\n" + "="*80)
    print("QUESTION GENERATION EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    answer = "Paris"
    
    question = model.generate_question(context, answer)
    
    print(f"\nContext: {context}")
    print(f"Answer:  {answer}")
    print(f"Generated Question: {question}")


def grammar_correction_example():
    """Example: Grammar correction."""
    print("\n" + "="*80)
    print("GRAMMAR CORRECTION EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    incorrect_texts = [
        "She don't like apples.",
        "I has a car.",
        "They was going to the store.",
    ]
    
    for text in incorrect_texts:
        corrected = model.correct_grammar(text)
        print(f"\nOriginal:  {text}")
        print(f"Corrected: {corrected}")


def batch_processing_example():
    """Example: Batch processing."""
    print("\n" + "="*80)
    print("BATCH PROCESSING EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]
    
    summaries = model.batch_generate(
        texts,
        task_prefix="summarize: ",
        max_length=30,
        batch_size=2
    )
    
    for i, (text, summary) in enumerate(zip(texts, summaries), 1):
        print(f"\n{i}. Original: {text}")
        print(f"   Summary:  {summary}")


def custom_generation_example():
    """Example: Custom text generation with parameters."""
    print("\n" + "="*80)
    print("CUSTOM GENERATION EXAMPLE")
    print("="*80)
    
    model = T5TextGenerator(model_name="t5-small")
    
    text = "Write a creative story about a robot."
    
    # Generate with different parameters
    print("\nWith conservative parameters (low temperature):")
    output1 = model.generate(
        text,
        max_length=100,
        temperature=0.7,
        num_beams=4
    )
    print(output1)
    
    print("\nWith creative parameters (high temperature):")
    output2 = model.generate(
        text,
        max_length=100,
        temperature=1.5,
        top_k=50,
        top_p=0.95
    )
    print(output2)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("T5 TEXT GENERATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate various T5 capabilities.")
    print("Note: Using t5-small for faster inference. Use t5-base or t5-large for better quality.")
    
    try:
        summarization_example()
        translation_example()
        paraphrase_example()
        question_generation_example()
        grammar_correction_example()
        batch_processing_example()
        custom_generation_example()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
