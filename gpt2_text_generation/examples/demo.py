"""
Examples demonstrating GPT-2 text generation capabilities.
"""

from gpt2_text_generation.model import GPT2Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_generation_example():
    """Example: Basic text generation."""
    print("\n" + "="*80)
    print("BASIC TEXT GENERATION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a magical forest,",
        "The secret to happiness lies in",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        output = generator.generate(prompt, max_length=100, temperature=0.8)[0]
        print(f"Generated: {output}\n")


def story_generation_example():
    """Example: Creative story generation."""
    print("\n" + "="*80)
    print("STORY GENERATION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    prompt = "In the year 2150, humanity discovered a signal from deep space."
    
    story = generator.generate_story(
        prompt,
        max_length=300,
        temperature=0.9
    )
    
    print(f"\nStory:\n{story}")


def dialogue_generation_example():
    """Example: Dialogue generation."""
    print("\n" + "="*80)
    print("DIALOGUE GENERATION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    context = """
    Sarah: What do you think about the new AI developments?
    Mike:
    """
    
    dialogue = generator.generate_dialogue(
        context,
        num_turns=3,
        max_turn_length=60,
        temperature=0.85
    )
    
    print(f"Context:{context}")
    print("\nGenerated dialogue:")
    for i, turn in enumerate(dialogue, 1):
        print(f"Turn {i}: {turn}")


def variations_example():
    """Example: Generate multiple variations."""
    print("\n" + "="*80)
    print("TEXT VARIATIONS")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    prompt = "The most important thing in life is"
    
    variations = generator.generate_variations(
        prompt,
        num_variations=5,
        max_length=80,
        temperature=1.2
    )
    
    print(f"\nPrompt: {prompt}\n")
    print("Variations:")
    for i, var in enumerate(variations, 1):
        print(f"\n{i}. {var}")


def controlled_generation_example():
    """Example: Generation with different parameters."""
    print("\n" + "="*80)
    print("CONTROLLED GENERATION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    prompt = "Artificial intelligence will transform society by"
    
    print(f"\nPrompt: {prompt}\n")
    
    # Conservative (focused)
    print("Conservative generation (temperature=0.7):")
    output = generator.generate(
        prompt,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )[0]
    print(output)
    
    # Creative (diverse)
    print("\n\nCreative generation (temperature=1.5):")
    output = generator.generate(
        prompt,
        max_length=100,
        temperature=1.5,
        top_k=50,
        top_p=0.95
    )[0]
    print(output)


def text_completion_example():
    """Example: Text completion."""
    print("\n" + "="*80)
    print("TEXT COMPLETION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    partials = [
        "To be or not to be,",
        "The best way to predict the future is",
        "In the beginning",
    ]
    
    for partial in partials:
        print(f"\nPartial text: {partial}")
        completed = generator.complete_text(
            partial,
            max_new_tokens=50,
            temperature=0.8
        )
        print(f"Completed: {completed}")


def constrained_generation_example():
    """Example: Generation with constraints."""
    print("\n" + "="*80)
    print("CONSTRAINED GENERATION")
    print("="*80)
    
    generator = GPT2Generator(model_name="gpt2")
    
    prompt = "Write a story about technology."
    
    # Must include specific words
    output = generator.generate_with_constraints(
        prompt,
        required_words=["innovation", "future"],
        max_length=150,
        temperature=0.9
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Required words: innovation, future\n")
    print(f"Output:\n{output}")
    
    # Check if required words are present
    output_lower = output.lower()
    print(f"\nContains 'innovation': {'innovation' in output_lower}")
    print(f"Contains 'future': {'future' in output_lower}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GPT-2 TEXT GENERATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate various GPT-2 capabilities.")
    print("Note: Using gpt2 base model. Use gpt2-medium or gpt2-large for better quality.")
    
    try:
        basic_generation_example()
        story_generation_example()
        dialogue_generation_example()
        variations_example()
        controlled_generation_example()
        text_completion_example()
        constrained_generation_example()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
