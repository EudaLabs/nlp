"""Demo examples for GPT-2 text generation."""
from gpt2_finetuning import GPT2Generator, GenerationConfig


def basic_generation_demo():
    """Demo basic text generation."""
    print("=" * 80)
    print("BASIC TEXT GENERATION DEMO")
    print("=" * 80)

    generator = GPT2Generator(model_name="gpt2")

    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
    ]

    for prompt in prompts:
        text = generator.generate(prompt, max_length=80, temperature=0.8)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}")
        print("-" * 80)


def creative_writing_demo():
    """Demo creative writing with different temperatures."""
    print("\n" + "=" * 80)
    print("CREATIVE WRITING DEMO")
    print("=" * 80)

    generator = GPT2Generator(model_name="gpt2")
    prompt = "Once upon a time in a magical forest"

    # Low temperature (focused)
    focused = generator.generate_focused(prompt, max_length=100)
    print(f"\n🎯 Focused (low temp):\n{focused}\n")

    # Medium temperature (balanced)
    balanced = generator.generate_balanced(prompt, max_length=100)
    print(f"⚖️ Balanced (medium temp):\n{balanced}\n")

    # High temperature (creative)
    creative = generator.generate_creative(prompt, max_length=100)
    print(f"🎨 Creative (high temp):\n{creative}\n")


def completion_demo():
    """Demo text completion."""
    print("=" * 80)
    print("TEXT COMPLETION DEMO")
    print("=" * 80)

    generator = GPT2Generator(model_name="gpt2")

    partial_texts = [
        "The meaning of life is",
        "Machine learning helps us",
        "The best way to learn programming is",
    ]

    for partial in partial_texts:
        completion = generator.complete(partial, max_length=40)
        print(f"\nPartial: {partial}")
        print(f"Completed: {completion}")


def batch_generation_demo():
    """Demo batch text generation."""
    print("\n" + "=" * 80)
    print("BATCH GENERATION DEMO")
    print("=" * 80)

    generator = GPT2Generator(model_name="gpt2")

    prompts = [
        "Write a story about a brave knight",
        "Explain quantum computing in simple terms",
        "List three benefits of reading books",
    ]

    print(f"\nGenerating {len(prompts)} texts in batch...")
    results = generator.batch_generate(prompts, max_length=60, batch_size=2)

    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Result: {result}")


def main():
    """Run all demos."""
    print("\n🤖 GPT-2 Text Generation Demos\n")

    try:
        basic_generation_demo()
        creative_writing_demo()
        completion_demo()
        batch_generation_demo()

        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
