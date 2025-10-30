"""Example usage of the NER system."""
from recognizer import NERRecognizer
from visualizer import print_entity_summary, visualize_entities


def basic_example():
    """Basic NER example."""
    print("=" * 70)
    print("Basic NER Example")
    print("=" * 70)
    
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
    in April 1976 in Cupertino, California. The company's first product was 
    the Apple I, which sold for $666.66. Today, Apple is worth over $2 trillion 
    and is headquartered at Apple Park.
    """
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        entities = recognizer.extract_entities(text)
        
        print(f"\nText: {text.strip()}\n")
        print(f"Found {len(entities)} entities:")
        print("-" * 70)
        
        for ent in entities:
            print(f"{ent.text:30} -> {ent.label_}")
        
        print("\n")
        print_entity_summary(entities)
    
    except Exception as e:
        print(f"Error: {e}")


def multi_backend_example():
    """Compare different NER backends."""
    print("\n" + "=" * 70)
    print("Multi-Backend Comparison")
    print("=" * 70)
    
    text = "Google CEO Sundar Pichai announced new AI products in Mountain View."
    
    backends = ["spacy", "bert"]
    
    for backend in backends:
        try:
            print(f"\n{backend.upper()} Results:")
            print("-" * 70)
            
            recognizer = NERRecognizer(backend=backend)
            entities = recognizer.extract_entities(text)
            
            for ent in entities:
                print(f"  {ent.text:25} -> {ent.label_}")
        
        except Exception as e:
            print(f"  Error with {backend}: {e}")


def batch_processing_example():
    """Batch processing example."""
    print("\n" + "=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    texts = [
        "Amazon was founded by Jeff Bezos in Seattle.",
        "Microsoft is based in Redmond, Washington.",
        "Facebook was created by Mark Zuckerberg at Harvard University.",
    ]
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        results = recognizer.batch_extract(texts)
        
        for i, (text, entities) in enumerate(zip(texts, results), 1):
            print(f"\n{i}. {text}")
            for ent in entities:
                print(f"   - {ent.text} ({ent.label_})")
    
    except Exception as e:
        print(f"Error: {e}")


def domain_specific_example():
    """Domain-specific NER example."""
    print("\n" + "=" * 70)
    print("News Article Processing")
    print("=" * 70)
    
    news = """
    Tesla Inc. reported record quarterly earnings on Wednesday, with CEO 
    Elon Musk stating that the company delivered 308,600 vehicles in Q4 2021. 
    The electric vehicle manufacturer's stock rose 13.5% following the 
    announcement, adding $100 billion to its market capitalization. Tesla's 
    Gigafactory in Austin, Texas is expected to begin production in 2022.
    """
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        entities = recognizer.extract_entities(news)
        
        # Categorize entities
        from collections import defaultdict
        
        categorized = defaultdict(list)
        for ent in entities:
            if ent.text not in categorized[ent.label_]:
                categorized[ent.label_].append(ent.text)
        
        print(f"\nArticle: {news.strip()[:100]}...\n")
        
        print("Extracted Information:")
        print("-" * 70)
        
        if "ORG" in categorized:
            print(f"Companies: {', '.join(categorized['ORG'])}")
        
        if "PERSON" in categorized:
            print(f"People: {', '.join(categorized['PERSON'])}")
        
        if "GPE" in categorized:
            print(f"Locations: {', '.join(categorized['GPE'])}")
        
        if "DATE" in categorized:
            print(f"Dates: {', '.join(categorized['DATE'])}")
        
        if "MONEY" in categorized:
            print(f"Money: {', '.join(categorized['MONEY'])}")
        
        if "PERCENT" in categorized:
            print(f"Percentages: {', '.join(categorized['PERCENT'])}")
        
        if "CARDINAL" in categorized:
            print(f"Numbers: {', '.join(categorized['CARDINAL'])}")
    
    except Exception as e:
        print(f"Error: {e}")


def visualization_example():
    """Visualization example."""
    print("\n" + "=" * 70)
    print("Visualization Example")
    print("=" * 70)
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        entities = recognizer.extract_entities(text)
        
        # Generate visualization
        html_file = "ner_example_viz.html"
        visualize_entities(text, entities, output_file=html_file)
        
        print(f"\nVisualization saved to {html_file}")
        print("Open this file in a web browser to see the highlighted entities.")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    multi_backend_example()
    batch_processing_example()
    domain_specific_example()
    visualization_example()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
