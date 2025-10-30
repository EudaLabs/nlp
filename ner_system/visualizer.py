"""Visualization tools for NER results."""
from typing import List, Optional

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def visualize_entities(
    text: str,
    entities: List,
    style: str = "ent",
    jupyter: bool = False,
    output_file: Optional[str] = None,
) -> str:
    """
    Visualize named entities in text.
    
    Args:
        text: Original text
        entities: List of Entity objects or dictionaries
        style: Visualization style ('ent' or 'dep')
        jupyter: Whether rendering in Jupyter notebook
        output_file: Optional path to save HTML
    
    Returns:
        HTML string of visualization
    """
    if not SPACY_AVAILABLE:
        raise ImportError("SpaCy required for visualization. Install with: pip install spacy")
    
    # Convert entities to displaCy format
    ents_data = []
    for ent in entities:
        if hasattr(ent, 'start_char'):
            ents_data.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            })
        else:
            # Assume dictionary format
            ents_data.append({
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
                "label": ent.get("label", ""),
            })
    
    # Create displaCy format
    doc_data = {
        "text": text,
        "ents": ents_data,
        "title": None,
    }
    
    # Generate HTML
    html = displacy.render(
        doc_data,
        style=style,
        manual=True,
        jupyter=jupyter,
    )
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    
    return html


def visualize_multiple(
    texts: List[str],
    entities_list: List[List],
    output_file: Optional[str] = None,
) -> str:
    """
    Visualize entities for multiple texts.
    
    Args:
        texts: List of texts
        entities_list: List of entity lists
        output_file: Optional path to save HTML
    
    Returns:
        HTML string of visualizations
    """
    if not SPACY_AVAILABLE:
        raise ImportError("SpaCy required. Install with: pip install spacy")
    
    docs_data = []
    for text, entities in zip(texts, entities_list):
        ents_data = []
        for ent in entities:
            if hasattr(ent, 'start_char'):
                ents_data.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                })
            else:
                ents_data.append({
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "label": ent.get("label", ""),
                })
        
        docs_data.append({
            "text": text,
            "ents": ents_data,
        })
    
    html = displacy.render(docs_data, style="ent", manual=True)
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    
    return html


def create_entity_summary(entities: List) -> dict:
    """
    Create a summary of entities by type.
    
    Args:
        entities: List of Entity objects
    
    Returns:
        Dictionary mapping entity types to lists of entity texts
    """
    from collections import defaultdict
    
    summary = defaultdict(list)
    
    for ent in entities:
        if hasattr(ent, 'label_'):
            label = ent.label_
            text = ent.text
        else:
            label = ent.get("label", "UNKNOWN")
            text = ent.get("text", "")
        
        if text and text not in summary[label]:
            summary[label].append(text)
    
    return dict(summary)


def print_entity_summary(entities: List):
    """
    Print a formatted summary of entities.
    
    Args:
        entities: List of Entity objects
    """
    summary = create_entity_summary(entities)
    
    print("Entity Summary:")
    print("=" * 50)
    
    for label, texts in sorted(summary.items()):
        print(f"\n{label}:")
        for text in texts:
            print(f"  - {text}")


if __name__ == "__main__":
    # Example usage
    from recognizer import NERRecognizer
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        entities = recognizer.extract_entities(text)
        
        print("Entities found:")
        for ent in entities:
            print(f"  {ent.text}: {ent.label_}")
        
        print("\n" + "=" * 50)
        print_entity_summary(entities)
        
        # Generate visualization
        html = visualize_entities(text, entities, output_file="entities_viz.html")
        print("\nVisualization saved to entities_viz.html")
    
    except Exception as e:
        print(f"Error: {e}")
