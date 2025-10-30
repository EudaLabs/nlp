"""Named Entity Recognition module with multiple backends."""
from typing import List, Optional, Union

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class Entity:
    """Represents a named entity."""
    
    def __init__(self, text: str, label: str, start: int, end: int):
        """
        Initialize entity.
        
        Args:
            text: Entity text
            label: Entity label/type
            start: Start character position
            end: End character position
        """
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end
    
    def __repr__(self):
        return f"Entity(text='{self.text}', label='{self.label_}', start={self.start_char}, end={self.end_char})"


class NERRecognizer:
    """Named Entity Recognition with multiple backend support."""
    
    def __init__(
        self,
        backend: str = "spacy",
        model: Optional[str] = None,
        device: str = None,
    ):
        """
        Initialize NER recognizer.
        
        Args:
            backend: Backend to use ('spacy' or 'bert')
            model: Model name/path (None for defaults)
            device: Device for computation ('cpu', 'cuda', or None for auto)
        """
        self.backend = backend.lower()
        
        if self.backend == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("SpaCy not installed. Install with: pip install spacy")
            
            model = model or "en_core_web_sm"
            try:
                self.nlp = spacy.load(model)
            except OSError:
                raise OSError(
                    f"Model '{model}' not found. Download with: "
                    f"python -m spacy download {model}"
                )
        
        elif self.backend == "bert":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers not installed. Install with: pip install transformers torch"
                )
            
            model = model or "dslim/bert-base-NER"
            
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            
            self.nlp = pipeline(
                "ner",
                model=model,
                aggregation_strategy="simple",
                device=device,
            )
        
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'spacy' or 'bert'")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
        
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        if self.backend == "spacy":
            return self._extract_spacy(text)
        elif self.backend == "bert":
            return self._extract_bert(text)
    
    def _extract_spacy(self, text: str) -> List[Entity]:
        """Extract entities using SpaCy."""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
        
        return entities
    
    def _extract_bert(self, text: str) -> List[Entity]:
        """Extract entities using BERT."""
        results = self.nlp(text)
        
        entities = []
        for result in results:
            entities.append(Entity(
                text=result["word"],
                label=result["entity_group"],
                start=result["start"],
                end=result["end"],
            ))
        
        return entities
    
    def batch_extract(self, texts: List[str]) -> List[List[Entity]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of entity lists
        """
        return [self.extract_entities(text) for text in texts]
    
    def get_entity_types(self) -> List[str]:
        """Get supported entity types."""
        if self.backend == "spacy":
            return list(self.nlp.pipe_labels.get("ner", []))
        else:
            # BERT NER common types
            return ["PER", "ORG", "LOC", "MISC"]


def extract_entities_simple(text: str, backend: str = "spacy") -> List[dict]:
    """
    Simple function to extract entities (convenience function).
    
    Args:
        text: Input text
        backend: Backend to use
    
    Returns:
        List of entity dictionaries
    """
    recognizer = NERRecognizer(backend=backend)
    entities = recognizer.extract_entities(text)
    
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in entities
    ]


if __name__ == "__main__":
    # Example usage
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    print("SpaCy NER:")
    try:
        recognizer = NERRecognizer(backend="spacy")
        entities = recognizer.extract_entities(text)
        
        for entity in entities:
            print(f"  {entity.text}: {entity.label_}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nBERT NER:")
    try:
        recognizer = NERRecognizer(backend="bert")
        entities = recognizer.extract_entities(text)
        
        for entity in entities:
            print(f"  {entity.text}: {entity.label_}")
    except Exception as e:
        print(f"  Error: {e}")
