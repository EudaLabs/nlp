"""Command-line interface for NER system."""
import argparse
import json
import sys

from .recognizer import NERRecognizer
from .visualizer import print_entity_summary, visualize_entities


def extract_command(args):
    """Extract entities from text."""
    recognizer = NERRecognizer(backend=args.backend, model=args.model)
    entities = recognizer.extract_entities(args.text)
    
    if args.format == "json":
        output = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in entities
        ]
        print(json.dumps(output, indent=2))
    else:
        print(f"\nFound {len(entities)} entities:")
        print("-" * 50)
        for ent in entities:
            print(f"{ent.text:30} {ent.label_:15} ({ent.start_char}-{ent.end_char})")
        
        print("\n" + "=" * 50)
        print_entity_summary(entities)


def extract_file_command(args):
    """Extract entities from file."""
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    
    recognizer = NERRecognizer(backend=args.backend, model=args.model)
    entities = recognizer.extract_entities(text)
    
    output = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in entities
    ]
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


def visualize_command(args):
    """Visualize entities."""
    recognizer = NERRecognizer(backend=args.backend, model=args.model)
    entities = recognizer.extract_entities(args.text)
    
    html = visualize_entities(
        args.text,
        entities,
        output_file=args.output or "entities.html",
    )
    
    output_file = args.output or "entities.html"
    print(f"Visualization saved to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Named Entity Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract entities from text")
    extract_parser.add_argument("text", help="Text to process")
    extract_parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="NER backend to use",
    )
    extract_parser.add_argument("--model", help="Model name/path")
    extract_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    
    # Extract from file command
    file_parser = subparsers.add_parser("extract-file", help="Extract entities from file")
    file_parser.add_argument("input", help="Input file path")
    file_parser.add_argument("--output", help="Output JSON file path")
    file_parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="NER backend to use",
    )
    file_parser.add_argument("--model", help="Model name/path")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize entities")
    viz_parser.add_argument("text", help="Text to visualize")
    viz_parser.add_argument("--output", help="Output HTML file path")
    viz_parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="NER backend to use",
    )
    viz_parser.add_argument("--model", help="Model name/path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "extract":
            extract_command(args)
        elif args.command == "extract-file":
            extract_file_command(args)
        elif args.command == "visualize":
            visualize_command(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
