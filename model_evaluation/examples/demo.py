"""Demo examples for model evaluation framework."""
from model_evaluation import (
    ClassificationEvaluator,
    GenerationEvaluator,
    QAEvaluator,
    NERvaluator,
    plot_confusion_matrix,
)


def classification_demo():
    """Demo classification evaluation."""
    print("=" * 80)
    print("CLASSIFICATION EVALUATION DEMO")
    print("=" * 80)

    evaluator = ClassificationEvaluator()

    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1, 0, 1, 2]

    metrics = evaluator.evaluate(y_true, y_pred)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")


def generation_demo():
    """Demo text generation evaluation."""
    print("\n" + "=" * 80)
    print("TEXT GENERATION EVALUATION DEMO")
    print("=" * 80)

    evaluator = GenerationEvaluator()

    predictions = [
        "The cat sat on the mat",
        "Machine learning is fascinating",
    ]
    references = [
        "The cat is sitting on the mat",
        "Machine learning is very interesting",
    ]

    metrics = evaluator.evaluate(predictions, references, metrics=["bleu", "rouge"])

    print(f"\nBLEU Score: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")


def qa_demo():
    """Demo QA evaluation."""
    print("\n" + "=" * 80)
    print("QUESTION ANSWERING EVALUATION DEMO")
    print("=" * 80)

    evaluator = QAEvaluator()

    predictions = [
        {"id": "1", "prediction_text": "Paris"},
        {"id": "2", "prediction_text": "1991"},
    ]
    references = [
        {"id": "1", "answers": {"text": ["Paris", "paris"], "answer_start": [0, 0]}},
        {"id": "2", "answers": {"text": ["1991"], "answer_start": [0]}},
    ]

    metrics = evaluator.evaluate(predictions, references)

    print(f"\nExact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}")


def ner_demo():
    """Demo NER evaluation."""
    print("\n" + "=" * 80)
    print("NER EVALUATION DEMO")
    print("=" * 80)

    evaluator = NERvaluator()

    predictions = [
        ["B-PER", "I-PER", "O", "B-LOC", "O"],
        ["O", "B-ORG", "I-ORG", "O", "O"],
    ]
    references = [
        ["B-PER", "I-PER", "O", "B-LOC", "O"],
        ["O", "B-ORG", "O", "O", "B-DATE"],
    ]

    metrics = evaluator.evaluate(predictions, references)

    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


def main():
    """Run all demos."""
    print("\n🎯 Model Evaluation Framework Demos\n")

    try:
        classification_demo()
        generation_demo()
        qa_demo()
        ner_demo()

        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
