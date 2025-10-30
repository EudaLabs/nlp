"""Training script for BERT text classification."""
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import DataConfig, ModelConfig, TrainingConfig
from .utils import (
    compute_metrics,
    format_time,
    get_device,
    logger,
    print_training_summary,
    save_metrics,
    set_seed,
)


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def prepare_imdb_dataset(tokenizer, max_length: int = 512):
    """Load and prepare IMDB dataset."""
    logger.info("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Calculate accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": total_loss / (step + 1),
            "acc": correct_predictions / total_samples,
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


def train_classifier(
    dataset_name: Optional[str] = "imdb",
    texts: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    data_config: Optional[DataConfig] = None,
    **kwargs,
):
    """
    Train a BERT classifier.
    
    Args:
        dataset_name: Name of dataset to use (e.g., "imdb")
        texts: Custom training texts
        labels: Custom training labels
        val_texts: Custom validation texts
        val_labels: Custom validation labels
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        **kwargs: Additional arguments to override configs
    """
    # Initialize configs
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    data_config = data_config or DataConfig()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
    
    # Set seed
    set_seed(training_config.seed)
    
    # Get device
    device = get_device()
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=model_config.num_classes,
    )
    model.to(device)
    
    # Prepare datasets
    if dataset_name == "imdb":
        train_dataset, val_dataset = prepare_imdb_dataset(tokenizer, model_config.max_length)
    elif texts is not None and labels is not None:
        train_dataset = TextClassificationDataset(
            texts, labels, tokenizer, model_config.max_length
        )
        if val_texts is not None and val_labels is not None:
            val_dataset = TextClassificationDataset(
                val_texts, val_labels, tokenizer, model_config.max_length
            )
        else:
            val_dataset = None
    else:
        raise ValueError("Either dataset_name or texts/labels must be provided")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None
    
    # Print training summary
    print_training_summary(
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset) if val_dataset else 0,
        num_classes=model_config.num_classes,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    total_steps = len(train_loader) * training_config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    best_val_loss = float("inf")
    training_stats = []
    
    for epoch in range(training_config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{training_config.epochs}")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            training_config.gradient_accumulation_steps,
            training_config.max_grad_norm,
        )
        
        # Evaluate
        if val_loader:
            val_loss, val_metrics = evaluate(model, val_loader, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                output_dir = training_config.output_dir
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Model saved to {output_dir}")
        
        epoch_time = time.time() - start_time
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"Epoch time: {format_time(epoch_time)}")
        
        training_stats.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss if val_loader else None,
            "val_metrics": val_metrics if val_loader else None,
        })
    
    # Save final model and metrics
    if not val_loader:
        output_dir = training_config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    metrics_path = os.path.join(training_config.output_dir, "training_stats.json")
    save_metrics({"training_stats": training_stats}, metrics_path)
    
    logger.info("\nTraining completed!")
    return model, tokenizer, training_stats


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description="Train BERT for text classification")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./models/bert-imdb", help="Output directory")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig(
        model_name=args.model,
        num_classes=args.num_classes,
        max_length=args.max_length,
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    
    # Train
    train_classifier(
        dataset_name=args.dataset,
        model_config=model_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    main()
