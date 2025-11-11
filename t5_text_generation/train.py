"""
T5 model training module.
"""
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
import os
from .config import T5Config


class T5Dataset(Dataset):
    """Dataset for T5 training."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: T5Tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dicts with 'input' and 'output' keys
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["output"]

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class T5Trainer:
    """Trainer for T5 models."""

    def __init__(
        self,
        model_name: str = "t5-small",
        output_dir: str = "./models/t5-finetuned",
        config: Optional[T5Config] = None,
    ):
        """
        Initialize trainer.

        Args:
            model_name: Name of the T5 model to use
            output_dir: Directory to save trained model
            config: T5Config object
        """
        self.config = config or T5Config(model_name=model_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Automatically detect device
        if self.config.device == "cpu" and torch.cuda.is_available():
            self.config.device = "cuda"

        print(f"Loading T5 model: {self.config.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        print(f"Model loaded on {self.config.device}")

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Train the T5 model.

        Args:
            train_data: List of dicts with 'input' and 'output' keys
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        epochs = epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate

        # Create datasets
        train_dataset = T5Dataset(
            train_data, self.tokenizer, max_length=self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Update weights
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (progress_bar.n + 1)
                    print(f"Step {global_step}, Average Loss: {avg_loss:.4f}")

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.output_dir}/checkpoint-{global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

            # Validation
            if val_data:
                val_loss = self.evaluate(val_data, batch_size)
                print(f"Validation Loss: {val_loss:.4f}")

        # Save final model
        self.save_model(self.output_dir)
        print(f"\nTraining completed! Model saved to {self.output_dir}")

    def evaluate(
        self, val_data: List[Dict[str, str]], batch_size: int = 8
    ) -> float:
        """
        Evaluate the model on validation data.

        Args:
            val_data: Validation data
            batch_size: Batch size for evaluation

        Returns:
            Average validation loss
        """
        val_dataset = T5Dataset(
            val_data, self.tokenizer, max_length=self.config.max_length
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def save_model(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")


def main():
    """Demo training script."""
    # Example training data
    train_data = [
        {
            "input": "summarize: The quick brown fox jumps over the lazy dog. This is a common English pangram.",
            "output": "A fox jumps over a dog.",
        },
        {
            "input": "paraphrase: Hello, how are you doing today?",
            "output": "Hi, how's it going today?",
        },
        {
            "input": "translate English to French: Good morning!",
            "output": "Bonjour!",
        },
    ]

    # Initialize trainer
    trainer = T5Trainer(model_name="t5-small", output_dir="./models/t5-demo")

    # Train model
    trainer.train(train_data, epochs=1, batch_size=2, learning_rate=5e-5)


if __name__ == "__main__":
    main()
