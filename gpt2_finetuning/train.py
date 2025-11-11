"""
GPT-2 model training module.
"""
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config as HFConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from typing import Optional
from .config import GPT2Config
import os


class GPT2Trainer:
    """Trainer for GPT-2 models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "./models/gpt2-finetuned",
        config: Optional[GPT2Config] = None,
    ):
        """Initialize GPT-2 trainer."""
        self.config = config or GPT2Config(model_name=model_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.config.device == "cpu" and torch.cuda.is_available():
            self.config.device = "cuda"

        print(f"Loading GPT-2 model: {self.config.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.config.device}")

    def train(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ):
        """Train GPT-2 model on text data."""
        epochs = epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate

        # Load dataset
        data_files = {"train": train_file}
        if val_file:
            data_files["validation"] = val_file

        datasets = load_dataset("text", data_files=data_files)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.block_size,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if val_file else None,
            evaluation_strategy="steps" if val_file else "no",
            save_total_limit=3,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation"),
            data_collator=data_collator,
        )

        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Training completed! Model saved to {self.output_dir}")


def main():
    """Demo training script."""
    # Create sample training data
    train_text = """Once upon a time in a land far away.
The quick brown fox jumps over the lazy dog.
Machine learning is transforming the world."""
    
    with open("sample_train.txt", "w") as f:
        f.write(train_text)

    trainer = GPT2Trainer(model_name="gpt2", output_dir="./models/gpt2-demo")
    trainer.train("sample_train.txt", epochs=1, batch_size=2)


if __name__ == "__main__":
    main()
