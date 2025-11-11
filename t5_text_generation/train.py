"""
Training script for fine-tuning T5 models on custom tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5Dataset(Dataset):
    """Custom dataset for T5 training."""
    
    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 150
    ):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }


class T5Trainer:
    """Trainer for fine-tuning T5 models."""
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing trainer with {model_name} on {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
    
    def prepare_data(
        self,
        train_inputs: List[str],
        train_targets: List[str],
        val_inputs: Optional[List[str]] = None,
        val_targets: Optional[List[str]] = None,
        max_input_length: int = 512,
        max_target_length: int = 150
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_inputs: Training input texts
            train_targets: Training target texts
            val_inputs: Validation input texts
            val_targets: Validation target texts
            max_input_length: Maximum input length
            max_target_length: Maximum target length
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = T5Dataset(
            train_inputs,
            train_targets,
            self.tokenizer,
            max_input_length,
            max_target_length
        )
        
        val_dataset = None
        if val_inputs and val_targets:
            val_dataset = T5Dataset(
                val_inputs,
                val_targets,
                self.tokenizer,
                max_input_length,
                max_target_length
            )
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        warmup_steps: int = 500,
        output_dir: str = "./t5_finetuned",
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1
    ):
        """
        Train the T5 model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate scheduler
            output_dir: Directory to save model
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
                    # Evaluation
                    if val_loader and global_step % eval_steps == 0:
                        val_loss = self.evaluate(val_loader)
                        logger.info(f"Step {global_step} - Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_model(output_path / "best_model")
                            logger.info("Saved best model")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_model(output_path / f"checkpoint-{global_step}")
                        logger.info(f"Saved checkpoint at step {global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")
            
            # Evaluate at end of epoch
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
        
        # Save final model
        self.save_model(output_path / "final_model")
        logger.info("Training completed!")
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_model(self, path: Path):
        """Save model and tokenizer."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: Path):
        """Load model and tokenizer."""
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model.to(self.device)


def main():
    """Example training script."""
    # Example data
    train_inputs = [
        "summarize: The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "translate English to German: Hello, how are you?",
    ]
    train_targets = [
        "A classic pangram example.",
        "Hallo, wie geht es dir?",
    ]
    
    # Initialize trainer
    trainer = T5Trainer(model_name="t5-small")
    
    # Prepare data
    train_dataset, _ = trainer.prepare_data(train_inputs, train_targets)
    
    # Train
    trainer.train(
        train_dataset,
        epochs=3,
        batch_size=2,
        output_dir="./t5_example_model"
    )


if __name__ == "__main__":
    main()
